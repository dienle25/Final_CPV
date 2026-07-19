"""UI-facing adapter and presentation helpers for the classroom demo.

The Streamlit layer intentionally depends on this small contract instead of a
concrete detector implementation.  The backend is imported lazily so the UI can
still open while the inference pipeline is being installed or developed.

Preferred backend contract
--------------------------
The first available factory listed in :data:`DEFAULT_PIPELINE_FACTORIES` is
instantiated once.  ``CLASSROOM_PIPELINE_FACTORY=module.path:Factory`` can be
used to point the UI at another implementation without editing this file.

The preferred backend object exposes these non-blocking methods::

    start(config: Mapping[str, Any]) -> Any
    stop() -> None
    snapshot() -> Mapping[str, Any]
    register_student(
        *, student_id: str, full_name: str, images: Sequence[bytes]
    ) -> Mapping[str, Any] | bool
    history(*, limit: int = 5000, filters: Mapping[str, Any] | None = None)
        -> Sequence[Mapping[str, Any]]
    list_students() -> Sequence[Mapping[str, Any]]
    export_csv(...) -> bytes | str | Path      # optional
    export_xlsx(...) -> bytes | str | Path     # optional

``start`` must return quickly and run capture/inference in a worker.  A
``snapshot`` should contain ``frame`` (BGR numpy array, image bytes or path),
``channels`` (``BGR``/``RGB``), ``stats``, ``people`` and ``events``.  The
adapter accepts a few common alternate method/key names during integration, but
new pipeline code should implement the contract above directly.
"""

from __future__ import annotations

import dataclasses
import csv
import hashlib
import importlib
import inspect
import io
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from threading import RLock
from typing import Any, Protocol, runtime_checkable

import numpy as np
import pandas as pd
from PIL import Image


DEFAULT_PIPELINE_FACTORIES = (
    "src.classroom_demo.pipeline:ClassroomDemoPipeline",
    "src.classroom_demo.pipeline:ClassroomPipeline",
    "src.classroom_demo.pipeline:create_pipeline",
    "src.classroom_demo.runtime:ClassroomDemoPipeline",
    "src.classroom_demo.runtime:ClassroomRuntime",
)

STATUS_LABELS = {
    "helmet": "Đội đúng",
    "helmet_correct": "Đội đúng",
    "correct": "Đội đúng",
    "proper": "Đội đúng",
    "with_helmet": "Đội đúng",
    "incorrect_helmet": "Đội sai",
    "helmet_incorrect": "Đội sai",
    "improper_helmet": "Đội sai",
    "improper": "Đội sai",
    "wrong": "Đội sai",
    "no_helmet": "Không mũ",
    "nohelmet": "Không mũ",
    "without_helmet": "Không mũ",
    "none": "Không mũ",
    "unknown": "Chưa rõ",
    "unidentified": "Chưa rõ",
    "uncertain": "Chưa rõ",
}

STATUS_REASON_LABELS = {
    "detector_no_helmet": "Mô hình phát hiện vùng đầu không có mũ",
    "helmet_aligned_with_face": "Mũ nằm đúng vùng đầu/khuôn mặt",
    "helmet_misaligned_with_face": "Nghi đội sai: mũ lệch rõ khỏi vùng đầu",
    "helmet_alignment_borderline": "Hình học mũ chưa đủ rõ; giữ Chưa rõ",
    "helmet_misaligned_low_confidence": "Hộp mũ lệch nhưng độ tin cậy còn thấp",
    "helmet_without_face_geometry": "Có mũ nhưng chưa thấy khuôn mặt để xác minh",
    "no_helmet_without_face_geometry": "Thiếu khuôn mặt để xác minh Không mũ",
    "no_helmet_misaligned_with_face": "Hộp Không mũ không khớp vùng khuôn mặt",
    "conflicting_helmet_evidence": "Mâu thuẫn Có mũ/Không mũ; giữ Chưa rõ",
    "no_head_evidence": "Chưa có bằng chứng vùng đầu",
    "low_head_confidence": "Độ tin cậy vùng đầu quá thấp",
    "unsupported_head_label": "Nhãn vùng đầu không được hỗ trợ",
}

REQUIRED_DEMO_MODELS = (
    "models/best.onnx",
    "models/person/object_detection_nanodet_2022nov.onnx",
    "models/face/face_detection_yunet_2023mar.onnx",
    "models/face/face_recognition_sface_2021dec.onnx",
)
REFERENCE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def localize_status_reason(value: Any) -> str:
    key = str(value or "").strip()
    return STATUS_REASON_LABELS.get(key, key.replace("_", " ") if key else "")


def inspect_demo_readiness(
    project_root: str | Path,
    *,
    recommended_images: int = 5,
) -> dict[str, Any]:
    """Run fast, filesystem-only readiness checks suitable for every UI rerun."""

    root = Path(project_root)
    errors: list[str] = []
    warnings: list[str] = []
    for relative in REQUIRED_DEMO_MODELS:
        path = root / relative
        if not path.is_file() or path.stat().st_size < 100_000:
            errors.append(f"Thiếu hoặc lỗi model: {relative}")

    roster_path = root / "data/students.csv"
    active_students: list[str] = []
    if not roster_path.is_file():
        errors.append("Thiếu data/students.csv")
    else:
        try:
            with roster_path.open("r", encoding="utf-8-sig", newline="") as stream:
                reader = csv.DictReader(stream)
                fields = set(reader.fieldnames or [])
                if not {"student_id", "full_name"}.issubset(fields):
                    errors.append("Roster thiếu cột student_id hoặc full_name")
                else:
                    for row in reader:
                        student_id = str(row.get("student_id") or "").strip()
                        full_name = str(row.get("full_name") or "").strip()
                        active = str(row.get("active", "1")).strip().casefold()
                        if not student_id or not full_name:
                            errors.append("Roster có dòng thiếu MSSV hoặc họ tên")
                            continue
                        if active not in {"0", "false", "no", "inactive", "disabled"}:
                            active_students.append(student_id)
        except (OSError, csv.Error) as exc:
            errors.append(f"Không đọc được roster: {exc}")

    reference_root = root / "data/students"
    reference_counts: dict[str, int] = {}
    ready_students = 0
    for student_id in active_students:
        folder = reference_root / student_id
        count = (
            sum(
                1
                for path in folder.rglob("*")
                if path.is_file() and path.suffix.casefold() in REFERENCE_SUFFIXES
            )
            if folder.is_dir()
            else 0
        )
        reference_counts[student_id] = count
        if count >= recommended_images:
            ready_students += 1
        else:
            warnings.append(
                f"{student_id}: mới có {count} ảnh, nên có ít nhất {recommended_images}"
            )
        if count == 0:
            errors.append(f"{student_id}: chưa có ảnh tham chiếu")

    return {
        "ready": not errors,
        "errors": errors,
        "warnings": warnings,
        "student_count": len(active_students),
        "ready_student_count": ready_students,
        "reference_counts": reference_counts,
        "model_count": len(REQUIRED_DEMO_MODELS) - sum(
            message.startswith("Thiếu hoặc lỗi model") for message in errors
        ),
    }


def connection_view(
    stats: Mapping[str, Any] | None,
    *,
    monitor_running: bool,
    stale_after_seconds: float = 3.0,
    now: datetime | None = None,
) -> dict[str, str]:
    """Reduce backend stats to one truthful operator-facing connection state."""

    values = dict(stats or {})
    state = str(values.get("state") or "").strip().casefold()
    error = str(values.get("last_error") or "").strip()
    connected = bool(values.get("connected"))
    sequence = int(values.get("frame_sequence") or 0)
    reconnects = int(values.get("reconnect_count") or 0)

    if state == "failed":
        return {
            "code": "failed",
            "label": "Lỗi nguồn hoặc mô hình",
            "color": "red",
            "icon": ":material/error:",
            "message": error,
        }
    if not monitor_running and state not in {"starting", "running", "reconnecting"}:
        return {
            "code": "stopped",
            "label": "Đã dừng",
            "color": "gray",
            "icon": ":material/pause_circle:",
            "message": "",
        }
    if connected and sequence > 0:
        last_frame = values.get("last_frame_at")
        parsed: datetime | None = None
        if isinstance(last_frame, datetime):
            parsed = last_frame
        elif isinstance(last_frame, str) and last_frame.strip():
            try:
                parsed = datetime.fromisoformat(last_frame.replace("Z", "+00:00"))
            except ValueError:
                parsed = None
        current = now or datetime.now(timezone.utc)
        if parsed is not None:
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            if current.tzinfo is None:
                current = current.replace(tzinfo=timezone.utc)
            if (current - parsed.astimezone(current.tzinfo)).total_seconds() > stale_after_seconds:
                return {
                    "code": "stale",
                    "label": "Khung hình đang chậm",
                    "color": "orange",
                    "icon": ":material/slow_motion_video:",
                    "message": "Không nhận được khung hình mới đúng thời gian dự kiến.",
                }
        return {
            "code": "live",
            "label": "Camera và mô hình đang trực tiếp",
            "color": "green",
            "icon": ":material/videocam:",
            "message": "",
        }
    if state == "reconnecting" or reconnects > 0:
        return {
            "code": "reconnecting",
            "label": "Đang kết nối lại",
            "color": "orange",
            "icon": ":material/sync:",
            "message": error,
        }
    return {
        "code": "connecting",
        "label": "Đang chờ khung hình đầu tiên",
        "color": "blue",
        "icon": ":material/hourglass_top:",
        "message": error,
    }


class PipelineUnavailable(RuntimeError):
    """Raised when no compatible classroom pipeline can be imported."""


class PipelineCallError(RuntimeError):
    """Raised when a loaded pipeline fails an operation requested by the UI."""


@dataclass(slots=True)
class UISnapshot:
    """Normalized, UI-safe view of the latest pipeline state."""

    frame: Any = None
    channels: str = "BGR"
    stats: dict[str, Any] = field(default_factory=dict)
    people: list[dict[str, Any]] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    message: str = ""


@runtime_checkable
class ClassroomPipelineProtocol(Protocol):
    """Preferred runtime interface consumed by :class:`PipelineAdapter`."""

    def start(self, config: Mapping[str, Any]) -> Any: ...

    def stop(self) -> None: ...

    def snapshot(self) -> Mapping[str, Any]: ...

    def register_student(
        self,
        *,
        student_id: str,
        full_name: str,
        images: Sequence[bytes],
    ) -> Mapping[str, Any] | bool: ...

    def history(
        self,
        *,
        limit: int = 5000,
        filters: Mapping[str, Any] | None = None,
    ) -> Sequence[Mapping[str, Any]]: ...

    def list_students(self) -> Sequence[Mapping[str, Any]]: ...


def _as_mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return dict(dataclasses.asdict(value))
    if hasattr(value, "__dict__"):
        return {
            key: item
            for key, item in vars(value).items()
            if not key.startswith("_")
        }
    return {}


def _coerce_records(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, pd.DataFrame):
        return [dict(row) for row in value.to_dict(orient="records")]
    if isinstance(value, Mapping):
        for key in ("items", "records", "rows", "data", "events", "people", "students"):
            nested = value.get(key)
            if isinstance(nested, Sequence) and not isinstance(nested, (str, bytes, bytearray)):
                return [_as_mapping(item) for item in nested]
        return [dict(value)]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_as_mapping(item) for item in value]
    return []


def _first(record: Mapping[str, Any], *keys: str, default: Any = "") -> Any:
    for key in keys:
        if key in record and record[key] is not None:
            return record[key]
    return default


def _normalise_confidence(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if 1.0 < number <= 100.0:
        number /= 100.0
    return max(0.0, min(1.0, number))


def localize_status(value: Any) -> str:
    """Return a stable Vietnamese label for common backend status names."""

    if value in (None, ""):
        return "Chưa rõ"
    raw = str(value).strip()
    normalized = re.sub(r"[\s-]+", "_", raw.lower())
    return STATUS_LABELS.get(normalized, raw)


class PipelineAdapter:
    """Lazy, thread-safe bridge between Streamlit and a classroom backend."""

    def __init__(self, project_root: str | Path) -> None:
        self.project_root = Path(project_root).resolve()
        self._backend: Any | None = None
        self._backend_ref = ""
        self._running = False
        self._last_error = ""
        self._last_snapshot = UISnapshot()
        self._lock = RLock()

    @property
    def backend_loaded(self) -> bool:
        return self._backend is not None

    @property
    def backend_name(self) -> str:
        return self._backend_ref or "Chưa nạp"

    @property
    def running(self) -> bool:
        return self._running

    @property
    def last_error(self) -> str:
        return self._last_error

    @property
    def last_snapshot(self) -> UISnapshot:
        return self._last_snapshot

    @staticmethod
    def _factory_candidates() -> tuple[str, ...]:
        override = os.getenv("CLASSROOM_PIPELINE_FACTORY", "").strip()
        if override:
            return (override, *DEFAULT_PIPELINE_FACTORIES)
        return DEFAULT_PIPELINE_FACTORIES

    def _instantiate(self, factory: Any) -> Any:
        try:
            parameters = inspect.signature(factory).parameters
        except (TypeError, ValueError):
            return factory()

        root_names = {"project_root", "project_dir", "root_dir", "base_dir"}
        if any(name in parameters for name in root_names):
            name = next(name for name in root_names if name in parameters)
            return factory(**{name: self.project_root})
        return factory()

    def _ensure_backend(self) -> Any:
        with self._lock:
            if self._backend is not None:
                return self._backend

            errors: list[str] = []
            for reference in self._factory_candidates():
                if ":" not in reference:
                    errors.append(f"{reference}: định dạng phải là module:Factory")
                    continue
                module_name, attribute = reference.rsplit(":", 1)
                try:
                    module = importlib.import_module(module_name)
                    factory = getattr(module, attribute)
                    backend = self._instantiate(factory)
                except Exception as exc:
                    errors.append(f"{reference}: {type(exc).__name__}: {exc}")
                    continue

                self._backend = backend
                self._backend_ref = reference
                self._last_error = ""
                return backend

            detail = " | ".join(errors)
            message = (
                "Chưa tìm thấy pipeline lớp học tương thích. "
                "Hãy triển khai src.classroom_demo.pipeline:ClassroomDemoPipeline "
                "hoặc đặt CLASSROOM_PIPELINE_FACTORY=module:Factory."
            )
            if detail:
                message = f"{message} Chi tiết: {detail}"
            self._last_error = message
            raise PipelineUnavailable(message)

    @staticmethod
    def _method(backend: Any, *names: str) -> Any | None:
        for name in names:
            candidate = getattr(backend, name, None)
            if callable(candidate):
                return candidate
        return None

    @staticmethod
    def _value(backend: Any, *names: str) -> Any:
        for name in names:
            if not hasattr(backend, name):
                continue
            candidate = getattr(backend, name)
            return candidate() if callable(candidate) else candidate
        return None

    @staticmethod
    def _call_with_config(method: Any, config: Mapping[str, Any]) -> Any:
        try:
            parameters = inspect.signature(method).parameters
        except (TypeError, ValueError):
            return method(dict(config))

        if not parameters:
            return method()
        if any(item.kind == inspect.Parameter.VAR_KEYWORD for item in parameters.values()):
            return method(**dict(config))
        if len(parameters) == 1:
            name = next(iter(parameters))
            if name in {"config", "settings", "options", "source_config"}:
                return method(dict(config))

        kwargs = {name: config[name] for name in parameters if name in config}
        required = {
            name
            for name, item in parameters.items()
            if item.default is inspect.Parameter.empty
            and item.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        }
        if required.issubset(kwargs):
            return method(**kwargs)
        return method(dict(config))

    def start(self, config: Mapping[str, Any]) -> Any:
        """Start a non-blocking backend worker with a canonical config mapping."""

        backend = self._ensure_backend()
        method = self._method(backend, "start", "start_source", "open")
        if method is None:
            raise PipelineCallError("Pipeline không cung cấp start(config).")
        try:
            result = self._call_with_config(method, config)
            if result is False:
                raise RuntimeError("Pipeline trả về False khi khởi động")
            self._running = True
            self._last_error = ""
            return result
        except Exception as exc:
            self._running = False
            self._last_error = f"Không thể bắt đầu pipeline: {type(exc).__name__}: {exc}"
            raise PipelineCallError(self._last_error) from exc

    def stop(self) -> None:
        """Stop capture/inference if a backend has been loaded."""

        with self._lock:
            if self._backend is None:
                self._running = False
                return
            method = self._method(self._backend, "stop", "stop_source", "close_source")
            try:
                if method is not None:
                    method()
            except Exception as exc:
                self._last_error = f"Không thể dừng pipeline: {type(exc).__name__}: {exc}"
                raise PipelineCallError(self._last_error) from exc
            finally:
                self._running = False

    def reset_backend(self) -> None:
        """Drop the cached backend instance so imports can be retried after development."""

        with self._lock:
            try:
                self.stop()
            except PipelineCallError:
                pass
            close_method = self._method(self._backend, "close", "shutdown") if self._backend else None
            if close_method is not None:
                try:
                    close_method()
                except Exception:
                    pass
            self._backend = None
            self._backend_ref = ""
            self._running = False
            self._last_error = ""

    @staticmethod
    def _normalise_snapshot(value: Any) -> UISnapshot:
        if isinstance(value, UISnapshot):
            return value

        if isinstance(value, tuple):
            if len(value) == 2:
                value = {"frame": value[0], "stats": value[1]}
            elif len(value) >= 4:
                value = {
                    "frame": value[0],
                    "stats": value[1],
                    "people": value[2],
                    "events": value[3],
                }
            elif value:
                value = {"frame": value[0]}

        payload = _as_mapping(value)
        frame = _first(payload, "frame", "annotated_frame", "latest_frame", "image", default=None)
        channels = str(_first(payload, "channels", "frame_channels", default="BGR")).upper()
        if channels not in {"BGR", "RGB"}:
            channels = "BGR"
        stats = _as_mapping(_first(payload, "stats", "metrics", "kpis", default={}))
        people = _coerce_records(
            _first(payload, "people", "current_people", "persons", "tracks", default=[])
        )
        events = _coerce_records(
            _first(payload, "events", "recent_events", "violations", default=[])
        )
        updated = _first(payload, "updated_at", "timestamp", default=None)
        if isinstance(updated, str):
            try:
                updated = datetime.fromisoformat(updated.replace("Z", "+00:00"))
            except ValueError:
                updated = None
        if not isinstance(updated, datetime):
            updated = datetime.now(timezone.utc)
        return UISnapshot(
            frame=frame,
            channels=channels,
            stats=stats,
            people=people,
            events=events,
            updated_at=updated,
            message=str(_first(payload, "message", "status_message", default="")),
        )

    def snapshot(self) -> UISnapshot:
        """Return the latest non-blocking snapshot from the backend."""

        backend = self._ensure_backend()
        try:
            method = self._method(backend, "snapshot", "get_snapshot", "poll", "read_snapshot")
            if method is not None:
                raw = method()
            else:
                raw = {
                    "frame": self._value(backend, "latest_frame", "get_latest_frame"),
                    "stats": self._value(backend, "stats", "get_stats", "metrics"),
                    "people": self._value(
                        backend, "current_people", "get_current_people", "people"
                    ),
                    "events": self._value(
                        backend, "recent_events", "get_recent_events", "events"
                    ),
                }
            snapshot = self._normalise_snapshot(raw)
            self._last_snapshot = snapshot
            self._last_error = ""
            return snapshot
        except Exception as exc:
            self._last_error = f"Không đọc được dữ liệu trực tiếp: {type(exc).__name__}: {exc}"
            raise PipelineCallError(self._last_error) from exc

    @staticmethod
    def _call_registration(
        method: Any,
        *,
        student_id: str,
        full_name: str,
        images: Sequence[bytes],
    ) -> Any:
        payload = {
            "student_id": student_id,
            "mssv": student_id,
            "student_code": student_id,
            "full_name": full_name,
            "name": full_name,
            "student_name": full_name,
            "images": list(images),
            "photos": list(images),
            "image_bytes": list(images),
        }
        try:
            parameters = inspect.signature(method).parameters
        except (TypeError, ValueError):
            return method(student_id=student_id, full_name=full_name, images=list(images))

        if any(item.kind == inspect.Parameter.VAR_KEYWORD for item in parameters.values()):
            return method(student_id=student_id, full_name=full_name, images=list(images))
        if len(parameters) == 1:
            name = next(iter(parameters))
            if name in {"student", "payload", "registration", "record"}:
                return method(
                    {"student_id": student_id, "full_name": full_name, "images": list(images)}
                )
        kwargs = {name: payload[name] for name in parameters if name in payload}
        return method(**kwargs)

    def register_student(
        self,
        *,
        student_id: str,
        full_name: str,
        images: Sequence[bytes],
    ) -> dict[str, Any]:
        backend = self._ensure_backend()
        method = self._method(
            backend,
            "register_student",
            "enroll_student",
            "add_student",
            "register_identity",
        )
        if method is None:
            raise PipelineCallError("Pipeline không cung cấp register_student(...).")
        try:
            result = self._call_registration(
                method,
                student_id=student_id,
                full_name=full_name,
                images=images,
            )
            if result is False:
                raise RuntimeError("Pipeline từ chối dữ liệu đăng ký")
            self._last_error = ""
            if isinstance(result, Mapping):
                return dict(result)
            return {"success": True, "student_id": student_id, "full_name": full_name}
        except Exception as exc:
            self._last_error = f"Đăng ký sinh viên thất bại: {type(exc).__name__}: {exc}"
            raise PipelineCallError(self._last_error) from exc

    @staticmethod
    def _call_collection_method(
        method: Any,
        *,
        limit: int,
        filters: Mapping[str, Any] | None,
    ) -> Any:
        try:
            parameters = inspect.signature(method).parameters
        except (TypeError, ValueError):
            return method()
        kwargs: dict[str, Any] = {}
        if "limit" in parameters:
            kwargs["limit"] = limit
        if "filters" in parameters:
            kwargs["filters"] = filters
        if any(item.kind == inspect.Parameter.VAR_KEYWORD for item in parameters.values()):
            kwargs = {"limit": limit, "filters": filters}
        return method(**kwargs)

    def history(
        self,
        *,
        limit: int = 5000,
        filters: Mapping[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        backend = self._ensure_backend()
        method = self._method(
            backend,
            "history",
            "get_history",
            "list_events",
            "recent_events",
        )
        try:
            if method is None:
                return list(self._last_snapshot.events)[:limit]
            result = self._call_collection_method(method, limit=limit, filters=filters)
            self._last_error = ""
            return _coerce_records(result)[:limit]
        except Exception as exc:
            self._last_error = f"Không đọc được lịch sử: {type(exc).__name__}: {exc}"
            raise PipelineCallError(self._last_error) from exc

    def list_students(self, *, limit: int = 1000) -> list[dict[str, Any]]:
        backend = self._ensure_backend()
        method = self._method(
            backend,
            "list_students",
            "get_students",
            "registered_students",
            "roster",
        )
        if method is None:
            return []
        try:
            result = self._call_collection_method(method, limit=limit, filters=None)
            self._last_error = ""
            return _coerce_records(result)[:limit]
        except Exception as exc:
            self._last_error = f"Không đọc được danh sách sinh viên: {type(exc).__name__}: {exc}"
            raise PipelineCallError(self._last_error) from exc


def materialize_uploaded_video(uploaded_file: Any, directory: str | Path) -> Path:
    """Persist one Streamlit upload under a deterministic, safe file name."""

    data = uploaded_file.getvalue()
    original_name = str(getattr(uploaded_file, "name", "video.mp4"))
    suffix = Path(original_name).suffix.lower()
    if suffix not in {".mp4", ".avi", ".mov", ".mkv", ".webm"}:
        suffix = ".mp4"
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", Path(original_name).stem).strip("_") or "video"
    digest = hashlib.sha256(data).hexdigest()[:12]
    target_dir = Path(directory)
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{stem[:48]}_{digest}{suffix}"
    if not target.exists() or target.stat().st_size != len(data):
        target.write_bytes(data)
    return target


def frame_to_jpeg_bytes(frame: Any, *, channels: str = "BGR", quality: int = 92) -> bytes:
    """Encode a current UI frame for face enrollment without OpenCV coupling."""

    if frame is None:
        raise ValueError("Chưa có khung hình hiện tại")
    if isinstance(frame, (bytes, bytearray, memoryview)):
        return bytes(frame)
    if isinstance(frame, (str, Path)):
        return Path(frame).read_bytes()
    if isinstance(frame, Image.Image):
        image = frame.convert("RGB")
    else:
        array = np.asarray(frame)
        if array.ndim == 2:
            image = Image.fromarray(array.astype(np.uint8), mode="L").convert("RGB")
        elif array.ndim == 3 and array.shape[2] in {3, 4}:
            array = np.clip(array, 0, 255).astype(np.uint8)
            if channels.upper() == "BGR":
                if array.shape[2] == 3:
                    array = array[:, :, ::-1]
                else:
                    array = array[:, :, [2, 1, 0, 3]]
            image = Image.fromarray(array).convert("RGB")
        else:
            raise ValueError("Định dạng khung hình không được hỗ trợ")
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=max(60, min(100, int(quality))))
    return buffer.getvalue()


def people_dataframe(records: Any) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for record in _coerce_records(records):
        rows.append(
            {
                "MSSV": _first(record, "student_id", "mssv", "student_code", "MSSV"),
                "Họ và tên": _first(record, "full_name", "name", "student_name", "Họ và tên"),
                "Trạng thái mũ": localize_status(
                    _first(record, "helmet_status", "status", "violation_type", "Trạng thái mũ")
                ),
                "Tin cậy khuôn mặt": _normalise_confidence(
                    _first(
                        record,
                        "face_confidence",
                        "identity_confidence",
                        "identity_similarity",
                        "face_score",
                        default=None,
                    )
                ),
                "Tin cậy detector mũ": _normalise_confidence(
                    _first(record, "helmet_confidence", "status_confidence", "confidence", default=None)
                ),
                "Giải thích": localize_status_reason(
                    _first(record, "status_reason", "reason", default="")
                ),
                "Track": _first(record, "track_id", "person_id", "track", default=""),
                "Cập nhật": _first(record, "last_seen", "updated_at", "timestamp", default=""),
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "MSSV",
            "Họ và tên",
            "Trạng thái mũ",
            "Tin cậy khuôn mặt",
            "Tin cậy detector mũ",
            "Giải thích",
            "Track",
            "Cập nhật",
        ],
    )


def events_dataframe(records: Any) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for record in _coerce_records(records):
        rows.append(
            {
                "Mã sự kiện": _first(record, "event_id", "id", "Mã sự kiện"),
                "Thời gian": _first(
                    record,
                    "detected_at",
                    "detected_at_utc",
                    "timestamp",
                    "created_at",
                    "Thời gian",
                ),
                "MSSV": _first(record, "student_id", "mssv", "student_code", "MSSV"),
                "Họ và tên": _first(record, "full_name", "name", "student_name", "Họ và tên"),
                "Trạng thái mũ": localize_status(
                    _first(record, "helmet_status", "status", "violation_type", "Trạng thái mũ")
                ),
                "Tin cậy": _normalise_confidence(
                    _first(record, "helmet_confidence", "confidence", "score", default=None)
                ),
                "Ảnh minh chứng": _first(
                    record, "evidence_path", "image_path", "snapshot_path", "Ảnh minh chứng"
                ),
                "Nguồn": _first(record, "source_name", "source", "Nguồn"),
            }
        )
    frame = pd.DataFrame(
        rows,
        columns=[
            "Mã sự kiện",
            "Thời gian",
            "MSSV",
            "Họ và tên",
            "Trạng thái mũ",
            "Tin cậy",
            "Ảnh minh chứng",
            "Nguồn",
        ],
    )
    if not frame.empty:
        parsed = pd.to_datetime(frame["Thời gian"], errors="coerce", utc=True)
        vietnam_tz = timezone(timedelta(hours=7))
        frame["Thời gian"] = parsed.dt.tz_convert(vietnam_tz).dt.tz_localize(None)
    return frame


def students_dataframe(records: Any) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for record in _coerce_records(records):
        rows.append(
            {
                "MSSV": _first(record, "student_id", "mssv", "student_code", "MSSV"),
                "Họ và tên": _first(record, "full_name", "name", "student_name", "Họ và tên"),
                "Số ảnh": _first(record, "image_count", "photo_count", "samples", default=""),
                "Cập nhật": _first(record, "updated_at", "created_at", "timestamp", default=""),
            }
        )
    return pd.DataFrame(rows, columns=["MSSV", "Họ và tên", "Số ảnh", "Cập nhật"])


def filter_events(
    frame: pd.DataFrame,
    *,
    student_query: str = "",
    statuses: Sequence[str] = (),
    start_date: date | None = None,
    end_date: date | None = None,
) -> pd.DataFrame:
    """Apply cheap report filters after the backend history has been loaded."""

    filtered = frame.copy()
    query = student_query.strip().casefold()
    if query and not filtered.empty:
        mask = (
            filtered["MSSV"].fillna("").astype(str).str.casefold().str.contains(query, regex=False)
            | filtered["Họ và tên"]
            .fillna("")
            .astype(str)
            .str.casefold()
            .str.contains(query, regex=False)
        )
        filtered = filtered[mask]
    if statuses and not filtered.empty:
        filtered = filtered[filtered["Trạng thái mũ"].isin(list(statuses))]
    if not filtered.empty and pd.api.types.is_datetime64_any_dtype(filtered["Thời gian"]):
        valid_times = filtered["Thời gian"].notna()
        if start_date is not None:
            valid_times &= filtered["Thời gian"].dt.date >= start_date
        if end_date is not None:
            valid_times &= filtered["Thời gian"].dt.date <= end_date
        filtered = filtered[valid_times]
    return filtered.reset_index(drop=True)


def dataframe_to_csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")


def dataframe_to_xlsx_bytes(frame: pd.DataFrame) -> bytes:
    """Build an XLSX report, trying either commonly available Pandas engine."""

    export_frame = frame.copy()
    for column in export_frame.select_dtypes(include=["datetimetz"]).columns:
        export_frame[column] = export_frame[column].dt.tz_localize(None)

    failures: list[str] = []
    for engine in ("xlsxwriter", "openpyxl"):
        buffer = io.BytesIO()
        try:
            with pd.ExcelWriter(buffer, engine=engine) as writer:
                export_frame.to_excel(writer, sheet_name="Lịch sử", index=False)
            return buffer.getvalue()
        except (ImportError, ModuleNotFoundError) as exc:
            failures.append(f"{engine}: {exc}")
    detail = " | ".join(failures) or "không có engine tương thích"
    raise RuntimeError(f"Không thể tạo XLSX ({detail})")


__all__ = [
    "ClassroomPipelineProtocol",
    "PipelineAdapter",
    "PipelineCallError",
    "PipelineUnavailable",
    "STATUS_LABELS",
    "UISnapshot",
    "dataframe_to_csv_bytes",
    "dataframe_to_xlsx_bytes",
    "events_dataframe",
    "filter_events",
    "frame_to_jpeg_bytes",
    "connection_view",
    "inspect_demo_readiness",
    "localize_status",
    "localize_status_reason",
    "materialize_uploaded_video",
    "people_dataframe",
    "students_dataframe",
]
