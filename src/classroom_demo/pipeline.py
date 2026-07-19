"""UI-independent background pipeline for the classroom demonstration."""

from __future__ import annotations

import copy
import dataclasses
import inspect
import threading
import time
import unicodedata
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Protocol

from .sources import CaptureWorker, FramePacket, SourceSpec, parse_source
from .storage import EventStore


class PipelineState(str, Enum):
    IDLE = "idle"
    STARTING = "starting"
    RUNNING = "running"
    RECONNECTING = "reconnecting"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"


@dataclass(slots=True)
class ProcessingResult:
    """Normalized output expected from a detector/face/tracking adapter."""

    frame: Any
    people: list[dict[str, Any]] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)


# Name commonly used by detector adapters.
PipelineResult = ProcessingResult


class CaptureWorkerLike(Protocol):
    source: SourceSpec

    @property
    def running(self) -> bool:
        ...

    @property
    def connected(self) -> bool:
        ...

    @property
    def last_error(self) -> str | None:
        ...

    def start(self) -> Any:
        ...

    def stop(self, timeout: float | None = 5.0) -> None:
        ...

    def get_latest(self, timeout: float | None = None) -> FramePacket | None:
        ...


def _config_value(config: Any, *names: str, default: Any = None) -> Any:
    if config is None:
        return default
    if isinstance(config, Mapping):
        for name in names:
            if name in config:
                return config[name]
        return default
    for name in names:
        if hasattr(config, name):
            return getattr(config, name)
    return default


def _object_to_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if dataclasses.is_dataclass(value):
        return dataclasses.asdict(value)
    converter = getattr(value, "to_dict", None)
    if callable(converter):
        converted = converter()
        if isinstance(converted, Mapping):
            return dict(converted)
    attributes = getattr(value, "__dict__", None)
    if isinstance(attributes, dict):
        return dict(attributes)
    raise TypeError(f"Expected a mapping-like result, received {type(value).__name__}")


def _items_to_dicts(values: Any) -> list[dict[str, Any]]:
    if values is None:
        return []
    if isinstance(values, Mapping):
        return [dict(values)]
    if isinstance(values, (str, bytes, bytearray)):
        raise TypeError("People/events must be mapping objects, not text or bytes")
    return [_object_to_dict(value) for value in values]


def _normalized_status(status: Any) -> str:
    text = unicodedata.normalize("NFKD", str(status or "unknown").casefold())
    # Vietnamese đ/Đ does not decompose under NFKD, so transliterate it
    # explicitly before matching the ASCII status aliases below.
    text = text.replace("đ", "d")
    return "".join(character for character in text if not unicodedata.combining(character))


def _status_style(status: Any) -> tuple[tuple[int, int, int], str]:
    normalized = _normalized_status(status).replace("-", "_").replace(" ", "_")
    if normalized in {
        "helmet",
        "helmet_correct",
        "correct",
        "proper",
        "with_helmet",
        "doi_dung",
        "doi_mu_dung_cach",
        "co_mu",
    }:
        return (30, 180, 75), "Đội đúng"
    if normalized in {
        "improper",
        "improper_helmet",
        "incorrect",
        "incorrect_helmet",
        "wrong",
        "wrong_helmet",
        "helmet_incorrect",
        "doi_sai",
        "doi_khong_dung_cach",
    }:
        return (245, 155, 35), "Đội sai"
    if normalized in {
        "no_helmet",
        "nohelmet",
        "without_helmet",
        "khong_mu",
        "khong_doi_mu",
    }:
        return (220, 50, 55), "Không mũ"
    return (125, 135, 150), str(status or "Chưa xác định")


def _box_from(person: Mapping[str, Any], *keys: str) -> tuple[int, int, int, int] | None:
    value: Any = None
    for key in keys:
        if person.get(key) is not None:
            value = person[key]
            break
    if value is None:
        return None
    if isinstance(value, Mapping):
        value = [value.get("x1"), value.get("y1"), value.get("x2"), value.get("y2")]
    try:
        numbers = tuple(int(round(float(item))) for item in value)
    except (TypeError, ValueError):
        return None
    if len(numbers) != 4:
        return None
    x1, y1, x2, y2 = numbers
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _font_candidates(explicit: str | Path | None = None) -> list[Path]:
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit))
    candidates.extend(
        [
            Path(r"C:\Windows\Fonts\segoeui.ttf"),
            Path(r"C:\Windows\Fonts\arial.ttf"),
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
            Path("/usr/share/fonts/dejavu/DejaVuSans.ttf"),
        ]
    )
    return candidates


def annotate_people(
    frame: Any,
    people: Iterable[Mapping[str, Any]],
    *,
    font_path: str | Path | None = None,
    font_size: int = 20,
) -> Any:
    """Draw person/face/head boxes and a Unicode ``MSSV – trạng thái`` label.

    Pillow is preferred because OpenCV's Hershey fonts cannot render Vietnamese.
    The function falls back to OpenCV when Pillow is unavailable.  Input/output
    use OpenCV's BGR channel order.
    """

    people_list = [dict(person) for person in people]
    if frame is None or not people_list:
        return frame.copy() if hasattr(frame, "copy") else frame

    try:
        import numpy as np
        from PIL import Image, ImageDraw, ImageFont

        array = np.asarray(frame)
        if array.ndim != 3 or array.shape[2] < 3:
            return frame.copy() if hasattr(frame, "copy") else frame
        rgb = array[:, :, :3][:, :, ::-1]
        image = Image.fromarray(rgb)
        draw = ImageDraw.Draw(image)
        font = None
        for candidate in _font_candidates(font_path):
            if candidate.exists():
                try:
                    font = ImageFont.truetype(str(candidate), max(12, int(font_size)))
                    break
                except OSError:
                    continue
        if font is None:
            font = ImageFont.load_default()

        width, height = image.size
        for person in people_list:
            color, display_status = _status_style(
                person.get("status", person.get("helmet_status", "unknown"))
            )
            person_box = _box_from(person, "bbox", "person_bbox", "rider_bbox", "body_bbox")
            face_box = _box_from(person, "face_bbox", "face_box")
            head_box = _box_from(person, "head_bbox", "helmet_bbox", "head_box")
            for box, box_color, line_width in (
                (person_box, color, 3),
                (head_box, color, 2),
                (face_box, (40, 125, 245), 2),
            ):
                if box is None:
                    continue
                x1, y1, x2, y2 = box
                clipped = (
                    max(0, min(x1, width - 1)),
                    max(0, min(y1, height - 1)),
                    max(1, min(x2, width)),
                    max(1, min(y2, height)),
                )
                draw.rectangle(clipped, outline=box_color, width=line_width)

            anchor = person_box or face_box or head_box
            if anchor is None:
                continue
            student_id = str(
                person.get("student_id", person.get("mssv", person.get("id", ""))) or "Chưa xác định"
            )
            label = f"{student_id} – {display_status}"
            x1, y1, _x2, _y2 = anchor
            try:
                left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
                text_width, text_height = right - left, bottom - top
            except AttributeError:
                text_width, text_height = draw.textsize(label, font=font)
            label_x = max(0, min(x1, width - text_width - 10))
            label_y = max(0, y1 - text_height - 12)
            draw.rounded_rectangle(
                (label_x, label_y, label_x + text_width + 10, label_y + text_height + 8),
                radius=4,
                fill=color,
            )
            draw.text((label_x + 5, label_y + 3), label, fill=(255, 255, 255), font=font)

        return np.asarray(image)[:, :, ::-1].copy()
    except (ImportError, OSError, ValueError, TypeError):
        return _annotate_people_cv2(frame, people_list)


def _annotate_people_cv2(frame: Any, people: Iterable[Mapping[str, Any]]) -> Any:
    try:
        import cv2
    except ImportError:
        return frame.copy() if hasattr(frame, "copy") else frame

    annotated = frame.copy() if hasattr(frame, "copy") else frame
    for person in people:
        rgb_color, display_status = _status_style(
            person.get("status", person.get("helmet_status", "unknown"))
        )
        color = tuple(reversed(rgb_color))
        box = _box_from(person, "bbox", "person_bbox", "rider_bbox", "body_bbox")
        if box is None:
            continue
        x1, y1, x2, y2 = box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        student_id = str(person.get("student_id", person.get("mssv", "UNKNOWN")) or "UNKNOWN")
        ascii_status = unicodedata.normalize("NFKD", display_status).encode("ascii", "ignore").decode()
        cv2.putText(
            annotated,
            f"{student_id} - {ascii_status}",
            (x1, max(18, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )
    return annotated


class DemoPipeline:
    """Coordinate capture, injected inference, annotation and persistence.

    The processor may be a callable or expose ``process_frame``/``process``. It
    can return :class:`ProcessingResult`, a mapping, ``(frame, people, events)``,
    or a bare annotated frame.  Optional context keywords are supplied only when
    accepted by the processor's signature.
    """

    def __init__(
        self,
        source: SourceSpec | str | int | None = None,
        *,
        processor: Any | None = None,
        capture_worker: CaptureWorkerLike | None = None,
        store: EventStore | None = None,
        processor_factory: Callable[[Any], Any] | None = None,
        capture_worker_factory: Callable[[SourceSpec], CaptureWorkerLike] | None = None,
        capture_factory: Callable[[str | int], Any] | None = None,
        store_factory: Callable[[Any], EventStore] | None = None,
        annotator: Callable[[Any, Iterable[Mapping[str, Any]]], Any] | None = None,
        recent_limit: int = 100,
        poll_timeout: float = 0.25,
        max_consecutive_errors: int = 1,
        save_event_snapshots: bool = True,
        name: str = "classroom-demo-pipeline",
    ) -> None:
        if recent_limit < 1:
            raise ValueError("recent_limit must be at least 1")
        if poll_timeout <= 0:
            raise ValueError("poll_timeout must be positive")
        if max_consecutive_errors < 1:
            raise ValueError("max_consecutive_errors must be at least 1")

        self._source_value = source
        self._processor = processor
        self._capture = capture_worker
        self._store = store
        self._processor_factory = processor_factory
        self._capture_worker_factory = capture_worker_factory
        self._low_level_capture_factory = capture_factory
        self._store_factory = store_factory
        self._annotator = annotator or annotate_people
        self._recent_limit = int(recent_limit)
        self._poll_timeout = float(poll_timeout)
        self._max_consecutive_errors = int(max_consecutive_errors)
        self._save_event_snapshots = bool(save_event_snapshots)
        self._name = name

        self._owns_capture = capture_worker is None
        self._owns_store = store is None
        self._owns_processor = processor is None
        self._session_id = ""
        self._config: Any = None
        self._font_path: str | Path | None = None
        self._font_size = 20
        self._registered_students: dict[str, dict[str, Any]] = {}

        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._state = PipelineState.IDLE
        self._error: str | None = None
        self._latest_frame: Any | None = None
        self._latest_frame_sequence = 0
        self._current_people: list[dict[str, Any]] = []
        self._recent_events: deque[dict[str, Any]] = deque(maxlen=self._recent_limit)
        self._processor_stats: dict[str, Any] = {}
        self._frames_processed = 0
        self._event_count = 0
        self._processing_fps = 0.0
        self._started_monotonic = 0.0
        self._started_at = ""
        self._stopped_at = ""
        self._last_frame_at = ""

    @property
    def running(self) -> bool:
        thread = self._thread
        return bool(thread and thread.is_alive())

    @property
    def state(self) -> PipelineState:
        with self._lock:
            return self._state

    @property
    def session_id(self) -> str:
        with self._lock:
            return self._session_id

    def start(self, config: Any = None) -> "DemoPipeline":
        """Start in the background; ``config`` may be a mapping or dataclass."""

        with self._lock:
            if self.running:
                return self
            self._config = config
            self._configure(config)
            if self._capture is None:
                raise RuntimeError("A video source/capture worker is required")
            if self._processor is None:
                raise RuntimeError(
                    "A frame processor is required. Inject processor=... or processor_factory=..."
                )

            self._sync_registered_students()
            self._stop_event.clear()
            self._state = PipelineState.STARTING
            self._error = None
            self._latest_frame = None
            self._latest_frame_sequence = 0
            self._current_people = []
            self._recent_events.clear()
            self._processor_stats = {}
            self._frames_processed = 0
            self._processing_fps = 0.0
            self._started_monotonic = time.monotonic()
            self._started_at = _utc_now_text()
            self._stopped_at = ""
            self._last_frame_at = ""
            if self._store is not None:
                try:
                    previous = self._store.recent(
                        self._recent_limit, session_id=self._session_id
                    )
                    self._recent_events.extend(reversed(previous))
                    self._event_count = self._store.count({"session_id": self._session_id})
                except Exception:
                    self._event_count = 0
            else:
                self._event_count = 0

            self._thread = threading.Thread(target=self._run, name=self._name, daemon=True)
            self._thread.start()
        return self

    def stop(self, timeout: float | None = 10.0) -> None:
        with self._lock:
            if self._state not in {PipelineState.STOPPED, PipelineState.FAILED}:
                self._state = PipelineState.STOPPING
            self._stop_event.set()
        capture = self._capture
        if capture is not None:
            capture.stop(timeout=min(timeout, 5.0) if timeout is not None else 5.0)
        thread = self._thread
        if thread and thread is not threading.current_thread():
            thread.join(timeout=timeout)
        with self._lock:
            if not self.running and self._state != PipelineState.FAILED:
                self._state = PipelineState.STOPPED
                self._stopped_at = self._stopped_at or _utc_now_text()

    def close(self) -> None:
        self.stop()
        if self._owns_store and self._store is not None:
            self._store.close()
        if self._owns_processor and self._processor is not None:
            closer = getattr(self._processor, "close", None)
            if callable(closer):
                closer()

    def snapshot(self, *, copy_frame: bool = True) -> dict[str, Any]:
        """Return one atomic UI snapshot: frame, stats, people and events."""

        with self._lock:
            frame = self._latest_frame
            if copy_frame and frame is not None and hasattr(frame, "copy"):
                frame = frame.copy()
            people = copy.deepcopy(self._current_people)
            events = copy.deepcopy(list(reversed(self._recent_events)))
            capture = self._capture
            capture_error = getattr(capture, "last_error", None) if capture else None
            connected = bool(getattr(capture, "connected", False)) if capture else False
            reconnect_count = int(getattr(capture, "reconnect_count", 0)) if capture else 0
            source = getattr(capture, "source", None) if capture else None
            source_label = getattr(source, "label", str(self._source_value or ""))
            stats = {
                "state": self._state.value,
                "running": self.running,
                "connected": connected,
                "session_id": self._session_id,
                "source": source_label,
                "frames_processed": self._frames_processed,
                "frame_sequence": self._latest_frame_sequence,
                "processing_fps": round(self._processing_fps, 2),
                "people_count": len(self._current_people),
                "event_count": self._event_count,
                "recent_event_count": len(self._recent_events),
                "reconnect_count": reconnect_count,
                "last_error": self._error or capture_error,
                "started_at": self._started_at,
                "stopped_at": self._stopped_at,
                "last_frame_at": self._last_frame_at,
                **copy.deepcopy(self._processor_stats),
            }
            return {
                "frame": frame,
                "channels": "BGR",
                "stats": stats,
                "people": people,
                "events": events,
                "updated_at": self._last_frame_at or _utc_now_text(),
                "message": self._error or "",
            }

    def latest_annotated_frame(self, *, copy_frame: bool = True) -> Any | None:
        return self.snapshot(copy_frame=copy_frame)["frame"]

    def current_people(self) -> list[dict[str, Any]]:
        return self.snapshot(copy_frame=False)["people"]

    def recent_events(self) -> list[dict[str, Any]]:
        return self.snapshot(copy_frame=False)["events"]

    def register_student(
        self,
        student_id: str,
        full_name: str,
        image_bytes: bytes | bytearray | memoryview | Iterable[Any] | None = None,
        *,
        images: Iterable[Any] | None = None,
    ) -> Any:
        """Register now when supported, or queue registration until start()."""

        student_id = str(student_id).strip()
        full_name = str(full_name).strip()
        if not student_id or not full_name:
            raise ValueError("student_id and full_name must not be empty")
        supplied_images = images if images is not None else image_bytes
        if isinstance(supplied_images, (bytes, bytearray, memoryview)):
            image_list = [bytes(supplied_images)]
        elif supplied_images is None:
            image_list = []
        else:
            image_list = list(supplied_images)
        if not image_list:
            raise ValueError("At least one student image is required")

        registration = {
            "student_id": student_id,
            "full_name": full_name,
            "images": image_list,
        }
        with self._lock:
            self._registered_students[student_id] = registration
            processor = self._processor
        if processor is None:
            return {"student_id": student_id, "full_name": full_name, "queued": True}
        result = self._register_with_processor(processor, registration)
        return result if result is not None else {
            "student_id": student_id,
            "full_name": full_name,
            "queued": False,
        }

    def history(
        self,
        filters: Mapping[str, Any] | None = None,
        *,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        if self._store is None:
            events = self.recent_events()
            return events if limit is None else events[: max(0, int(limit))]
        return self._store.history(filters, limit=limit, newest_first=True)

    def list_students(self, *, limit: int = 1000) -> list[dict[str, Any]]:
        """Return the in-memory enrollment roster in a UI-friendly shape."""

        with self._lock:
            registrations = list(self._registered_students.values())[: max(0, int(limit))]
        return [
            {
                "student_id": item["student_id"],
                "full_name": item["full_name"],
                "name": item["full_name"],
                "image_count": len(item["images"]),
            }
            for item in registrations
        ]

    def export_csv(
        self,
        path: str | Path | None = None,
        filters: Mapping[str, Any] | None = None,
    ) -> Path:
        if self._store is None:
            raise RuntimeError("CSV export requires an EventStore")
        return self._store.export_csv(path, filters=filters)

    def export_xlsx(
        self,
        path: str | Path | None = None,
        filters: Mapping[str, Any] | None = None,
    ) -> Path:
        if self._store is None:
            raise RuntimeError("XLSX export requires an EventStore")
        return self._store.export_xlsx(path, filters=filters)

    def wait_until_ready(self, timeout: float = 10.0) -> bool:
        deadline = time.monotonic() + max(0.0, timeout)
        while time.monotonic() < deadline:
            state = self.state
            if state == PipelineState.RUNNING and self.snapshot(copy_frame=False)["frame"] is not None:
                return True
            if state == PipelineState.FAILED:
                return False
            time.sleep(0.01)
        return False

    def _configure(self, config: Any) -> None:
        session_id = _config_value(config, "session_id", "session", default="")
        if not session_id:
            session_id = datetime.now().strftime("classroom_%Y%m%d_%H%M%S_%f")
        self._session_id = str(session_id)
        self._font_path = _config_value(config, "font_path", default=self._font_path)
        self._font_size = int(_config_value(config, "font_size", default=self._font_size))
        self._save_event_snapshots = bool(
            _config_value(
                config,
                "save_event_snapshots",
                "save_evidence",
                default=self._save_event_snapshots,
            )
        )

        configured_processor = _config_value(config, "processor", default=None)
        if configured_processor is not None:
            self._processor = configured_processor
            self._owns_processor = False
        elif self._processor is None and self._processor_factory is not None:
            self._processor = self._processor_factory(config)
            self._owns_processor = True

        configured_store = _config_value(config, "store", default=None)
        if configured_store is not None:
            self._store = configured_store
            self._owns_store = False
        elif self._store is None:
            if self._store_factory is not None:
                self._store = self._store_factory(config)
            else:
                output_dir = Path(
                    _config_value(config, "output_dir", default="outputs/classroom")
                )
                cooldown = float(
                    _config_value(config, "cooldown_seconds", "event_cooldown", default=5.0)
                )
                self._store = EventStore(
                    _config_value(config, "db_path", default=output_dir / "db" / "events.db"),
                    snapshot_dir=_config_value(
                        config, "snapshot_dir", default=output_dir / "snapshots"
                    ),
                    export_dir=_config_value(
                        config, "export_dir", default=output_dir / "exports"
                    ),
                    cooldown_seconds=cooldown,
                )
            self._owns_store = True

        source_value = _config_value(
            config,
            "source",
            "video_source",
            "camera_url",
            "ip_webcam_url",
            default=self._source_value,
        )
        configured_capture = _config_value(config, "capture_worker", default=None)
        if configured_capture is not None:
            self._capture = configured_capture
            self._owns_capture = False
        elif self._capture is None:
            if source_value is None:
                return
            spec = parse_source(
                source_value,
                ip_webcam=_config_value(config, "ip_webcam", default=None),
            )
            self._source_value = spec
            if self._capture_worker_factory is not None:
                self._capture = self._capture_worker_factory(spec)
            else:
                self._capture = CaptureWorker(
                    spec,
                    capture_factory=self._low_level_capture_factory,
                    initial_backoff=float(
                        _config_value(config, "initial_backoff", default=0.25)
                    ),
                    max_backoff=float(_config_value(config, "max_backoff", default=5.0)),
                    startup_timeout=float(
                        _config_value(config, "source_startup_timeout", default=15.0)
                    ),
                )
            self._owns_capture = True

    def _run(self) -> None:
        assert self._capture is not None
        consecutive_errors = 0
        last_sequence = 0
        try:
            self._capture.start()
            with self._lock:
                self._state = PipelineState.RUNNING

            while not self._stop_event.is_set():
                packet = self._capture.get_latest(timeout=self._poll_timeout)
                if packet is None:
                    if self._stop_event.is_set():
                        break
                    if not self._capture.running:
                        if self._frames_processed == 0 and self._capture.last_error:
                            raise RuntimeError(self._capture.last_error)
                        break
                    with self._lock:
                        self._state = (
                            PipelineState.RUNNING
                            if self._capture.connected
                            else PipelineState.RECONNECTING
                        )
                    continue
                if packet.sequence <= last_sequence:
                    continue
                last_sequence = packet.sequence

                try:
                    result = self._process_packet(packet)
                    consecutive_errors = 0
                except Exception as exc:
                    consecutive_errors += 1
                    with self._lock:
                        self._error = f"{type(exc).__name__}: {exc}"
                    if consecutive_errors >= self._max_consecutive_errors:
                        raise
                    continue

                annotated = self._annotator_with_options(result.frame, result.people)
                stored_events = self._persist_events(result.events, annotated, packet)
                elapsed = max(time.monotonic() - self._started_monotonic, 1e-6)
                with self._lock:
                    self._frames_processed += 1
                    self._processing_fps = self._frames_processed / elapsed
                    self._latest_frame = annotated
                    self._latest_frame_sequence = packet.sequence
                    self._current_people = copy.deepcopy(result.people)
                    self._processor_stats = copy.deepcopy(result.stats)
                    self._last_frame_at = datetime.fromtimestamp(
                        packet.captured_at, tz=timezone.utc
                    ).isoformat(timespec="milliseconds").replace("+00:00", "Z")
                    for event in stored_events:
                        self._recent_events.append(copy.deepcopy(event))
                    self._event_count += len(stored_events)
                    self._error = None
                    self._state = PipelineState.RUNNING
        except Exception as exc:
            with self._lock:
                self._error = f"{type(exc).__name__}: {exc}"
                self._state = PipelineState.FAILED
        finally:
            self._stop_event.set()
            self._capture.stop(timeout=5.0)
            with self._lock:
                if self._state != PipelineState.FAILED:
                    self._state = PipelineState.STOPPED
                self._stopped_at = _utc_now_text()

    def _process_packet(self, packet: FramePacket) -> ProcessingResult:
        processor = self._processor
        if processor is None:
            raise RuntimeError("Frame processor is unavailable")
        target = getattr(processor, "process_frame", None)
        if not callable(target):
            target = getattr(processor, "process", None)
        if not callable(target) and callable(processor):
            target = processor
        if not callable(target):
            raise TypeError("Processor must be callable or expose process_frame/process")

        context = {
            "frame_index": self._frames_processed,
            "sequence": packet.sequence,
            "timestamp": datetime.fromtimestamp(packet.captured_at, tz=timezone.utc),
            "session_id": self._session_id,
            "packet": packet,
        }
        raw = _call_with_supported_keywords(target, packet.frame, context)
        return self._normalize_result(raw, packet.frame)

    @staticmethod
    def _normalize_result(raw: Any, original_frame: Any) -> ProcessingResult:
        if isinstance(raw, ProcessingResult):
            return raw
        if raw is None:
            return ProcessingResult(frame=original_frame)
        if isinstance(raw, Mapping):
            frame = raw.get("annotated_frame", raw.get("frame", original_frame))
            people = raw.get("people", raw.get("current_people", []))
            events = raw.get("events", raw.get("new_events", []))
            stats = raw.get("stats", raw.get("metadata", {}))
            return ProcessingResult(
                frame=frame,
                people=_items_to_dicts(people),
                events=_items_to_dicts(events),
                stats=dict(stats or {}),
            )
        if isinstance(raw, tuple):
            if len(raw) == 3:
                frame, people, events = raw
            elif len(raw) == 2:
                frame, people = raw
                events = []
            elif len(raw) == 1:
                frame, people, events = raw[0], [], []
            else:
                raise ValueError("Processor tuple result must contain one to three values")
            return ProcessingResult(
                frame=frame,
                people=_items_to_dicts(people),
                events=_items_to_dicts(events),
            )
        return ProcessingResult(frame=raw)

    def _annotator_with_options(
        self, frame: Any, people: Iterable[Mapping[str, Any]]
    ) -> Any:
        if self._annotator is annotate_people:
            return annotate_people(
                frame,
                people,
                font_path=self._font_path,
                font_size=self._font_size,
            )
        return self._annotator(frame, people)

    def _persist_events(
        self,
        events: Iterable[Mapping[str, Any]],
        annotated_frame: Any,
        packet: FramePacket,
    ) -> list[dict[str, Any]]:
        stored: list[dict[str, Any]] = []
        for incoming in events:
            event = dict(incoming)
            if event.get("persisted") or self._store is None:
                normalized = self._normalize_unstored_event(event, packet)
                stored.append(normalized)
                continue

            student_id = str(event.get("student_id", event.get("mssv", "")) or "")
            name = str(event.get("name", event.get("full_name", "")) or "")
            status = str(event.get("status", event.get("helmet_status", "unknown")) or "unknown")
            confidence_value = event.get(
                "confidence", event.get("helmet_confidence", event.get("score", 0.0))
            )
            try:
                confidence = max(0.0, min(1.0, float(confidence_value or 0.0)))
            except (TypeError, ValueError):
                confidence = 0.0
            event_time = event.get(
                "timestamp", datetime.fromtimestamp(packet.captured_at, tz=timezone.utc)
            )
            snapshot = None
            is_violation = _status_style(status)[1] in {"Đội sai", "Không mũ"}
            if self._save_event_snapshots and is_violation:
                snapshot = event.get("snapshot", event.get("snapshot_frame"))
                if snapshot is None:
                    snapshot = (
                        annotated_frame.copy()
                        if hasattr(annotated_frame, "copy")
                        else annotated_frame
                    )
            dedupe_key = event.get(
                "dedupe_key",
                event.get("track_id", event.get("person_id", event.get("face_id"))),
            )
            metadata = dict(event.get("metadata") or {})
            for key in ("track_id", "person_id", "face_id", "source"):
                if key in event and key not in metadata:
                    metadata[key] = event[key]
            metadata.setdefault("source", packet.source.label)
            inserted = self._store.record_event(
                session_id=str(event.get("session_id") or self._session_id),
                student_id=student_id,
                name=name,
                status=status,
                confidence=confidence,
                timestamp=event_time,
                snapshot=snapshot,
                dedupe_key=dedupe_key,
                metadata=metadata,
            )
            if inserted is not None:
                stored.append(inserted)
        return stored

    def _normalize_unstored_event(
        self, event: Mapping[str, Any], packet: FramePacket
    ) -> dict[str, Any]:
        result = dict(event)
        result.setdefault("session_id", self._session_id)
        result.setdefault("student_id", result.get("mssv", ""))
        result.setdefault("name", result.get("full_name", ""))
        result.setdefault("status", result.get("helmet_status", "unknown"))
        result.setdefault("confidence", result.get("score", 0.0))
        result.setdefault(
            "timestamp",
            datetime.fromtimestamp(packet.captured_at, tz=timezone.utc)
            .isoformat(timespec="milliseconds")
            .replace("+00:00", "Z"),
        )
        return result

    def _sync_registered_students(self) -> None:
        if self._processor is None:
            return
        for registration in self._registered_students.values():
            self._register_with_processor(self._processor, registration)

    @staticmethod
    def _register_with_processor(processor: Any, registration: Mapping[str, Any]) -> Any:
        candidates = [processor]
        for attribute in ("face_registry", "face_recognizer", "registry"):
            child = getattr(processor, attribute, None)
            if child is not None:
                candidates.append(child)
        for candidate in candidates:
            registrar = getattr(candidate, "register_student", None)
            if not callable(registrar):
                registrar = getattr(candidate, "register", None)
            if callable(registrar):
                context = {
                    "student_id": registration["student_id"],
                    "full_name": registration["full_name"],
                    "name": registration["full_name"],
                    "images": registration["images"],
                    "image_bytes": registration["images"],
                }
                return _call_registration(registrar, context)
        raise RuntimeError("The injected processor does not support student registration")

    def __enter__(self) -> "DemoPipeline":
        return self.start()

    def __exit__(self, _exc_type: Any, _exc: Any, _traceback: Any) -> None:
        self.close()


def _call_with_supported_keywords(
    target: Callable[..., Any], frame: Any, context: Mapping[str, Any]
) -> Any:
    try:
        signature = inspect.signature(target)
    except (TypeError, ValueError):
        return target(frame)
    accepts_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    keywords = {
        key: value
        for key, value in context.items()
        if accepts_kwargs or key in signature.parameters
    }
    return target(frame, **keywords)


def _call_registration(target: Callable[..., Any], context: Mapping[str, Any]) -> Any:
    try:
        signature = inspect.signature(target)
    except (TypeError, ValueError):
        return target(context["student_id"], context["full_name"], context["images"])
    accepts_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    keywords = {
        key: value
        for key, value in context.items()
        if accepts_kwargs or key in signature.parameters
    }
    if keywords:
        return target(**keywords)
    return target(context["student_id"], context["full_name"], context["images"])


def _utc_now_text() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z"
    )


def _create_default_processor(config: Any) -> Any:
    """Import the heavy end-to-end processor only when the UI starts a run."""

    try:
        from .processor import create_default_processor
    except ImportError as exc:
        raise RuntimeError(
            "The default classroom processor is unavailable. "
            "Install src.classroom_demo.processor or inject processor=..."
        ) from exc

    try:
        signature = inspect.signature(create_default_processor)
    except (TypeError, ValueError):
        return create_default_processor(config)
    if not signature.parameters:
        return create_default_processor()
    if "config" in signature.parameters:
        return create_default_processor(config=config)
    return create_default_processor(config)


class ClassroomDemoPipeline(DemoPipeline):
    """Zero-argument UI backend with a lazily constructed default processor."""

    def __init__(
        self,
        *args: Any,
        project_root: str | Path | None = None,
        processor: Any | None = None,
        processor_factory: Callable[[Any], Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self._project_root = (
            Path(project_root).resolve()
            if project_root is not None
            else Path(__file__).resolve().parents[2]
        )
        if processor is None and processor_factory is None:
            processor_factory = _create_default_processor
        created_store = "store" not in kwargs
        if created_store:
            output_dir = self._project_root / "outputs" / "classroom"
            kwargs["store"] = EventStore(
                output_dir / "db" / "events.db",
                snapshot_dir=output_dir / "snapshots",
                export_dir=output_dir / "exports",
                cooldown_seconds=5.0,
            )
        super().__init__(
            *args,
            processor=processor,
            processor_factory=processor_factory,
            **kwargs,
        )
        if created_store:
            # The store was constructed here even though it was passed to the
            # generic base class as an injected object.
            self._owns_store = True

    def _discard_owned_runtime(self) -> None:
        """Drop stopped source/model objects so changed UI config takes effect."""

        if self.running:
            return
        if self._owns_capture and self._capture is not None:
            self._capture = None
        if self._owns_processor and self._processor is not None:
            closer = getattr(self._processor, "close", None)
            if callable(closer):
                closer()
            self._processor = None

    def start(self, config: Any = None) -> "ClassroomDemoPipeline":
        if not self.running:
            self._discard_owned_runtime()
            if isinstance(config, Mapping):
                config = {"project_root": str(self._project_root), **dict(config)}
        super().start(config)
        return self

    def _processor_for_enrollment(self) -> Any:
        """Load models on first enrollment without requiring a running camera."""

        with self._lock:
            if self._processor is None:
                if self._processor_factory is None:
                    raise RuntimeError("The classroom processor is unavailable")
                self._processor = self._processor_factory(
                    {"project_root": str(self._project_root)}
                )
                self._owns_processor = True
            return self._processor

    def register_student(
        self,
        student_id: str,
        full_name: str,
        image_bytes: bytes | bytearray | memoryview | Iterable[Any],
    ) -> Any:
        """Persist enrollment immediately, including before camera start."""

        if isinstance(image_bytes, (bytes, bytearray, memoryview)):
            images = [bytes(image_bytes)]
        else:
            images = list(image_bytes)
        if not images:
            raise ValueError("At least one student image is required")
        processor = self._processor_for_enrollment()
        return self._register_with_processor(
            processor,
            {
                "student_id": str(student_id).strip(),
                "full_name": str(full_name).strip(),
                "images": images,
            },
        )

    def list_students(self, *, limit: int = 1000) -> list[dict[str, Any]]:
        """List the on-disk roster without forcing model initialization."""

        processor = self._processor
        provider = getattr(processor, "list_students", None) if processor else None
        if callable(provider):
            return list(provider())[: max(0, int(limit))]

        from .face_recognition import load_roster_csv

        roster = self._project_root / "data" / "students.csv"
        references = self._project_root / "data" / "students"
        records = load_roster_csv(roster, reference_root=references)
        return [
            {
                "student_id": record.student_id,
                "full_name": record.full_name,
                "name": record.full_name,
                "image_count": len(record.reference_images),
                "active": str(record.metadata.get("active", "1")).strip().casefold()
                not in {"0", "false", "no"},
            }
            for record in records[: max(0, int(limit))]
        ]


ClassroomPipeline = ClassroomDemoPipeline


def create_pipeline(**kwargs: Any) -> ClassroomDemoPipeline:
    return ClassroomDemoPipeline(**kwargs)
