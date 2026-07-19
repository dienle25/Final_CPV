"""Runnable frame processor that wires all classroom inference components."""

from __future__ import annotations

import csv
import io
import os
import tempfile
import threading
import time
import unicodedata
from collections import Counter, deque
from collections.abc import Mapping
from dataclasses import dataclass, fields
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from .detectors import (
    NanoDetPersonDetector,
    YoloHelmetOnnxDetector,
    YuNetFaceDetector,
)
from .face_recognition import (
    DEFAULT_COSINE_MARGIN,
    DEFAULT_COSINE_THRESHOLD,
    FaceGallery,
    FaceMatch,
    SFaceEncoder,
    normalize_embedding,
    load_roster_csv,
)
from .helmet_status import (
    HelmetStatus,
    HelmetStatusEstimator,
    PersonAssociation,
)
from .tracking import IoUTracker


@dataclass(slots=True)
class ClassroomProcessorConfig:
    """Paths and conservative defaults for a Windows classroom demo."""

    project_root: str | Path | None = None
    person_model: str | Path = "models/person/object_detection_nanodet_2022nov.onnx"
    helmet_model: str | Path = "models/best.onnx"
    yunet_model: str | Path = "models/face/face_detection_yunet_2023mar.onnx"
    sface_model: str | Path = "models/face/face_recognition_sface_2021dec.onnx"
    roster_csv: str | Path = "data/students.csv"
    reference_root: str | Path = "data/students"
    providers: tuple[str, ...] | None = None

    person_confidence: float = 0.35
    helmet_confidence: float = 0.25
    face_confidence: float = 0.85
    # YuNet scores on the user's real CE190579 enrollment clip are 0.85-0.88
    # despite strong SFace matches (median cosine ~0.84).  Keep the detector
    # gate aligned with ``face_confidence`` and let cosine + top-2 margin +
    # temporal voting provide the identity safety checks.
    recognition_minimum_face_score: float = 0.85
    recognition_minimum_face_width: float = 64.0
    recognition_minimum_face_height: float = 80.0
    registration_minimum_face_width: float = 32.0
    registration_minimum_face_height: float = 48.0
    registration_inlier_threshold: float = 0.25
    registration_consistency_threshold: float = 0.35
    registration_duplicate_threshold: float = 0.55
    detection_iou: float = 0.45
    tracker_iou: float = 0.25
    tracker_max_missed: int = 15

    cosine_threshold: float = DEFAULT_COSINE_THRESHOLD
    cosine_margin: float = DEFAULT_COSINE_MARGIN
    recognition_interval_frames: int = 2
    identity_history: int = 8
    identity_min_votes: int = 3
    identity_min_ratio: float = 0.75
    identity_reject_reset_count: int = 2
    identity_stale_frames: int = 8
    duplicate_identity_min_margin: float = 0.05
    status_history: int = 8
    status_min_votes: int = 2
    status_min_ratio: float = 0.60
    status_stale_frames: int = 9
    status_unknown_reset_count: int = 3
    temporal_state_ttl_frames: int = 90

    def root(self) -> Path:
        if self.project_root is not None:
            return Path(self.project_root).resolve()
        return Path(__file__).resolve().parents[2]

    def resolve(self, value: str | Path) -> Path:
        path = Path(value)
        return path.resolve() if path.is_absolute() else (self.root() / path).resolve()

    def __post_init__(self) -> None:
        if self.recognition_interval_frames < 1:
            raise ValueError("recognition_interval_frames must be at least one")
        if self.identity_history < 1 or self.status_history < 1:
            raise ValueError("Temporal history sizes must be positive")
        if self.identity_min_votes < 1 or self.status_min_votes < 1:
            raise ValueError("Temporal minimum votes must be positive")
        if self.identity_reject_reset_count < 1 or self.status_unknown_reset_count < 1:
            raise ValueError("Temporal reset counts must be positive")
        if self.identity_stale_frames < 1 or self.status_stale_frames < 1:
            raise ValueError("Temporal stale-frame limits must be positive")
        if not 0.0 <= self.identity_min_ratio <= 1.0:
            raise ValueError("identity_min_ratio must be in [0, 1]")
        if not 0.0 <= self.status_min_ratio <= 1.0:
            raise ValueError("status_min_ratio must be in [0, 1]")
        if not 0.0 <= self.duplicate_identity_min_margin <= 1.0:
            raise ValueError("duplicate_identity_min_margin must be in [0, 1]")
        for name in (
            "face_confidence",
            "recognition_minimum_face_score",
            "registration_inlier_threshold",
            "registration_consistency_threshold",
            "registration_duplicate_threshold",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        for name in (
            "recognition_minimum_face_width",
            "recognition_minimum_face_height",
            "registration_minimum_face_width",
            "registration_minimum_face_height",
        ):
            if float(getattr(self, name)) <= 0.0:
                raise ValueError(f"{name} must be positive")


@dataclass(slots=True)
class _TemporalPersonState:
    identity_votes: deque[tuple[str, float]]
    status_votes: deque[HelmetStatus]
    last_seen_frame: int = -1
    last_recognition_frame: int = -1
    last_status_frame: int = -1
    last_similarity: float = 0.0
    last_helmet_confidence: float = 0.0
    last_identity_accept_frame: int = -1
    identity_reject_streak: int = 0
    status_unknown_streak: int = 0
    emitted_observation: tuple[str | None, HelmetStatus] | None = None


class ClassroomFrameProcessor:
    """End-to-end processor for one IP Webcam/OpenCV frame at a time.

    Construction uses real models and the roster by default; no dependency
    injection is required by the UI.  ``process_frame`` is serialized because
    it mutates tracker and temporal-vote state, while each DirectML detector
    also protects its own ONNX Runtime session.
    """

    def __init__(
        self,
        config: ClassroomProcessorConfig | None = None,
        *,
        person_detector: Any | None = None,
        helmet_detector: Any | None = None,
        face_detector: Any | None = None,
        face_encoder: Any | None = None,
    ) -> None:
        self.config = config or ClassroomProcessorConfig()
        providers = self.config.providers
        self.person_detector = person_detector or NanoDetPersonDetector(
            self.config.resolve(self.config.person_model),
            confidence_threshold=self.config.person_confidence,
            iou_threshold=self.config.detection_iou,
            providers=providers,
        )
        self.helmet_detector = helmet_detector or YoloHelmetOnnxDetector(
            self.config.resolve(self.config.helmet_model),
            confidence_threshold=self.config.helmet_confidence,
            iou_threshold=self.config.detection_iou,
            providers=providers,
        )
        self.face_detector = face_detector or YuNetFaceDetector(
            self.config.resolve(self.config.yunet_model),
            score_threshold=self.config.face_confidence,
        )
        self.face_encoder = face_encoder or SFaceEncoder(
            self.config.resolve(self.config.sface_model)
        )
        self.person_tracker = IoUTracker(
            iou_threshold=self.config.tracker_iou,
            max_missed=self.config.tracker_max_missed,
            class_aware=True,
        )
        self.status_estimator = HelmetStatusEstimator()
        self._gallery: FaceGallery | None = None
        self.gallery_warnings: tuple[str, ...] = ()
        self._states: dict[int, _TemporalPersonState] = {}
        self._frame_index = -1
        self._lock = threading.RLock()
        self.rebuild_gallery()

    @property
    def gallery(self) -> FaceGallery | None:
        with self._lock:
            return self._gallery

    @staticmethod
    def _decode_image_bytes(payload: bytes) -> np.ndarray:
        if not payload:
            raise ValueError("Reference image bytes are empty")
        try:
            import cv2  # type: ignore

            decoded = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
            if decoded is None:
                raise ValueError("Reference image bytes cannot be decoded")
            return decoded
        except ImportError:
            from PIL import Image, UnidentifiedImageError

            try:
                rgb = np.asarray(Image.open(io.BytesIO(payload)).convert("RGB"))
            except (OSError, UnidentifiedImageError) as error:
                raise ValueError("Reference image bytes cannot be decoded") from error
            return np.ascontiguousarray(rgb[:, :, ::-1])

    @classmethod
    def _load_image_path(cls, path: Path) -> np.ndarray | None:
        try:
            return cls._decode_image_bytes(path.read_bytes())
        except (OSError, ValueError):
            return None

    @staticmethod
    def _encode_jpeg(image: np.ndarray) -> bytes:
        if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] < 3:
            raise ValueError("Reference image must be a BGR color numpy array")
        try:
            import cv2  # type: ignore

            ok, encoded = cv2.imencode(".jpg", image[:, :, :3])
            if not ok:
                raise ValueError("Reference image could not be JPEG-encoded")
            return encoded.tobytes()
        except ImportError:
            from PIL import Image

            buffer = io.BytesIO()
            Image.fromarray(image[:, :, :3][:, :, ::-1]).save(
                buffer,
                format="JPEG",
                quality=95,
            )
            return buffer.getvalue()

    def _new_state(self) -> _TemporalPersonState:
        return _TemporalPersonState(
            identity_votes=deque(maxlen=self.config.identity_history),
            status_votes=deque(maxlen=self.config.status_history),
        )

    def rebuild_gallery(self, *, strict: bool = False) -> FaceGallery | None:
        """Reload roster/references and atomically replace the live gallery."""

        with self._lock:
            records = load_roster_csv(
                self.config.resolve(self.config.roster_csv),
                reference_root=self.config.resolve(self.config.reference_root),
            )
            try:
                gallery = FaceGallery.build(
                    records,
                    self.face_detector,
                    self.face_encoder,
                    image_loader=self._load_image_path,
                    cosine_threshold=self.config.cosine_threshold,
                    cosine_margin=self.config.cosine_margin,
                    strict=strict,
                )
            except ValueError as error:
                if strict:
                    raise
                self._gallery = None
                self.gallery_warnings = (str(error),)
                return None
            self._gallery = gallery
            self.gallery_warnings = gallery.warnings
            return gallery

    @staticmethod
    def _normalized_csv_header(value: str) -> str:
        decomposed = unicodedata.normalize("NFKD", value)
        ascii_text = "".join(
            character
            for character in decomposed
            if not unicodedata.combining(character)
        )
        return "".join(
            character for character in ascii_text.casefold() if character.isalnum()
        )

    def _read_roster_table(
        self,
    ) -> tuple[list[str], list[dict[str, str]], str, str, str | None]:
        roster_path = self.config.resolve(self.config.roster_csv)
        if not roster_path.is_file():
            return (
                ["student_id", "full_name", "active"],
                [],
                "student_id",
                "full_name",
                "active",
            )
        with roster_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = list(reader.fieldnames or [])
            rows = [
                {str(key): str(value or "") for key, value in row.items() if key is not None}
                for row in reader
            ]
        normalized = {
            self._normalized_csv_header(column): column for column in fieldnames
        }
        id_column = next(
            (
                normalized[key]
                for key in ("studentid", "mssv", "masosinhvien", "id")
                if key in normalized
            ),
            None,
        )
        name_column = next(
            (
                normalized[key]
                for key in ("fullname", "hoten", "name")
                if key in normalized
            ),
            None,
        )
        active_column = normalized.get("active")
        if id_column is None or name_column is None:
            raise ValueError(
                "students.csv must contain student_id/MSSV and full_name/Họ tên"
            )
        return fieldnames, rows, id_column, name_column, active_column

    def _write_roster_atomic(
        self,
        fieldnames: list[str],
        rows: list[dict[str, str]],
    ) -> None:
        roster_path = self.config.resolve(self.config.roster_csv)
        roster_path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{roster_path.stem}_",
            suffix=".tmp",
            dir=roster_path.parent,
        )
        os.close(descriptor)
        temporary_path = Path(temporary_name)
        try:
            with temporary_path.open("w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(rows)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, roster_path)
        finally:
            temporary_path.unlink(missing_ok=True)

    @staticmethod
    def _registration_items(
        images: Any,
        image_bytes: bytes | list[bytes] | tuple[bytes, ...] | None,
    ) -> list[Any]:
        items: list[Any] = []
        if images is not None:
            if isinstance(images, (np.ndarray, bytes, bytearray, str, Path)) or hasattr(
                images, "getvalue"
            ) or hasattr(images, "read"):
                items.append(images)
            else:
                items.extend(list(images))
        if image_bytes is not None:
            if isinstance(image_bytes, (bytes, bytearray)):
                items.append(bytes(image_bytes))
            else:
                items.extend(bytes(value) for value in image_bytes)
        return items

    def _decode_registration_item(self, item: Any) -> np.ndarray:
        if isinstance(item, np.ndarray):
            image = item.copy()
        elif isinstance(item, (bytes, bytearray)):
            image = self._decode_image_bytes(bytes(item))
        elif isinstance(item, (str, Path)):
            source = Path(item)
            if not source.is_file():
                raise FileNotFoundError(f"Reference image not found: {source}")
            image = self._decode_image_bytes(source.read_bytes())
        elif hasattr(item, "getvalue"):
            image = self._decode_image_bytes(bytes(item.getvalue()))
        elif hasattr(item, "read"):
            image = self._decode_image_bytes(bytes(item.read()))
        else:
            raise TypeError(f"Unsupported reference image type: {type(item).__name__}")
        if image.ndim != 3 or image.shape[2] < 3 or image.size == 0:
            raise ValueError("Reference image must decode as a non-empty color image")
        return np.ascontiguousarray(image[:, :, :3])

    def list_students(self) -> list[dict[str, Any]]:
        """Return roster entries and current reference-image counts."""

        with self._lock:
            records = load_roster_csv(
                self.config.resolve(self.config.roster_csv),
                reference_root=self.config.resolve(self.config.reference_root),
            )
            return [
                {
                    "student_id": record.student_id,
                    "full_name": record.full_name,
                    "active": str(record.metadata.get("active", "1")).strip().casefold()
                    not in {"0", "false", "no"},
                    "image_count": len(record.reference_images),
                    "reference_images": [str(path) for path in record.reference_images],
                }
                for record in records
            ]

    def register_student(
        self,
        student_id: str,
        full_name: str,
        images: Any = None,
        *,
        image_bytes: bytes | list[bytes] | tuple[bytes, ...] | None = None,
        rebuild: bool = True,
    ) -> dict[str, Any]:
        """Atomically add/update a roster student and validated face images.

        Every payload is decoded before any file changes and must contain
        exactly one YuNet face.  JPEG encoding happens in memory, then image
        renames and the UTF-8 roster replacement occur while frame processing
        is locked.  This is safe for registration before or during a live run.
        """

        student_id = student_id.strip()
        full_name = " ".join(full_name.split())
        if not student_id or student_id != Path(student_id).name:
            raise ValueError("student_id is empty or unsafe for a directory name")
        if not all(character.isalnum() or character in {"-", "_"} for character in student_id):
            raise ValueError("student_id may contain only letters, digits, '-' and '_'")
        if not full_name:
            raise ValueError("full_name cannot be empty")
        items = self._registration_items(images, image_bytes)
        if not items:
            raise ValueError("At least one reference image is required")

        with self._lock:
            encoded_images: list[bytes] = []
            features: list[np.ndarray] = []
            for index, item in enumerate(items, start=1):
                image = self._decode_registration_item(item)
                faces = self.face_detector.detect(image)
                if len(faces) != 1:
                    raise ValueError(
                        f"Reference image {index} must contain exactly one face; "
                        f"YuNet found {len(faces)}"
                    )
                x1, y1, x2, y2 = faces[0].bbox
                face_width = x2 - x1
                face_height = y2 - y1
                if (
                    face_width < self.config.registration_minimum_face_width
                    or face_height < self.config.registration_minimum_face_height
                ):
                    raise ValueError(
                        f"Reference image {index} face is too small "
                        f"({face_width:.0f}x{face_height:.0f}px); move closer to the camera"
                    )
                # Validate that SFace can align/extract before committing files.
                feature = normalize_embedding(self.face_encoder.extract(image, faces[0]))
                features.append(feature)
                encoded_images.append(self._encode_jpeg(image))

            feature_dimensions = {feature.size for feature in features}
            if len(feature_dimensions) != 1:
                raise ValueError("Reference images produced inconsistent embeddings")
            new_centroid = normalize_embedding(np.mean(np.stack(features), axis=0))
            weakest_inlier = min(float(np.dot(feature, new_centroid)) for feature in features)
            if weakest_inlier < self.config.registration_inlier_threshold:
                raise ValueError(
                    "Reference images are not mutually consistent; make sure every image "
                    "shows the same student"
                )

            gallery = self._gallery
            if gallery is not None:
                similarities = {
                    candidate_id: float(np.dot(new_centroid, centroid))
                    for candidate_id, centroid in gallery.centroids.items()
                }
                target_similarity = similarities.get(student_id)
                if (
                    target_similarity is not None
                    and target_similarity < self.config.registration_consistency_threshold
                ):
                    raise ValueError(
                        f"New images do not match the existing face for {student_id} "
                        f"(cosine={target_similarity:.3f})"
                    )
                other_scores = {
                    candidate_id: similarity
                    for candidate_id, similarity in similarities.items()
                    if candidate_id != student_id
                }
                if other_scores:
                    other_id, other_similarity = max(
                        other_scores.items(),
                        key=lambda item: (item[1], item[0]),
                    )
                    conflicts_with_other = (
                        other_similarity >= self.config.registration_duplicate_threshold
                        and (
                            target_similarity is None
                            or other_similarity >= target_similarity - 0.02
                        )
                    )
                    if conflicts_with_other:
                        raise ValueError(
                            f"New images are too similar to registered student {other_id}; "
                            "registration was blocked to prevent duplicate identities"
                        )

            fieldnames, rows, id_column, name_column, active_column = (
                self._read_roster_table()
            )
            existing_row = next(
                (row for row in rows if row.get(id_column, "").strip() == student_id),
                None,
            )
            if existing_row is None:
                existing_row = {column: "" for column in fieldnames}
                existing_row[id_column] = student_id
                rows.append(existing_row)
            existing_row[name_column] = full_name
            if active_column is not None:
                existing_row[active_column] = "1"

            destination_dir = self.config.resolve(self.config.reference_root) / student_id
            destination_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            final_paths = [
                destination_dir / f"reference_{timestamp}_{index:02d}.jpg"
                for index in range(1, len(encoded_images) + 1)
            ]
            temporary_paths = [
                destination_dir / f".{path.name}.tmp" for path in final_paths
            ]
            committed_paths: list[Path] = []
            try:
                for temporary_path, payload in zip(temporary_paths, encoded_images):
                    temporary_path.write_bytes(payload)
                for temporary_path, final_path in zip(temporary_paths, final_paths):
                    os.replace(temporary_path, final_path)
                    committed_paths.append(final_path)
                self._write_roster_atomic(fieldnames, rows)
            except Exception:
                for temporary_path in temporary_paths:
                    temporary_path.unlink(missing_ok=True)
                for committed_path in committed_paths:
                    committed_path.unlink(missing_ok=True)
                raise

            if rebuild:
                self.rebuild_gallery(strict=False)
            student = next(
                item for item in self.list_students() if item["student_id"] == student_id
            )
            student["added_images"] = [str(path) for path in final_paths]
            student["gallery_size"] = len(self._gallery.centroids) if self._gallery else 0
            student["gallery_warnings"] = self.gallery_warnings
            return student

    def register_reference_image(
        self,
        student_id: str,
        image: np.ndarray | str | Path,
        *,
        filename: str | None = None,
        rebuild: bool = True,
    ) -> Path:
        """Store one consented reference image and optionally rebuild centroids."""

        del filename  # Filenames are generated to guarantee collision-free atomic writes.
        records = {
            item["student_id"]: item for item in self.list_students()
        }
        if student_id not in records:
            raise KeyError(f"Student ID is not present in roster: {student_id}")
        result = self.register_student(
            student_id,
            records[student_id]["full_name"],
            images=[image],
            rebuild=rebuild,
        )
        return Path(result["added_images"][-1])

    # UI-friendly alias.
    add_reference_image = register_reference_image

    def _resolve_identity(
        self,
        state: _TemporalPersonState,
        frame_index: int,
    ) -> tuple[str | None, str | None, float]:
        if (
            not state.identity_votes
            or state.last_identity_accept_frame < 0
            or frame_index - state.last_identity_accept_frame
            > self.config.identity_stale_frames
        ):
            return None, None, state.last_similarity
        counts = Counter(student_id for student_id, _score in state.identity_votes)
        student_id, votes = sorted(
            counts.items(),
            key=lambda item: (-item[1], item[0]),
        )[0]
        if votes < self.config.identity_min_votes:
            return None, None, state.last_similarity
        if votes / len(state.identity_votes) < self.config.identity_min_ratio:
            return None, None, state.last_similarity
        scores = [
            score
            for candidate_id, score in state.identity_votes
            if candidate_id == student_id
        ]
        gallery = self._gallery
        record = gallery.records.get(student_id) if gallery is not None else None
        return (
            student_id,
            record.full_name if record is not None else None,
            float(sum(scores) / len(scores)),
        )

    def _resolve_status(
        self,
        state: _TemporalPersonState,
        frame_index: int,
    ) -> HelmetStatus:
        if (
            not state.status_votes
            or state.last_status_frame < 0
            or frame_index - state.last_status_frame > self.config.status_stale_frames
        ):
            return HelmetStatus.UNKNOWN
        counts = Counter(state.status_votes)
        status, votes = sorted(
            counts.items(),
            key=lambda item: (-item[1], item[0].value),
        )[0]
        if votes < self.config.status_min_votes:
            return HelmetStatus.UNKNOWN
        if votes / len(state.status_votes) < self.config.status_min_ratio:
            return HelmetStatus.UNKNOWN
        return status

    def _recognize_if_due(
        self,
        frame: np.ndarray,
        association: PersonAssociation,
        state: _TemporalPersonState,
        frame_index: int,
    ) -> FaceMatch | None:
        gallery = self._gallery
        if association.face is None or gallery is None:
            return None
        if (
            state.last_recognition_frame >= 0
            and frame_index - state.last_recognition_frame
            < self.config.recognition_interval_frames
        ):
            return None
        state.last_recognition_frame = frame_index
        x1, y1, x2, y2 = association.face.bbox
        if association.face.score < self.config.recognition_minimum_face_score:
            state.identity_reject_streak += 1
            if state.identity_reject_streak >= self.config.identity_reject_reset_count:
                state.identity_votes.clear()
            return FaceMatch(
                student_id=None,
                full_name=None,
                similarity=0.0,
                second_similarity=None,
                margin=0.0,
                accepted=False,
                reason="low_face_confidence",
            )
        if (
            x2 - x1 < self.config.recognition_minimum_face_width
            or y2 - y1 < self.config.recognition_minimum_face_height
        ):
            state.identity_reject_streak += 1
            if state.identity_reject_streak >= self.config.identity_reject_reset_count:
                state.identity_votes.clear()
            return FaceMatch(
                student_id=None,
                full_name=None,
                similarity=0.0,
                second_similarity=None,
                margin=0.0,
                accepted=False,
                reason="face_too_small",
            )
        match = gallery.recognize(frame, association.face, self.face_encoder)
        state.last_similarity = match.similarity
        if match.accepted and match.student_id is not None:
            if (
                state.identity_votes
                and state.identity_votes[-1][0] != match.student_id
            ):
                # A track that suddenly looks like another registered person
                # must earn a fresh confirmation sequence instead of inheriting
                # the previous person's votes.
                state.identity_votes.clear()
            state.identity_votes.append((match.student_id, match.similarity))
            state.last_identity_accept_frame = frame_index
            state.identity_reject_streak = 0
        else:
            state.identity_reject_streak += 1
            if state.identity_reject_streak >= self.config.identity_reject_reset_count:
                state.identity_votes.clear()
        return match

    def _events_for_state(
        self,
        *,
        state: _TemporalPersonState,
        frame_index: int,
        track_id: int,
        student_id: str | None,
        full_name: str | None,
        status: HelmetStatus,
        face_similarity: float,
        helmet_confidence: float,
    ) -> list[dict[str, Any]]:
        observation = (student_id, status)
        if observation == state.emitted_observation:
            return []
        if student_id is None and status is HelmetStatus.UNKNOWN:
            return []
        timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
        previous = state.emitted_observation
        event = {
            "type": "person_observation_changed",
            "timestamp": timestamp,
            "frame_index": frame_index,
            "track_id": track_id,
            # Anonymous and identified observations need distinct cooldown
            # identities.  Otherwise the MSSV-bearing update a few frames
            # later would be suppressed by EventStore as a duplicate.
            "dedupe_key": f"track:{track_id}:student:{student_id or 'unknown'}",
            "student_id": student_id,
            "full_name": full_name,
            "helmet_status": status.value,
            "helmet_status_code": status.name,
            "face_similarity": float(face_similarity),
            "helmet_confidence": float(helmet_confidence),
            "previous_student_id": previous[0] if previous is not None else None,
            "previous_helmet_status": (
                previous[1].value if previous is not None else HelmetStatus.UNKNOWN.value
            ),
        }
        # Updating the pair (rather than identity/status independently) ensures
        # the same helmet result is emitted again when an MSSV becomes stable.
        state.emitted_observation = observation
        return [event]

    def _purge_states(self, frame_index: int) -> None:
        stale = [
            track_id
            for track_id, state in self._states.items()
            if frame_index - state.last_seen_frame > self.config.temporal_state_ttl_frames
        ]
        for track_id in stale:
            self._states.pop(track_id, None)

    def _arbitrate_duplicate_identities(
        self,
        people: list[dict[str, Any]],
        events: list[dict[str, Any]],
    ) -> None:
        """Enforce that one MSSV cannot belong to two live tracks at once."""

        by_student: dict[str, list[dict[str, Any]]] = {}
        for person in people:
            student_id = person.get("student_id")
            if student_id:
                by_student.setdefault(str(student_id), []).append(person)

        for student_id, candidates in by_student.items():
            if len(candidates) < 2:
                continue
            ranked = sorted(
                candidates,
                key=lambda person: (
                    -float(person.get("identity_similarity") or 0.0),
                    int(person.get("track_id") or 0),
                ),
            )
            top_score = float(ranked[0].get("identity_similarity") or 0.0)
            second_score = float(ranked[1].get("identity_similarity") or 0.0)
            losers = (
                ranked
                if top_score - second_score < self.config.duplicate_identity_min_margin
                else ranked[1:]
            )
            losing_tracks = {int(person["track_id"]) for person in losers}
            for person in losers:
                track_id = int(person["track_id"])
                state = self._states.get(track_id)
                if state is not None:
                    state.identity_votes.clear()
                    state.last_identity_accept_frame = -1
                    state.identity_reject_streak = self.config.identity_reject_reset_count
                    try:
                        status = HelmetStatus(str(person.get("helmet_status")))
                    except ValueError:
                        status = HelmetStatus.UNKNOWN
                    state.emitted_observation = (None, status)
                person["student_id"] = None
                person["full_name"] = None
                person["display_identity"] = "Chưa xác định"
                person["identity_current_reason"] = "duplicate_identity_conflict"

            for event in events:
                if (
                    int(event.get("track_id") or -1) in losing_tracks
                    and event.get("student_id") == student_id
                ):
                    event["student_id"] = None
                    event["full_name"] = None
                    event["dedupe_key"] = (
                        f"track:{int(event.get('track_id') or -1)}:student:unknown"
                    )

    def process_frame(
        self,
        frame: np.ndarray,
        *,
        frame_index: int | None = None,
    ) -> dict[str, Any]:
        """Process one BGR frame and return ``people``, ``events`` and ``stats``."""

        if not isinstance(frame, np.ndarray) or frame.ndim != 3:
            raise ValueError("process_frame expects a BGR numpy frame")
        with self._lock:
            current_frame = self._frame_index + 1 if frame_index is None else int(frame_index)
            if current_frame <= self._frame_index:
                raise ValueError("frame_index must increase between process_frame calls")
            self._frame_index = current_frame
            started = time.perf_counter()

            stage_started = time.perf_counter()
            person_detections = self.person_detector.detect(frame)
            tracked_people = self.person_tracker.update(
                person_detections,
                frame_index=current_frame,
            )
            person_ms = (time.perf_counter() - stage_started) * 1000.0

            stage_started = time.perf_counter()
            helmet_detections = self.helmet_detector.detect(frame)
            helmet_ms = (time.perf_counter() - stage_started) * 1000.0

            stage_started = time.perf_counter()
            faces = self.face_detector.detect(frame)
            face_detection_ms = (time.perf_counter() - stage_started) * 1000.0

            associations = self.status_estimator.associate(
                tracked_people,
                faces,
                helmet_detections,
            )
            people_payload: list[dict[str, Any]] = []
            events: list[dict[str, Any]] = []
            status_counts: Counter[str] = Counter()
            recognition_started = time.perf_counter()

            for association in associations:
                tracked = tracked_people[association.person_index]
                track_id = tracked.track_id
                state = self._states.setdefault(track_id, self._new_state())
                if (
                    state.last_seen_frame >= 0
                    and current_frame - state.last_seen_frame > 1
                ):
                    # IoU tracking has no appearance ReID.  Reusing a track after
                    # a gap must never transfer the previous person's MSSV/status.
                    state = self._new_state()
                    self._states[track_id] = state
                state.last_seen_frame = current_frame
                raw_status = association.status
                if raw_status is not HelmetStatus.UNKNOWN:
                    state.status_votes.append(raw_status)
                    state.last_status_frame = current_frame
                    state.last_helmet_confidence = association.confidence
                    state.status_unknown_streak = 0
                else:
                    state.status_unknown_streak += 1
                    if (
                        state.status_unknown_streak
                        >= self.config.status_unknown_reset_count
                    ):
                        state.status_votes.clear()
                        state.last_status_frame = -1
                current_match = self._recognize_if_due(
                    frame,
                    association,
                    state,
                    current_frame,
                )
                student_id, full_name, identity_similarity = self._resolve_identity(
                    state,
                    current_frame,
                )
                stable_status = self._resolve_status(state, current_frame)
                status_counts[stable_status.value] += 1
                events.extend(
                    self._events_for_state(
                        state=state,
                        frame_index=current_frame,
                        track_id=track_id,
                        student_id=student_id,
                        full_name=full_name,
                        status=stable_status,
                        face_similarity=identity_similarity,
                        helmet_confidence=state.last_helmet_confidence,
                    )
                )
                people_payload.append(
                    {
                        "track_id": track_id,
                        "bbox": tracked.bbox,
                        "person_confidence": tracked.score,
                        "student_id": student_id,
                        "full_name": full_name,
                        "display_identity": (
                            f"{student_id} - {full_name}"
                            if student_id and full_name
                            else "Chưa xác định"
                        ),
                        "identity_similarity": identity_similarity,
                        "identity_current_reason": (
                            current_match.reason if current_match is not None else None
                        ),
                        "helmet_status": stable_status.value,
                        "helmet_status_code": stable_status.name,
                        "raw_helmet_status": raw_status.value,
                        "helmet_confidence": association.confidence,
                        "status_reason": association.reason,
                        "face_bbox": (
                            association.face.bbox if association.face is not None else None
                        ),
                        "head_bbox": (
                            association.head.bbox if association.head is not None else None
                        ),
                    }
                )
            recognition_ms = (time.perf_counter() - recognition_started) * 1000.0
            self._arbitrate_duplicate_identities(people_payload, events)
            self._purge_states(current_frame)
            total_ms = (time.perf_counter() - started) * 1000.0
            stats = {
                "frame_index": current_frame,
                "people_count": len(people_payload),
                "face_count": len(faces),
                "helmet_detection_count": len(helmet_detections),
                "recognized_count": sum(
                    person["student_id"] is not None for person in people_payload
                ),
                "unknown_identity_count": sum(
                    person["student_id"] is None for person in people_payload
                ),
                "status_counts": dict(status_counts),
                "gallery_size": len(self._gallery.centroids) if self._gallery else 0,
                "gallery_warnings": self.gallery_warnings,
                "timing_ms": {
                    "person": round(person_ms, 2),
                    "helmet": round(helmet_ms, 2),
                    "face_detection": round(face_detection_ms, 2),
                    "face_recognition_and_association": round(recognition_ms, 2),
                    "total": round(total_ms, 2),
                },
                "processing_fps": round(1000.0 / total_ms, 2) if total_ms > 0 else 0.0,
            }
            return {
                "people": people_payload,
                "events": events,
                "stats": stats,
            }

    def reset(self) -> None:
        with self._lock:
            self.person_tracker.reset()
            self._states.clear()
            self._frame_index = -1

    def close(self) -> None:
        self.reset()


def create_default_processor(
    config: ClassroomProcessorConfig | Mapping[str, Any] | object | None = None,
) -> ClassroomFrameProcessor:
    """Factory used by UI/engine code without manual component wiring."""

    resolved = coerce_processor_config(config)
    return ClassroomFrameProcessor(resolved)


def coerce_processor_config(
    config: ClassroomProcessorConfig | Mapping[str, Any] | object | None,
) -> ClassroomProcessorConfig:
    """Extract processor fields from a wider UI/application configuration.

    The UI also owns source, output and evidence settings.  Ignoring those keys
    here lets one shared mapping cross the integration boundary without making
    the inference layer depend on UI/storage concepts.
    """

    if config is None:
        return ClassroomProcessorConfig()
    if isinstance(config, ClassroomProcessorConfig):
        return config
    if isinstance(config, Mapping):
        incoming = dict(config)
    elif hasattr(config, "__dict__"):
        incoming = dict(vars(config))
    else:
        raise TypeError("config must be a mapping, object with attributes, or None")

    aliases = {
        "model_path": "helmet_model",
        "helmet_model_path": "helmet_model",
        "person_model_path": "person_model",
        "yunet_model_path": "yunet_model",
        "face_detection_model_path": "yunet_model",
        "sface_model_path": "sface_model",
        "face_recognition_model_path": "sface_model",
        "roster_path": "roster_csv",
        "reference_dir": "reference_root",
        "face_match_threshold": "cosine_threshold",
        "face_cosine_threshold": "cosine_threshold",
        "face_match_margin": "cosine_margin",
        "person_confidence_threshold": "person_confidence",
        "helmet_confidence_threshold": "helmet_confidence",
        "face_detection_confidence": "face_confidence",
    }
    normalized = dict(incoming)
    for source, destination in aliases.items():
        if source in incoming and destination not in normalized:
            normalized[destination] = incoming[source]

    if "detection_confidence" in incoming:
        normalized.setdefault("person_confidence", incoming["detection_confidence"])
        normalized.setdefault("helmet_confidence", incoming["detection_confidence"])

    if "providers" in normalized and normalized["providers"] is not None:
        normalized["providers"] = tuple(normalized["providers"])
    elif "device" in incoming:
        device = str(incoming["device"]).strip().casefold()
        provider_map: dict[str, tuple[str, ...] | None] = {
            "auto": None,
            "dml": ("DmlExecutionProvider", "CPUExecutionProvider"),
            "directml": ("DmlExecutionProvider", "CPUExecutionProvider"),
            "cpu": ("CPUExecutionProvider",),
            "cuda": ("CUDAExecutionProvider", "CPUExecutionProvider"),
            "gpu": ("CUDAExecutionProvider", "CPUExecutionProvider"),
        }
        if device not in provider_map:
            raise ValueError(f"Unsupported inference device: {incoming['device']}")
        normalized["providers"] = provider_map[device]

    allowed = {field.name for field in fields(ClassroomProcessorConfig)}
    processor_values = {
        key: value for key, value in normalized.items() if key in allowed
    }
    return ClassroomProcessorConfig(**processor_values)
