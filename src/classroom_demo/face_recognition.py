"""Roster loading and OpenCV SFace recognition for the classroom demo."""

from __future__ import annotations

import csv
import math
import re
import threading
import unicodedata
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from .detectors import FaceDetection
from .geometry import area


DEFAULT_COSINE_THRESHOLD = 0.50
DEFAULT_COSINE_MARGIN = 0.10
RECOMMENDED_REFERENCE_IMAGES = 5
SPARSE_REFERENCE_COSINE_THRESHOLD = 0.55
_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def normalize_embedding(values: np.ndarray | Sequence[float]) -> np.ndarray:
    """Return a finite, unit-length ``float32`` embedding."""

    embedding = np.asarray(values, dtype=np.float32).reshape(-1)
    if embedding.size == 0 or not np.all(np.isfinite(embedding)):
        raise ValueError("Face embedding must contain finite values")
    norm = float(np.linalg.norm(embedding))
    if norm <= 1e-12:
        raise ValueError("Face embedding has zero norm")
    return embedding / norm


def cosine_similarity(
    first: np.ndarray | Sequence[float],
    second: np.ndarray | Sequence[float],
) -> float:
    a = normalize_embedding(first)
    b = normalize_embedding(second)
    if a.shape != b.shape:
        raise ValueError(f"Embedding dimensions differ: {a.shape} vs {b.shape}")
    return float(np.clip(np.dot(a, b), -1.0, 1.0))


@dataclass(frozen=True, slots=True)
class StudentRecord:
    student_id: str
    full_name: str
    reference_images: tuple[Path, ...] = ()
    metadata: Mapping[str, str] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        student_id = self.student_id.strip()
        full_name = " ".join(self.full_name.split())
        if not student_id:
            raise ValueError("student_id cannot be empty")
        if not full_name:
            raise ValueError(f"full_name cannot be empty for {student_id}")
        object.__setattr__(self, "student_id", student_id)
        object.__setattr__(self, "full_name", full_name)
        object.__setattr__(
            self,
            "reference_images",
            tuple(Path(path) for path in self.reference_images),
        )


@dataclass(frozen=True, slots=True)
class FaceMatch:
    """Open-set gallery result.

    A best candidate is accepted only when both the absolute cosine threshold
    and the best-vs-runner-up margin pass.
    """

    student_id: str | None
    full_name: str | None
    similarity: float
    second_similarity: float | None
    margin: float
    accepted: bool
    reason: str


def _normalize_header(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value)
    ascii_text = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return "".join(ch for ch in ascii_text.casefold() if ch.isalnum())


def _find_column(
    normalized_columns: Mapping[str, str],
    aliases: Sequence[str],
) -> str | None:
    for alias in aliases:
        original = normalized_columns.get(_normalize_header(alias))
        if original is not None:
            return original
    return None


def _discover_reference_images(root: Path, student_id: str) -> tuple[Path, ...]:
    candidates: list[Path] = []
    student_directory = root / student_id
    if student_directory.is_dir():
        candidates.extend(
            path
            for path in student_directory.rglob("*")
            if path.is_file() and path.suffix.casefold() in _IMAGE_SUFFIXES
        )
    candidates.extend(
        path
        for path in root.glob(f"{student_id}.*")
        if path.is_file() and path.suffix.casefold() in _IMAGE_SUFFIXES
    )
    return tuple(sorted(set(path.resolve() for path in candidates)))


def record_is_active(record: StudentRecord) -> bool:
    """Return whether a roster entry is eligible for live recognition."""

    raw_value = str(record.metadata.get("active", "1")).strip().casefold()
    return raw_value not in {"0", "false", "no", "inactive", "disabled"}


def load_roster_csv(
    csv_path: str | Path,
    *,
    reference_root: str | Path | None = None,
) -> list[StudentRecord]:
    """Load a UTF-8 roster with practical English/Vietnamese column aliases.

    Required columns are an ID (for example ``MSSV`` or ``student_id``) and a
    full name, or separate surname/middle/given-name columns.  Reference image
    paths may be separated with ``;`` or ``|``.  If absent, images are
    discovered under ``reference_root/<MSSV>/`` and ``reference_root/MSSV.*``.
    """

    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"Roster CSV not found: {path}")
    image_root = Path(reference_root) if reference_root is not None else path.parent
    image_root = image_root.resolve()

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"Roster has no header: {path}")
        normalized = {_normalize_header(name): name for name in reader.fieldnames}
        id_column = _find_column(
            normalized,
            ("student_id", "mssv", "ma_so_sinh_vien", "student code", "id"),
        )
        full_name_column = _find_column(
            normalized,
            ("full_name", "ho_ten", "họ tên", "họ và tên", "name"),
        )
        surname_column = _find_column(normalized, ("surname", "last_name", "ho", "họ"))
        middle_column = _find_column(
            normalized,
            ("middle_name", "ten_dem", "tên đệm", "middle"),
        )
        given_column = _find_column(
            normalized,
            ("given_name", "first_name", "ten", "tên"),
        )
        image_column = _find_column(
            normalized,
            (
                "reference_images",
                "reference_image",
                "image_paths",
                "images",
                "image",
                "anh",
                "ảnh",
                "anh_khuon_mat",
            ),
        )
        if id_column is None:
            raise ValueError("Roster needs an MSSV/student_id column")
        if full_name_column is None and not (surname_column or given_column):
            raise ValueError("Roster needs a full-name or component name columns")

        records: list[StudentRecord] = []
        seen_ids: set[str] = set()
        for line_number, row in enumerate(reader, start=2):
            student_id = (row.get(id_column) or "").strip()
            if not student_id and not any((value or "").strip() for value in row.values()):
                continue
            if not student_id:
                raise ValueError(f"Missing student ID at roster line {line_number}")
            if student_id in seen_ids:
                raise ValueError(f"Duplicate student ID in roster: {student_id}")

            if full_name_column is not None:
                full_name = row.get(full_name_column) or ""
            else:
                full_name = " ".join(
                    filter(
                        None,
                        (
                            (row.get(surname_column) or "") if surname_column else "",
                            (row.get(middle_column) or "") if middle_column else "",
                            (row.get(given_column) or "") if given_column else "",
                        ),
                    )
                )

            references: tuple[Path, ...]
            raw_references = (row.get(image_column) or "").strip() if image_column else ""
            if raw_references:
                resolved: list[Path] = []
                for raw_reference in re.split(r"[;|]", raw_references):
                    raw_reference = raw_reference.strip()
                    if not raw_reference:
                        continue
                    reference = Path(raw_reference)
                    if not reference.is_absolute():
                        reference = image_root / reference
                    resolved.append(reference.resolve())
                references = tuple(resolved)
            else:
                references = _discover_reference_images(image_root, student_id)

            metadata = {
                str(key): str(value or "")
                for key, value in row.items()
                if key is not None
            }
            records.append(
                StudentRecord(
                    student_id=student_id,
                    full_name=full_name,
                    reference_images=references,
                    metadata=metadata,
                )
            )
            seen_ids.add(student_id)
    if not records:
        raise ValueError(f"Roster contains no students: {path}")
    return records


class SFaceEncoder:
    """Thread-safe OpenCV ``FaceRecognizerSF`` feature extractor."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        backend_id: int = 0,
        target_id: int = 0,
        recognizer: Any | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        self._lock = threading.Lock()
        if recognizer is None:
            if not self.model_path.is_file():
                raise FileNotFoundError(f"SFace model not found: {self.model_path}")
            try:
                import cv2  # type: ignore
            except ImportError as error:
                raise RuntimeError("opencv-python is required for SFace") from error
            recognizer = cv2.FaceRecognizerSF.create(
                str(self.model_path),
                "",
                int(backend_id),
                int(target_id),
            )
        self.recognizer = recognizer

    def extract(self, frame: np.ndarray, face: FaceDetection) -> np.ndarray:
        """Align a YuNet face and return its unit-length SFace embedding."""

        with self._lock:
            aligned = self.recognizer.alignCrop(frame, face.to_yunet_row())
            feature = self.recognizer.feature(aligned)
        return normalize_embedding(feature)

    def extract_aligned(self, aligned_face: np.ndarray) -> np.ndarray:
        with self._lock:
            feature = self.recognizer.feature(aligned_face)
        return normalize_embedding(feature)


class _FaceDetector(Protocol):
    def detect(self, frame: np.ndarray) -> list[FaceDetection]: ...


class _FaceEncoder(Protocol):
    def extract(self, frame: np.ndarray, face: FaceDetection) -> np.ndarray: ...


def _opencv_image_loader(path: Path) -> np.ndarray | None:
    try:
        import cv2  # type: ignore
    except ImportError as error:
        raise RuntimeError("opencv-python is required to load roster images") from error
    return cv2.imread(str(path))


class FaceGallery:
    """SFace centroid gallery with open-set threshold and ambiguity margin."""

    def __init__(
        self,
        records: Sequence[StudentRecord],
        centroids: Mapping[str, np.ndarray | Sequence[float]],
        *,
        cosine_threshold: float = DEFAULT_COSINE_THRESHOLD,
        cosine_margin: float = DEFAULT_COSINE_MARGIN,
        warnings: Sequence[str] = (),
        reference_counts: Mapping[str, int] | None = None,
        sparse_reference_threshold: float = SPARSE_REFERENCE_COSINE_THRESHOLD,
        sparse_reference_count: int = RECOMMENDED_REFERENCE_IMAGES,
    ) -> None:
        self.records = {record.student_id: record for record in records}
        if not 0.0 <= cosine_threshold <= 1.0:
            raise ValueError("cosine_threshold must be in [0, 1]")
        if not 0.0 <= cosine_margin <= 2.0:
            raise ValueError("cosine_margin must be in [0, 2]")
        if not 0.0 <= sparse_reference_threshold <= 1.0:
            raise ValueError("sparse_reference_threshold must be in [0, 1]")
        if sparse_reference_count < 1:
            raise ValueError("sparse_reference_count must be positive")
        self.cosine_threshold = float(cosine_threshold)
        self.cosine_margin = float(cosine_margin)
        self.sparse_reference_threshold = float(sparse_reference_threshold)
        self.sparse_reference_count = int(sparse_reference_count)
        self.warnings = tuple(warnings)
        self.reference_counts = {
            student_id: max(0, int(count))
            for student_id, count in (reference_counts or {}).items()
        }
        self.centroids: dict[str, np.ndarray] = {}
        embedding_size: int | None = None
        for student_id, centroid in centroids.items():
            if student_id not in self.records:
                raise ValueError(f"Centroid has no roster record: {student_id}")
            normalized = normalize_embedding(centroid)
            if embedding_size is None:
                embedding_size = normalized.size
            elif normalized.size != embedding_size:
                raise ValueError("All face centroids must use one embedding dimension")
            self.centroids[student_id] = normalized.copy()
        if not self.centroids:
            raise ValueError("Face gallery has no usable centroids")

    @classmethod
    def build(
        cls,
        records: Sequence[StudentRecord],
        face_detector: _FaceDetector,
        face_encoder: _FaceEncoder,
        *,
        image_loader: Callable[[Path], np.ndarray | None] = _opencv_image_loader,
        cosine_threshold: float = DEFAULT_COSINE_THRESHOLD,
        cosine_margin: float = DEFAULT_COSINE_MARGIN,
        strict: bool = False,
    ) -> "FaceGallery":
        """Build one normalized centroid from all valid references per student."""

        centroids: dict[str, np.ndarray] = {}
        reference_counts: dict[str, int] = {}
        warnings: list[str] = []
        for record in records:
            if not record_is_active(record):
                continue
            features: list[np.ndarray] = []
            if not record.reference_images:
                message = f"{record.student_id}: no reference images"
                if strict:
                    raise ValueError(message)
                warnings.append(message)
                continue
            for image_path in record.reference_images:
                image = image_loader(image_path)
                if image is None:
                    message = f"{record.student_id}: cannot read {image_path}"
                    if strict:
                        raise ValueError(message)
                    warnings.append(message)
                    continue
                faces = face_detector.detect(image)
                if not faces:
                    message = f"{record.student_id}: no face in {image_path}"
                    if strict:
                        raise ValueError(message)
                    warnings.append(message)
                    continue
                # Prefer a large, confident reference face when a photo contains
                # bystanders.  This is deterministic for equal scores.
                face = max(
                    faces,
                    key=lambda candidate: (
                        area(candidate.bbox) * max(candidate.score, 0.0),
                        candidate.score,
                    ),
                )
                try:
                    features.append(normalize_embedding(face_encoder.extract(image, face)))
                except (RuntimeError, ValueError) as error:
                    message = f"{record.student_id}: {image_path}: {error}"
                    if strict:
                        raise ValueError(message) from error
                    warnings.append(message)
            if features:
                dimensions = {feature.size for feature in features}
                if len(dimensions) != 1:
                    raise ValueError(
                        f"{record.student_id}: inconsistent reference embedding dimensions"
                    )
                centroids[record.student_id] = normalize_embedding(
                    np.mean(np.stack(features), axis=0)
                )
                reference_counts[record.student_id] = len(features)
                if len(features) < RECOMMENDED_REFERENCE_IMAGES:
                    warnings.append(
                        f"{record.student_id}: only {len(features)} usable reference "
                        f"image(s); recommend at least {RECOMMENDED_REFERENCE_IMAGES}"
                    )
            else:
                warnings.append(f"{record.student_id}: no usable face embeddings")
        return cls(
            records,
            centroids,
            cosine_threshold=cosine_threshold,
            cosine_margin=cosine_margin,
            warnings=warnings,
            reference_counts=reference_counts,
        )

    @classmethod
    def from_roster_csv(
        cls,
        csv_path: str | Path,
        face_detector: _FaceDetector,
        face_encoder: _FaceEncoder,
        *,
        reference_root: str | Path | None = None,
        image_loader: Callable[[Path], np.ndarray | None] = _opencv_image_loader,
        cosine_threshold: float = DEFAULT_COSINE_THRESHOLD,
        cosine_margin: float = DEFAULT_COSINE_MARGIN,
        strict: bool = False,
    ) -> "FaceGallery":
        records = load_roster_csv(csv_path, reference_root=reference_root)
        return cls.build(
            records,
            face_detector,
            face_encoder,
            image_loader=image_loader,
            cosine_threshold=cosine_threshold,
            cosine_margin=cosine_margin,
            strict=strict,
        )

    def match(self, embedding: np.ndarray | Sequence[float]) -> FaceMatch:
        query = normalize_embedding(embedding)
        scored: list[tuple[float, str]] = []
        for student_id, centroid in self.centroids.items():
            if query.shape != centroid.shape:
                raise ValueError(
                    f"Query dimension {query.size} does not match gallery {centroid.size}"
                )
            scored.append((float(np.dot(query, centroid)), student_id))
        scored.sort(key=lambda item: (-item[0], item[1]))
        best_similarity, best_id = scored[0]
        second_similarity = scored[1][0] if len(scored) > 1 else None
        margin = (
            best_similarity - second_similarity
            if second_similarity is not None
            else math.inf
        )
        reference_count = self.reference_counts.get(best_id)
        effective_threshold = self.cosine_threshold
        if reference_count is not None and reference_count < self.sparse_reference_count:
            effective_threshold = max(
                effective_threshold,
                self.sparse_reference_threshold,
            )
        above_threshold = best_similarity >= effective_threshold
        unambiguous = margin >= self.cosine_margin
        accepted = above_threshold and unambiguous
        if not above_threshold:
            reason = "below_threshold"
        elif not unambiguous:
            reason = "ambiguous_margin"
        else:
            reason = "matched"
        record = self.records[best_id]
        return FaceMatch(
            student_id=best_id if accepted else None,
            full_name=record.full_name if accepted else None,
            similarity=best_similarity,
            second_similarity=second_similarity,
            margin=margin,
            accepted=accepted,
            reason=reason,
        )

    def recognize(
        self,
        frame: np.ndarray,
        face: FaceDetection,
        encoder: _FaceEncoder,
    ) -> FaceMatch:
        return self.match(encoder.extract(frame, face))
