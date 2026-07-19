"""Person/face/head association and four-state helmet heuristic.

The bundled helmet model has ``helmet``, ``no_helmet`` and ``rider`` classes;
it does not contain a trained "worn incorrectly" class.  ``Đội sai`` below is therefore
an explicit geometry heuristic: a helmet-like box is present but is displaced
from the face/head position.  ``Chưa rõ`` is used whenever the evidence is not
strong enough, so the demo never presents that heuristic as model ground truth.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Protocol, TypeVar

from .detectors import Detection, FaceDetection
from .geometry import (
    BBox,
    area,
    center,
    containment,
    expand_box,
    greedy_one_to_one,
    horizontal_overlap,
    iou,
    normalized_center_distance,
    point_in_box,
    upper_region,
    width,
    height,
)


HELMET_LABELS = {"helmet", "withhelmet", "with_helmet", "with helmet"}
NO_HELMET_LABELS = {
    "nohelmet",
    "no_helmet",
    "no helmet",
    "withouthelmet",
    "without_helmet",
    "without helmet",
}


class HelmetStatus(str, Enum):
    WORN_CORRECTLY = "Đội đúng"
    WORN_INCORRECTLY = "Đội sai"
    NO_HELMET = "Không mũ"
    UNKNOWN = "Chưa rõ"


class BoxLike(Protocol):
    @property
    def bbox(self) -> BBox: ...


PersonT = TypeVar("PersonT", bound=BoxLike)


def normalize_label(label: str) -> str:
    return "".join(
        character
        for character in label.casefold().strip()
        if character.isalnum() or character == "_"
    )


_NORMALIZED_HELMET_LABELS = {normalize_label(label) for label in HELMET_LABELS}
_NORMALIZED_NO_HELMET_LABELS = {
    normalize_label(label) for label in NO_HELMET_LABELS
}


def helmet_role(label: str) -> str | None:
    normalized = normalize_label(label)
    if normalized in _NORMALIZED_HELMET_LABELS:
        return "helmet"
    if normalized in _NORMALIZED_NO_HELMET_LABELS:
        return "no_helmet"
    return None


@dataclass(frozen=True, slots=True)
class HelmetHeuristicConfig:
    minimum_detection_score: float = 0.25
    require_face_for_correct_wear: bool = True
    minimum_horizontal_overlap: float = 0.55
    maximum_horizontal_center_offset: float = 0.55
    maximum_head_center_y: float = 0.60
    maximum_head_top_y: float = 0.40
    minimum_head_bottom_y: float = 0.05
    maximum_head_bottom_y: float = 1.35
    minimum_head_face_area_ratio: float = 0.12
    maximum_head_face_area_ratio: float = 4.00
    minimum_incorrect_detection_score: float = 0.55
    conflict_minimum_score: float = 0.35
    conflict_iou_threshold: float = 0.50
    relaxed_minimum_horizontal_overlap: float = 0.35
    relaxed_maximum_horizontal_center_offset: float = 0.80
    relaxed_maximum_head_center_y: float = 0.85
    relaxed_maximum_head_top_y: float = 0.65
    relaxed_minimum_head_bottom_y: float = -0.15
    relaxed_maximum_head_bottom_y: float = 1.60
    relaxed_minimum_head_face_area_ratio: float = 0.08
    relaxed_maximum_head_face_area_ratio: float = 5.00

    def __post_init__(self) -> None:
        if not 0.0 <= self.minimum_detection_score <= 1.0:
            raise ValueError("minimum_detection_score must be in [0, 1]")
        if self.minimum_horizontal_overlap < 0.0:
            raise ValueError("minimum_horizontal_overlap cannot be negative")
        if self.maximum_horizontal_center_offset < 0.0:
            raise ValueError("maximum_horizontal_center_offset cannot be negative")
        if self.minimum_head_face_area_ratio < 0.0:
            raise ValueError("minimum_head_face_area_ratio cannot be negative")
        if self.maximum_head_face_area_ratio <= self.minimum_head_face_area_ratio:
            raise ValueError("maximum area ratio must exceed minimum area ratio")
        if not 0.0 <= self.minimum_incorrect_detection_score <= 1.0:
            raise ValueError("minimum incorrect score must be in [0, 1]")
        if not 0.0 <= self.conflict_minimum_score <= 1.0:
            raise ValueError("conflict minimum score must be in [0, 1]")
        if not 0.0 <= self.conflict_iou_threshold <= 1.0:
            raise ValueError("conflict IoU threshold must be in [0, 1]")
        if self.relaxed_minimum_horizontal_overlap > self.minimum_horizontal_overlap:
            raise ValueError("relaxed overlap cannot be stricter than correct overlap")
        if (
            self.relaxed_minimum_head_face_area_ratio
            >= self.relaxed_maximum_head_face_area_ratio
        ):
            raise ValueError("relaxed area-ratio range is invalid")


@dataclass(frozen=True, slots=True)
class PersonAssociation:
    person_index: int
    person: BoxLike
    face_index: int | None
    face: FaceDetection | None
    head_index: int | None
    head: Detection | None
    status: HelmetStatus
    confidence: float
    reason: str


def _face_person_score(person_box: BBox, face: FaceDetection) -> float | None:
    region = expand_box(upper_region(person_box, 0.68), x_ratio=0.10, y_ratio=0.05)
    face_center = center(face.bbox)
    coverage = containment(face.bbox, expand_box(person_box, x_ratio=0.05, y_ratio=0.02))
    if coverage < 0.45 or not point_in_box(face_center, region):
        return None
    person_area = area(person_box)
    face_ratio = area(face.bbox) / person_area if person_area > 0.0 else 0.0
    if not 0.002 <= face_ratio <= 0.85:
        return None
    # A close-up phone frame can make a valid face occupy more than 45% of the
    # person's detector box (CE190579 enrollment: about 64%). Only allow that
    # extended range when the face is almost fully contained, which prevents
    # the relaxed path from stealing a neighbouring face in a crowded frame.
    if face_ratio > 0.45 and coverage < 0.85:
        return None
    distance = normalized_center_distance(face.bbox, region, reference=person_box)
    return 1.20 * coverage + face.score - 0.25 * distance


def associate_faces_to_people(
    people: Sequence[BoxLike],
    faces: Sequence[FaceDetection],
) -> dict[int, int]:
    """Return one-to-one ``person_index -> face_index`` assignments."""

    candidates: list[tuple[float, int, int]] = []
    for person_index, person in enumerate(people):
        for face_index, face in enumerate(faces):
            score = _face_person_score(person.bbox, face)
            if score is not None:
                candidates.append((score, person_index, face_index))
    return greedy_one_to_one(candidates)


def _head_person_score(
    person_box: BBox,
    head: Detection,
    face: FaceDetection | None,
) -> float | None:
    if helmet_role(head.label) is None:
        return None
    region = expand_box(upper_region(person_box, 0.70), x_ratio=0.12, y_ratio=0.08)
    head_center = center(head.bbox)
    coverage = containment(head.bbox, expand_box(person_box, x_ratio=0.08, y_ratio=0.03))
    if coverage < 0.35 or not point_in_box(head_center, region):
        return None
    person_area = area(person_box)
    head_ratio = area(head.bbox) / person_area if person_area > 0.0 else 0.0
    if not 0.001 <= head_ratio <= 0.55:
        return None
    score = head.score + 0.80 * coverage
    if face is not None:
        score += 0.40 * horizontal_overlap(head.bbox, face.bbox)
        score -= 0.25 * normalized_center_distance(
            head.bbox,
            face.bbox,
            reference=face.bbox,
        )
    else:
        score -= 0.15 * normalized_center_distance(
            head.bbox,
            region,
            reference=person_box,
        )
    return score


def associate_heads_to_people(
    people: Sequence[BoxLike],
    heads: Sequence[Detection],
    *,
    face_matches: dict[int, int] | None = None,
    faces: Sequence[FaceDetection] = (),
) -> dict[int, int]:
    """Return one-to-one ``person_index -> helmet-head index`` assignments."""

    candidates: list[tuple[float, int, int]] = []
    face_matches = face_matches or {}
    for person_index, person in enumerate(people):
        face_index = face_matches.get(person_index)
        face = faces[face_index] if face_index is not None else None
        for head_index, head in enumerate(heads):
            score = _head_person_score(person.bbox, head, face)
            if score is not None:
                candidates.append((score, person_index, head_index))
    return greedy_one_to_one(candidates)


def helmet_alignment_is_correct(
    face_box: BBox,
    helmet_box: BBox,
    config: HelmetHeuristicConfig,
) -> bool:
    """Check whether a helmet-like box is plausibly centered over the face."""

    face_width = width(face_box)
    face_height = height(face_box)
    face_area = area(face_box)
    if face_width <= 0.0 or face_height <= 0.0 or face_area <= 0.0:
        return False
    face_x, _face_y = center(face_box)
    helmet_x, helmet_y = center(helmet_box)
    _fx1, face_top, _fx2, face_bottom = face_box
    _hx1, helmet_top, _hx2, helmet_bottom = helmet_box
    area_ratio = area(helmet_box) / face_area
    horizontal_offset = abs(helmet_x - face_x) / face_width
    head_center_y = (helmet_y - face_top) / face_height
    head_top_y = (helmet_top - face_top) / face_height
    head_bottom_y = (helmet_bottom - face_top) / face_height
    return all(
        (
            horizontal_overlap(face_box, helmet_box)
            >= config.minimum_horizontal_overlap,
            horizontal_offset <= config.maximum_horizontal_center_offset,
            head_center_y <= config.maximum_head_center_y,
            head_top_y <= config.maximum_head_top_y,
            head_bottom_y >= config.minimum_head_bottom_y,
            head_bottom_y <= config.maximum_head_bottom_y,
            config.minimum_head_face_area_ratio
            <= area_ratio
            <= config.maximum_head_face_area_ratio,
            helmet_bottom <= face_bottom + 0.35 * face_height,
        )
    )


def helmet_alignment_is_plausible(
    face_box: BBox,
    helmet_box: BBox,
    config: HelmetHeuristicConfig,
) -> bool:
    """Return true for borderline-but-plausible head geometry.

    A model box that barely misses the strict correct-wear limits is ambiguous,
    not proof that the helmet is worn incorrectly.
    """

    face_width = width(face_box)
    face_height = height(face_box)
    face_area = area(face_box)
    if face_width <= 0.0 or face_height <= 0.0 or face_area <= 0.0:
        return False
    face_x, _ = center(face_box)
    helmet_x, helmet_y = center(helmet_box)
    _fx1, face_top, _fx2, face_bottom = face_box
    _hx1, helmet_top, _hx2, helmet_bottom = helmet_box
    area_ratio = area(helmet_box) / face_area
    horizontal_offset = abs(helmet_x - face_x) / face_width
    head_center_y = (helmet_y - face_top) / face_height
    head_top_y = (helmet_top - face_top) / face_height
    head_bottom_y = (helmet_bottom - face_top) / face_height
    return all(
        (
            horizontal_overlap(face_box, helmet_box)
            >= config.relaxed_minimum_horizontal_overlap,
            horizontal_offset <= config.relaxed_maximum_horizontal_center_offset,
            head_center_y <= config.relaxed_maximum_head_center_y,
            head_top_y <= config.relaxed_maximum_head_top_y,
            head_bottom_y >= config.relaxed_minimum_head_bottom_y,
            head_bottom_y <= config.relaxed_maximum_head_bottom_y,
            config.relaxed_minimum_head_face_area_ratio
            <= area_ratio
            <= config.relaxed_maximum_head_face_area_ratio,
            helmet_bottom <= face_bottom + 0.55 * face_height,
        )
    )


def classify_helmet_status(
    face: FaceDetection | None,
    head: Detection | None,
    *,
    config: HelmetHeuristicConfig | None = None,
) -> tuple[HelmetStatus, str]:
    """Classify one associated person and return ``(status, reason)``."""

    config = config or HelmetHeuristicConfig()
    if head is None:
        return HelmetStatus.UNKNOWN, "no_head_evidence"
    if head.score < config.minimum_detection_score:
        return HelmetStatus.UNKNOWN, "low_head_confidence"
    role = helmet_role(head.label)
    if role == "no_helmet":
        if face is None:
            return HelmetStatus.UNKNOWN, "no_helmet_without_face_geometry"
        if not helmet_alignment_is_plausible(face.bbox, head.bbox, config):
            return HelmetStatus.UNKNOWN, "no_helmet_misaligned_with_face"
        return HelmetStatus.NO_HELMET, "detector_no_helmet"
    if role != "helmet":
        return HelmetStatus.UNKNOWN, "unsupported_head_label"
    if face is None:
        if config.require_face_for_correct_wear:
            return HelmetStatus.UNKNOWN, "helmet_without_face_geometry"
        return HelmetStatus.WORN_CORRECTLY, "helmet_detector_only"
    if helmet_alignment_is_correct(face.bbox, head.bbox, config):
        return HelmetStatus.WORN_CORRECTLY, "helmet_aligned_with_face"
    if helmet_alignment_is_plausible(face.bbox, head.bbox, config):
        return HelmetStatus.UNKNOWN, "helmet_alignment_borderline"
    if head.score < config.minimum_incorrect_detection_score:
        return HelmetStatus.UNKNOWN, "helmet_misaligned_low_confidence"
    return HelmetStatus.WORN_INCORRECTLY, "helmet_misaligned_with_face"


def _has_conflicting_head_evidence(
    person: BoxLike,
    face: FaceDetection | None,
    selected_index: int,
    heads: Sequence[Detection],
    config: HelmetHeuristicConfig,
) -> bool:
    selected = heads[selected_index]
    selected_role = helmet_role(selected.label)
    if selected_role not in {"helmet", "no_helmet"}:
        return False
    for index, candidate in enumerate(heads):
        if index == selected_index:
            continue
        candidate_role = helmet_role(candidate.label)
        if candidate_role is None or candidate_role == selected_role:
            continue
        if candidate.score < config.conflict_minimum_score:
            continue
        if iou(selected.bbox, candidate.bbox) < config.conflict_iou_threshold:
            continue
        if _head_person_score(person.bbox, candidate, face) is not None:
            return True
    return False


def associate_people_faces_heads(
    people: Sequence[BoxLike],
    faces: Sequence[FaceDetection],
    heads: Sequence[Detection],
    *,
    config: HelmetHeuristicConfig | None = None,
) -> list[PersonAssociation]:
    """Associate all three detector layers and emit one result per person."""

    config = config or HelmetHeuristicConfig()
    face_matches = associate_faces_to_people(people, faces)
    head_matches = associate_heads_to_people(
        people,
        heads,
        face_matches=face_matches,
        faces=faces,
    )
    associations: list[PersonAssociation] = []
    for person_index, person in enumerate(people):
        face_index = face_matches.get(person_index)
        face = faces[face_index] if face_index is not None else None
        head_index = head_matches.get(person_index)
        head = heads[head_index] if head_index is not None else None
        if (
            head is not None
            and head_index is not None
            and _has_conflicting_head_evidence(
                person,
                face,
                head_index,
                heads,
                config,
            )
        ):
            status, reason = HelmetStatus.UNKNOWN, "conflicting_helmet_evidence"
        else:
            status, reason = classify_helmet_status(face, head, config=config)
        confidence = head.score if head is not None else 0.0
        associations.append(
            PersonAssociation(
                person_index=person_index,
                person=person,
                face_index=face_index,
                face=face,
                head_index=head_index,
                head=head,
                status=status,
                confidence=confidence,
                reason=reason,
            )
        )
    return associations


class HelmetStatusEstimator:
    """State-free facade used by a frame-processing engine."""

    def __init__(self, config: HelmetHeuristicConfig | None = None) -> None:
        self.config = config or HelmetHeuristicConfig()

    def classify(
        self,
        face: FaceDetection | None,
        head: Detection | None,
    ) -> tuple[HelmetStatus, str]:
        return classify_helmet_status(face, head, config=self.config)

    def associate(
        self,
        people: Sequence[BoxLike],
        faces: Sequence[FaceDetection],
        heads: Sequence[Detection],
    ) -> list[PersonAssociation]:
        return associate_people_faces_heads(
            people,
            faces,
            heads,
            config=self.config,
        )
