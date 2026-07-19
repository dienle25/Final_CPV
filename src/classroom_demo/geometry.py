"""Small, dependency-free geometry helpers for the classroom pipeline.

All boxes use ``(x1, y1, x2, y2)`` in image pixels.  The functions accept
integers or floats and deliberately return floats so detector, tracker and
association code can share one representation without repeated rounding.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from typing import TypeAlias


BBox: TypeAlias = tuple[float, float, float, float]


def as_bbox(values: Sequence[float]) -> BBox:
    """Validate and normalize a four-value box.

    Coordinates are reordered when a caller supplies the corners backwards.
    Non-finite values are rejected because they otherwise poison IoU matching.
    """

    if len(values) != 4:
        raise ValueError(f"A bounding box needs four values, got {len(values)}")
    x1, y1, x2, y2 = (float(value) for value in values)
    if not all(math.isfinite(value) for value in (x1, y1, x2, y2)):
        raise ValueError(f"Bounding box contains a non-finite value: {values!r}")
    return (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))


def width(box: Sequence[float]) -> float:
    x1, _y1, x2, _y2 = as_bbox(box)
    return max(0.0, x2 - x1)


def height(box: Sequence[float]) -> float:
    _x1, y1, _x2, y2 = as_bbox(box)
    return max(0.0, y2 - y1)


def area(box: Sequence[float]) -> float:
    """Return box area, or zero for a degenerate box."""

    normalized = as_bbox(box)
    return width(normalized) * height(normalized)


def center(box: Sequence[float]) -> tuple[float, float]:
    x1, y1, x2, y2 = as_bbox(box)
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def intersection(box_a: Sequence[float], box_b: Sequence[float]) -> BBox:
    ax1, ay1, ax2, ay2 = as_bbox(box_a)
    bx1, by1, bx2, by2 = as_bbox(box_b)
    x1, y1 = max(ax1, bx1), max(ay1, by1)
    x2, y2 = min(ax2, bx2), min(ay2, by2)
    if x2 <= x1 or y2 <= y1:
        return (x1, y1, x1, y1)
    return (x1, y1, x2, y2)


def intersection_area(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    return area(intersection(box_a, box_b))


def iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    """Intersection over union in the closed interval ``[0, 1]``."""

    intersection_value = intersection_area(box_a, box_b)
    union = area(box_a) + area(box_b) - intersection_value
    return intersection_value / union if union > 0.0 else 0.0


def containment(inner: Sequence[float], outer: Sequence[float]) -> float:
    """Fraction of ``inner`` covered by ``outer``."""

    inner_area = area(inner)
    if inner_area <= 0.0:
        return 0.0
    return intersection_area(inner, outer) / inner_area


def horizontal_overlap(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    """Horizontal overlap divided by the narrower box width."""

    ax1, _ay1, ax2, _ay2 = as_bbox(box_a)
    bx1, _by1, bx2, _by2 = as_bbox(box_b)
    overlap = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    denominator = min(ax2 - ax1, bx2 - bx1)
    return overlap / denominator if denominator > 0.0 else 0.0


def point_in_box(
    point: tuple[float, float],
    box: Sequence[float],
    *,
    inclusive: bool = True,
) -> bool:
    px, py = point
    x1, y1, x2, y2 = as_bbox(box)
    if inclusive:
        return x1 <= px <= x2 and y1 <= py <= y2
    return x1 < px < x2 and y1 < py < y2


def clip_box(box: Sequence[float], image_width: int, image_height: int) -> BBox:
    """Clip a box to an image while preserving the ``xyxy`` invariant."""

    if image_width <= 0 or image_height <= 0:
        raise ValueError("Image width and height must be positive")
    x1, y1, x2, y2 = as_bbox(box)
    return (
        min(max(x1, 0.0), float(image_width)),
        min(max(y1, 0.0), float(image_height)),
        min(max(x2, 0.0), float(image_width)),
        min(max(y2, 0.0), float(image_height)),
    )


def expand_box(
    box: Sequence[float],
    *,
    x_ratio: float = 0.0,
    y_ratio: float = 0.0,
) -> BBox:
    """Expand on every side by a fraction of the original width/height."""

    if x_ratio < 0.0 or y_ratio < 0.0:
        raise ValueError("Expansion ratios cannot be negative")
    x1, y1, x2, y2 = as_bbox(box)
    dx = (x2 - x1) * x_ratio
    dy = (y2 - y1) * y_ratio
    return (x1 - dx, y1 - dy, x2 + dx, y2 + dy)


def upper_region(box: Sequence[float], ratio: float = 0.62) -> BBox:
    """Return the upper portion of a person box used for head association."""

    if not 0.0 < ratio <= 1.0:
        raise ValueError("Upper-region ratio must be in (0, 1]")
    x1, y1, x2, y2 = as_bbox(box)
    return (x1, y1, x2, y1 + (y2 - y1) * ratio)


def normalized_center_distance(
    box_a: Sequence[float],
    box_b: Sequence[float],
    *,
    reference: Sequence[float] | None = None,
) -> float:
    """Center distance divided by the diagonal of ``reference``.

    The value may exceed one when boxes are far apart.  A degenerate reference
    returns infinity instead of raising during a live frame.
    """

    ax, ay = center(box_a)
    bx, by = center(box_b)
    ref = as_bbox(reference if reference is not None else box_a)
    diagonal = math.hypot(width(ref), height(ref))
    return math.hypot(ax - bx, ay - by) / diagonal if diagonal > 0.0 else math.inf


def greedy_one_to_one(
    candidates: Iterable[tuple[float, int, int]],
) -> dict[int, int]:
    """Choose deterministic maximum-score one-to-one assignments.

    Candidate tuples are ``(score, left_index, right_index)``.  Ties prefer the
    lower indexes, which keeps unit tests and recorded demos reproducible.
    """

    assignments: dict[int, int] = {}
    used_right: set[int] = set()
    ordered = sorted(candidates, key=lambda item: (-item[0], item[1], item[2]))
    for _score, left_index, right_index in ordered:
        if left_index in assignments or right_index in used_right:
            continue
        assignments[left_index] = right_index
        used_right.add(right_index)
    return assignments


def non_max_suppression(
    boxes: Sequence[Sequence[float]],
    scores: Sequence[float],
    iou_threshold: float,
) -> list[int]:
    """Return retained indexes using score-ordered greedy NMS."""

    if len(boxes) != len(scores):
        raise ValueError("boxes and scores must have equal length")
    if not 0.0 <= iou_threshold <= 1.0:
        raise ValueError("IoU threshold must be in [0, 1]")
    ordered = sorted(range(len(scores)), key=lambda idx: (-float(scores[idx]), idx))
    kept: list[int] = []
    while ordered:
        current = ordered.pop(0)
        kept.append(current)
        ordered = [
            index
            for index in ordered
            if iou(boxes[current], boxes[index]) <= iou_threshold
        ]
    return kept
