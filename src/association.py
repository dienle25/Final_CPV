"""Rider-to-head association and temporal label smoothing.

The detector produces independent boxes for ``helmet``, ``no_helmet`` and
``rider``. This module converts those independent detections into a rider-level
status by matching each head box to at most one tracked rider. The matching is
kept geometric and deterministic so it is easy to explain during a defense.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass


HELMET_ALIASES = {"helmet", "withhelmet", "with_helmet", "with helmet"}
NO_HELMET_ALIASES = {
    "nohelmet",
    "no_helmet",
    "no helmet",
    "withouthelmet",
    "without_helmet",
    "without helmet",
    "riderwithouthelmet",
    "rider_without_helmet",
}
RIDER_ALIASES = {"rider", "motorcyclist", "motorcycle_rider", "person_on_motorcycle"}


def normalize_label(name: str) -> str:
    """Normalize a class label while preserving underscores."""
    return "".join(ch for ch in name.lower().strip() if ch.isalnum() or ch == "_")


def is_alias(name: str, aliases: set[str]) -> bool:
    """Return whether a class label belongs to an alias set."""
    normalized = normalize_label(name)
    return normalized in {normalize_label(alias) for alias in aliases}


@dataclass(frozen=True)
class Detection:
    """One YOLO detection converted to simple Python values."""

    bbox: tuple[int, int, int, int]
    confidence: float
    class_id: int
    class_name: str
    track_id: int = -1

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @property
    def width(self) -> float:
        return float(max(1, self.bbox[2] - self.bbox[0]))

    @property
    def height(self) -> float:
        return float(max(1, self.bbox[3] - self.bbox[1]))

    @property
    def helmet_status(self) -> str | None:
        if is_alias(self.class_name, HELMET_ALIASES):
            return "helmet"
        if is_alias(self.class_name, NO_HELMET_ALIASES):
            return "no_helmet"
        return None


@dataclass
class RiderVoteState:
    """Recent helmet decisions for one ByteTrack rider ID."""

    votes: deque[str]
    last_seen_frame: int = -1
    last_vote_frame: int = -1


def point_in_rider_zone(
    point: tuple[float, float],
    rider_box: tuple[int, int, int, int],
    *,
    upper_rider_ratio: float = 0.55,
    match_padding: float = 0.08,
) -> bool:
    """Check whether a head center lies in the upper portion of a rider box."""
    px, py = point
    x1, y1, x2, y2 = rider_box
    width = max(1.0, float(x2 - x1))
    padded_x1 = x1 - width * match_padding
    padded_x2 = x2 + width * match_padding
    upper_y2 = y1 + (y2 - y1) * upper_rider_ratio
    return padded_x1 <= px <= padded_x2 and y1 <= py <= upper_y2


def _association_score(
    rider: Detection,
    head: Detection,
    *,
    upper_rider_ratio: float,
    match_padding: float,
) -> float | None:
    """Score a plausible pair; larger values are better."""
    if not point_in_rider_zone(
        head.center,
        rider.bbox,
        upper_rider_ratio=upper_rider_ratio,
        match_padding=match_padding,
    ):
        return None

    rx1, ry1, rx2, ry2 = rider.bbox
    rider_width = max(1.0, float(rx2 - rx1))
    rider_height = max(1.0, float(ry2 - ry1))
    rider_top_center_x = (rx1 + rx2) / 2.0
    expected_head_y = ry1 + rider_height * 0.18
    hx, hy = head.center

    normalized_horizontal = abs(hx - rider_top_center_x) / rider_width
    normalized_vertical = abs(hy - expected_head_y) / rider_height
    size_ratio = min(1.0, (head.width * head.height) / (rider_width * rider_height) * 15.0)

    return (
        head.confidence
        - 0.35 * normalized_horizontal
        - 0.20 * normalized_vertical
        + 0.05 * size_ratio
    )


def associate_heads_to_riders(
    riders: list[Detection],
    heads: list[Detection],
    *,
    upper_rider_ratio: float = 0.55,
    match_padding: float = 0.08,
) -> dict[int, Detection]:
    """Greedily create one-to-one head assignments keyed by rider track ID.

    A head detection cannot vote for multiple riders. This fixes the common
    overlap failure where a single ``no_helmet`` box is counted for two nearby
    rider boxes.
    """
    candidates: list[tuple[float, int, int]] = []
    for rider_index, rider in enumerate(riders):
        if rider.track_id < 0:
            continue
        for head_index, head in enumerate(heads):
            score = _association_score(
                rider,
                head,
                upper_rider_ratio=upper_rider_ratio,
                match_padding=match_padding,
            )
            if score is not None:
                candidates.append((score, rider_index, head_index))

    assignments: dict[int, Detection] = {}
    used_riders: set[int] = set()
    used_heads: set[int] = set()
    for _score, rider_index, head_index in sorted(candidates, reverse=True):
        if rider_index in used_riders or head_index in used_heads:
            continue
        rider = riders[rider_index]
        assignments[rider.track_id] = heads[head_index]
        used_riders.add(rider_index)
        used_heads.add(head_index)
    return assignments


def stable_label(
    votes: deque[str],
    *,
    min_votes: int = 4,
    min_ratio: float = 0.60,
) -> str:
    """Return a stable majority label or ``unknown``."""
    if not votes:
        return "unknown"
    counts = Counter(votes)
    top_label, top_count = counts.most_common(1)[0]
    if top_count < max(1, int(min_votes)):
        return "unknown"
    if top_count / len(votes) < float(min_ratio):
        return "unknown"
    return top_label
