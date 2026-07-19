"""A small deterministic IoU tracker with no external tracking dependency."""

from __future__ import annotations

import threading
from collections.abc import Sequence
from dataclasses import dataclass

from .detectors import Detection
from .geometry import BBox, greedy_one_to_one, iou


@dataclass(slots=True)
class _TrackState:
    track_id: int
    detection: Detection
    hits: int
    age: int
    missed: int
    last_frame: int


@dataclass(frozen=True, slots=True)
class TrackedDetection:
    """Public immutable view of a detection and its stable local ID."""

    track_id: int
    detection: Detection
    hits: int
    age: int
    missed: int = 0

    @property
    def bbox(self) -> BBox:
        return self.detection.bbox

    @property
    def score(self) -> float:
        return self.detection.score

    @property
    def class_id(self) -> int:
        return self.detection.class_id

    @property
    def label(self) -> str:
        return self.detection.label


class IoUTracker:
    """Greedy one-to-one IoU tracking for classroom person/face boxes.

    This intentionally omits motion models and ReID.  For a static classroom
    camera it provides explainable short-term IDs without ByteTrack/LAP.  Stale
    tracks remain eligible for ``max_missed`` updates and are then discarded.
    """

    def __init__(
        self,
        *,
        iou_threshold: float = 0.30,
        max_missed: int = 12,
        min_hits: int = 1,
        class_aware: bool = True,
        first_track_id: int = 1,
    ) -> None:
        if not 0.0 <= iou_threshold <= 1.0:
            raise ValueError("iou_threshold must be in [0, 1]")
        if max_missed < 0:
            raise ValueError("max_missed cannot be negative")
        if min_hits < 1:
            raise ValueError("min_hits must be at least one")
        if first_track_id < 0:
            raise ValueError("first_track_id cannot be negative")
        self.iou_threshold = float(iou_threshold)
        self.max_missed = int(max_missed)
        self.min_hits = int(min_hits)
        self.class_aware = bool(class_aware)
        self._next_track_id = int(first_track_id)
        self._frame_index = -1
        self._tracks: dict[int, _TrackState] = {}
        self._lock = threading.Lock()

    def reset(self, *, first_track_id: int = 1) -> None:
        with self._lock:
            self._tracks.clear()
            self._next_track_id = int(first_track_id)
            self._frame_index = -1

    def _new_track(self, detection: Detection, frame_index: int) -> _TrackState:
        track = _TrackState(
            track_id=self._next_track_id,
            detection=detection,
            hits=1,
            age=1,
            missed=0,
            last_frame=frame_index,
        )
        self._next_track_id += 1
        self._tracks[track.track_id] = track
        return track

    @staticmethod
    def _view(track: _TrackState) -> TrackedDetection:
        return TrackedDetection(
            track_id=track.track_id,
            detection=track.detection,
            hits=track.hits,
            age=track.age,
            missed=track.missed,
        )

    def update(
        self,
        detections: Sequence[Detection],
        *,
        frame_index: int | None = None,
        include_unmatched: bool = False,
    ) -> list[TrackedDetection]:
        """Update tracks and return observations for this frame.

        Results that correspond to current detections follow the input order.
        When ``include_unmatched`` is true, surviving missed tracks are appended
        after them; this is useful for short UI dropouts but should not be fed
        into face/helmet association as a fresh observation.
        """

        with self._lock:
            next_frame = self._frame_index + 1 if frame_index is None else int(frame_index)
            if next_frame <= self._frame_index:
                raise ValueError(
                    f"frame_index must increase ({next_frame} <= {self._frame_index})"
                )
            frame_step = next_frame - self._frame_index if self._frame_index >= 0 else 1
            self._frame_index = next_frame

            # An explicit frame jump represents skipped observations.  Expire
            # tracks before matching so an old box cannot reclaim an ID after a
            # gap larger than the configured tolerance.
            expired_before_match = [
                track_id
                for track_id, track in self._tracks.items()
                if next_frame - track.last_frame - 1 > self.max_missed
            ]
            for track_id in expired_before_match:
                self._tracks.pop(track_id, None)

            track_ids = sorted(self._tracks)
            candidates: list[tuple[float, int, int]] = []
            for track_position, track_id in enumerate(track_ids):
                track = self._tracks[track_id]
                for detection_index, detection in enumerate(detections):
                    if self.class_aware and detection.class_id != track.detection.class_id:
                        continue
                    overlap = iou(track.detection.bbox, detection.bbox)
                    if overlap >= self.iou_threshold:
                        candidates.append((overlap, track_position, detection_index))
            matches_by_position = greedy_one_to_one(candidates)
            matches = {
                track_ids[position]: detection_index
                for position, detection_index in matches_by_position.items()
            }

            matched_detections: set[int] = set(matches.values())
            detection_to_track: dict[int, _TrackState] = {}
            for track_id in track_ids:
                track = self._tracks[track_id]
                detection_index = matches.get(track_id)
                track.age += frame_step
                if detection_index is None:
                    track.missed += frame_step
                    continue
                track.detection = detections[detection_index]
                track.hits += 1
                track.missed = 0
                track.last_frame = next_frame
                detection_to_track[detection_index] = track

            stale_ids = [
                track_id
                for track_id, track in self._tracks.items()
                if track.missed > self.max_missed
            ]
            for track_id in stale_ids:
                self._tracks.pop(track_id, None)

            for detection_index, detection in enumerate(detections):
                if detection_index in matched_detections:
                    continue
                detection_to_track[detection_index] = self._new_track(
                    detection,
                    next_frame,
                )

            current = [
                self._view(detection_to_track[index])
                for index in range(len(detections))
                if detection_to_track[index].hits >= self.min_hits
            ]
            if include_unmatched:
                current_ids = {item.track_id for item in current}
                current.extend(
                    self._view(track)
                    for track_id, track in sorted(self._tracks.items())
                    if track_id not in current_ids
                    and track.missed > 0
                    and track.hits >= self.min_hits
                )
            return current

    def snapshot(self, *, include_tentative: bool = False) -> list[TrackedDetection]:
        with self._lock:
            return [
                self._view(track)
                for _track_id, track in sorted(self._tracks.items())
                if include_tentative or track.hits >= self.min_hits
            ]
