from __future__ import annotations

import unittest

from src.classroom_demo.detectors import Detection
from src.classroom_demo.tracking import IoUTracker


def _person(box: tuple[float, float, float, float]) -> Detection:
    return Detection(box, 0.9, 0, "person")


class TrackingTests(unittest.TestCase):
    def test_ids_survive_motion_input_order_and_short_dropout(self) -> None:
        tracker = IoUTracker(iou_threshold=0.2, max_missed=1)
        first = tracker.update([_person((0, 0, 100, 200)), _person((200, 0, 300, 200))])
        self.assertEqual([item.track_id for item in first], [1, 2])

        second = tracker.update(
            [_person((205, 0, 305, 200)), _person((5, 0, 105, 200))]
        )
        self.assertEqual([item.track_id for item in second], [2, 1])

        tracker.update([])
        returned = tracker.update([_person((10, 0, 110, 200))])
        self.assertEqual(returned[0].track_id, 1)

    def test_stale_track_is_replaced(self) -> None:
        tracker = IoUTracker(iou_threshold=0.2, max_missed=1)
        self.assertEqual(tracker.update([_person((0, 0, 100, 200))])[0].track_id, 1)
        tracker.update([])
        tracker.update([])
        self.assertEqual(tracker.update([_person((0, 0, 100, 200))])[0].track_id, 2)

    def test_explicit_frame_jump_expires_track(self) -> None:
        tracker = IoUTracker(iou_threshold=0.2, max_missed=1)
        self.assertEqual(
            tracker.update([_person((0, 0, 100, 200))], frame_index=0)[0].track_id,
            1,
        )
        returned = tracker.update(
            [_person((0, 0, 100, 200))],
            frame_index=3,
        )
        self.assertEqual(returned[0].track_id, 2)


if __name__ == "__main__":
    unittest.main()
