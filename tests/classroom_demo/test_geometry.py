from __future__ import annotations

import unittest

from src.classroom_demo.geometry import (
    area,
    as_bbox,
    clip_box,
    containment,
    greedy_one_to_one,
    iou,
    non_max_suppression,
    upper_region,
)


class GeometryTests(unittest.TestCase):
    def test_box_metrics_and_clipping(self) -> None:
        self.assertEqual(as_bbox((10, 20, 0, 5)), (0.0, 5.0, 10.0, 20.0))
        self.assertEqual(area((0, 0, 10, 10)), 100.0)
        self.assertAlmostEqual(iou((0, 0, 10, 10), (5, 0, 15, 10)), 1 / 3)
        self.assertAlmostEqual(containment((2, 2, 4, 4), (0, 0, 10, 10)), 1.0)
        self.assertEqual(clip_box((-2, 3, 20, 12), 16, 10), (0.0, 3.0, 16.0, 10.0))
        self.assertEqual(upper_region((0, 10, 20, 110), 0.5), (0.0, 10.0, 20.0, 60.0))

    def test_deterministic_assignment_and_nms(self) -> None:
        assignments = greedy_one_to_one(
            [(0.9, 0, 0), (0.8, 0, 1), (0.85, 1, 0), (0.7, 1, 1)]
        )
        self.assertEqual(assignments, {0: 0, 1: 1})

        kept = non_max_suppression(
            [(0, 0, 10, 10), (1, 1, 11, 11), (30, 30, 40, 40)],
            [0.9, 0.8, 0.7],
            0.5,
        )
        self.assertEqual(kept, [0, 2])


if __name__ == "__main__":
    unittest.main()
