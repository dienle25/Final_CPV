from __future__ import annotations

import unittest

from src.classroom_demo.detectors import Detection, FaceDetection
from src.classroom_demo.helmet_status import (
    HelmetStatus,
    associate_people_faces_heads,
    classify_helmet_status,
)


def _face(box: tuple[float, float, float, float]) -> FaceDetection:
    x1, y1, x2, y2 = box
    return FaceDetection(
        bbox=box,
        score=0.95,
        landmarks=(
            (x1 + 20, y1 + 30),
            (x2 - 20, y1 + 30),
            ((x1 + x2) / 2, (y1 + y2) / 2),
            (x1 + 25, y2 - 20),
            (x2 - 25, y2 - 20),
        ),
    )


class HelmetStatusTests(unittest.TestCase):
    def test_four_status_heuristic(self) -> None:
        face = _face((60, 60, 140, 180))
        correct = Detection((50, 35, 150, 125), 0.9, 0, "helmet")
        incorrect = Detection((120, 130, 195, 210), 0.9, 0, "helmet")
        absent = Detection((60, 60, 140, 180), 0.9, 1, "no_helmet")

        self.assertEqual(classify_helmet_status(face, correct)[0], HelmetStatus.WORN_CORRECTLY)
        self.assertEqual(classify_helmet_status(face, incorrect)[0], HelmetStatus.WORN_INCORRECTLY)
        self.assertEqual(classify_helmet_status(face, absent)[0], HelmetStatus.NO_HELMET)
        self.assertEqual(classify_helmet_status(face, None)[0], HelmetStatus.UNKNOWN)
        self.assertEqual(classify_helmet_status(None, correct)[0], HelmetStatus.UNKNOWN)

    def test_person_face_head_one_to_one_association(self) -> None:
        people = [
            Detection((0, 0, 180, 400), 0.9, 0, "person"),
            Detection((200, 0, 380, 400), 0.9, 0, "person"),
        ]
        faces = [_face((50, 50, 130, 170)), _face((250, 50, 330, 170))]
        heads = [
            Detection((40, 25, 140, 120), 0.9, 0, "helmet"),
            Detection((250, 50, 330, 170), 0.9, 1, "no_helmet"),
        ]
        results = associate_people_faces_heads(people, faces, heads)
        self.assertEqual([result.face_index for result in results], [0, 1])
        self.assertEqual([result.head_index for result in results], [0, 1])
        self.assertEqual(
            [result.status for result in results],
            [HelmetStatus.WORN_CORRECTLY, HelmetStatus.NO_HELMET],
        )

    def test_conflicting_helmet_and_no_helmet_evidence_is_unknown(self) -> None:
        people = [Detection((0, 0, 200, 400), 0.95, 0, "person")]
        faces = [_face((60, 60, 140, 180))]
        helmet = Detection((50, 35, 150, 125), 0.92, 0, "helmet")
        no_helmet = Detection((55, 40, 145, 130), 0.88, 1, "no_helmet")

        for heads in ([helmet, no_helmet], [no_helmet, helmet]):
            with self.subTest(order=[head.label for head in heads]):
                result = associate_people_faces_heads(people, faces, heads)[0]
                self.assertEqual(result.status, HelmetStatus.UNKNOWN)
                self.assertEqual(result.reason, "conflicting_helmet_evidence")

    def test_borderline_helmet_alignment_remains_unknown(self) -> None:
        face = _face((60, 60, 140, 180))
        borderline = Detection((105, 35, 185, 125), 0.90, 0, "helmet")

        status, reason = classify_helmet_status(face, borderline)

        self.assertEqual(status, HelmetStatus.UNKNOWN)
        self.assertEqual(reason, "helmet_alignment_borderline")

    def test_low_confidence_misaligned_helmet_remains_unknown(self) -> None:
        face = _face((60, 60, 140, 180))
        low_confidence = Detection((120, 130, 195, 210), 0.40, 0, "helmet")

        status, reason = classify_helmet_status(face, low_confidence)

        self.assertEqual(status, HelmetStatus.UNKNOWN)
        self.assertEqual(reason, "helmet_misaligned_low_confidence")

    def test_no_helmet_without_face_geometry_remains_unknown(self) -> None:
        no_helmet = Detection((60, 60, 140, 180), 0.90, 1, "no_helmet")

        status, reason = classify_helmet_status(None, no_helmet)

        self.assertEqual(status, HelmetStatus.UNKNOWN)
        self.assertEqual(reason, "no_helmet_without_face_geometry")

    def test_close_up_face_can_still_be_associated_to_person(self) -> None:
        person = Detection((10, 110, 480, 834), 0.90, 0, "person")
        face = _face((67, 194, 468, 732))

        result = associate_people_faces_heads([person], [face], [])[0]

        self.assertIs(result.face, face)
        self.assertEqual(result.face_index, 0)


if __name__ == "__main__":
    unittest.main()
