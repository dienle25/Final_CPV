from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.classroom_demo.detectors import FaceDetection
from src.classroom_demo.face_recognition import (
    FaceGallery,
    StudentRecord,
    load_roster_csv,
)


def _face() -> FaceDetection:
    return FaceDetection(
        bbox=(10, 10, 90, 100),
        score=0.95,
        landmarks=((30, 40), (65, 40), (48, 58), (32, 78), (64, 78)),
    )


class FaceRecognitionTests(unittest.TestCase):
    def test_roster_aliases_and_reference_discovery(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reference = root / "students" / "CE001" / "front.png"
            reference.parent.mkdir(parents=True)
            reference.write_bytes(b"not-decoded-in-this-test")
            roster = root / "students.csv"
            roster.write_text(
                "MSSV,Họ tên\nCE001,Nguyễn Văn A\n",
                encoding="utf-8-sig",
            )
            records = load_roster_csv(roster, reference_root=root / "students")
            self.assertEqual(records[0].student_id, "CE001")
            self.assertEqual(records[0].full_name, "Nguyễn Văn A")
            self.assertEqual(records[0].reference_images, (reference.resolve(),))

    def test_threshold_and_best_second_margin(self) -> None:
        records = [
            StudentRecord("A", "Student A"),
            StudentRecord("B", "Student B"),
        ]
        gallery = FaceGallery(
            records,
            {"A": [1, 0, 0], "B": [0.8, 0.6, 0]},
            cosine_threshold=0.42,
            cosine_margin=0.05,
        )
        accepted = gallery.match([1, 0, 0])
        self.assertTrue(accepted.accepted)
        self.assertEqual(accepted.student_id, "A")

        ambiguous = gallery.match([0.95, 0.31225, 0])
        self.assertFalse(ambiguous.accepted)
        self.assertEqual(ambiguous.reason, "ambiguous_margin")

        unknown = gallery.match([0, 0, 1])
        self.assertFalse(unknown.accepted)
        self.assertEqual(unknown.reason, "below_threshold")

    def test_builds_normalized_reference_centroids(self) -> None:
        records = [
            StudentRecord(
                "A",
                "Student A",
                (Path("a_front.png"), Path("a_side.png")),
            ),
            StudentRecord("B", "Student B", (Path("b_front.png"),)),
        ]

        class Detector:
            def detect(self, _image: np.ndarray) -> list[FaceDetection]:
                return [_face()]

        class Encoder:
            def extract(self, image: np.ndarray, _detected: FaceDetection) -> np.ndarray:
                return image.reshape(-1)[:3]

        vectors = {
            "a_front.png": np.asarray([[[1.0, 0.0, 0.0]]], dtype=np.float32),
            "a_side.png": np.asarray([[[0.8, 0.6, 0.0]]], dtype=np.float32),
            "b_front.png": np.asarray([[[0.0, 1.0, 0.0]]], dtype=np.float32),
        }
        gallery = FaceGallery.build(
            records,
            Detector(),
            Encoder(),
            image_loader=lambda path: vectors[path.name],
        )
        self.assertAlmostEqual(float(np.linalg.norm(gallery.centroids["A"])), 1.0, places=6)
        self.assertEqual(gallery.match([0, 1, 0]).student_id, "B")

    def test_inactive_roster_record_is_excluded_from_gallery(self) -> None:
        records = [
            StudentRecord(
                "ACTIVE",
                "Active Student",
                (Path("active.png"),),
                metadata={"active": "1"},
            ),
            StudentRecord(
                "INACTIVE",
                "Inactive Student",
                (Path("inactive.png"),),
                metadata={"active": "0"},
            ),
        ]

        class Detector:
            def detect(self, _image: np.ndarray) -> list[FaceDetection]:
                return [_face()]

        class Encoder:
            def extract(self, image: np.ndarray, _detected: FaceDetection) -> np.ndarray:
                return image.reshape(-1)[:3]

        vectors = {
            "active.png": np.asarray([[[1.0, 0.0, 0.0]]], dtype=np.float32),
            "inactive.png": np.asarray([[[0.0, 1.0, 0.0]]], dtype=np.float32),
        }
        gallery = FaceGallery.build(
            records,
            Detector(),
            Encoder(),
            image_loader=lambda path: vectors[path.name],
        )

        self.assertEqual(set(gallery.centroids), {"ACTIVE"})
        self.assertNotIn("INACTIVE", gallery.reference_counts)
        self.assertEqual(gallery.match([1, 0, 0]).student_id, "ACTIVE")

    def test_sparse_identity_requires_cosine_similarity_of_at_least_point_55(self) -> None:
        records = [
            StudentRecord("A", "Student A"),
            StudentRecord("B", "Student B"),
        ]
        query = np.asarray(
            [0.54, np.sqrt(1.0 - 0.54**2), 0.0],
            dtype=np.float32,
        )

        sparse_gallery = FaceGallery(
            records,
            {"A": [1, 0, 0], "B": [0, 0, 1]},
            cosine_threshold=0.50,
            cosine_margin=0.10,
            reference_counts={"A": 1, "B": 5},
        )
        sparse_match = sparse_gallery.match(query)
        self.assertFalse(sparse_match.accepted)
        self.assertEqual(sparse_match.reason, "below_threshold")
        self.assertAlmostEqual(sparse_match.similarity, 0.54, places=5)

        dense_gallery = FaceGallery(
            records,
            {"A": [1, 0, 0], "B": [0, 0, 1]},
            cosine_threshold=0.50,
            cosine_margin=0.10,
            reference_counts={"A": 5, "B": 5},
        )
        dense_match = dense_gallery.match(query)
        self.assertTrue(dense_match.accepted)
        self.assertEqual(dense_match.student_id, "A")


if __name__ == "__main__":
    unittest.main()
