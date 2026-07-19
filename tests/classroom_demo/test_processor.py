from __future__ import annotations

import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from src.classroom_demo.detectors import Detection, FaceDetection
from src.classroom_demo.processor import (
    ClassroomFrameProcessor,
    ClassroomProcessorConfig,
    coerce_processor_config,
    create_default_processor,
)


def _face() -> FaceDetection:
    return FaceDetection(
        bbox=(20, 20, 80, 90),
        score=0.95,
        landmarks=((35, 40), (65, 40), (50, 55), (38, 72), (62, 72)),
    )


class _PersonDetector:
    def detect(self, _frame: np.ndarray) -> list[Detection]:
        return [Detection((0, 0, 100, 200), 0.9, 0, "person")]


class _HelmetDetector:
    def detect(self, _frame: np.ndarray) -> list[Detection]:
        return [Detection((20, 20, 80, 90), 0.9, 1, "no_helmet")]


class _FaceDetector:
    def detect(self, _frame: np.ndarray) -> list[FaceDetection]:
        return [_face()]


class _FaceEncoder:
    def extract(self, frame: np.ndarray, _face_detection: FaceDetection) -> np.ndarray:
        if int(frame[0, 0, 0]) > 200:
            return np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
        return np.asarray([1.0, 0.0, 0.0], dtype=np.float32)


class _TwoPersonDetector:
    def detect(self, _frame: np.ndarray) -> list[Detection]:
        return [
            Detection((0, 0, 100, 200), 0.9, 0, "person"),
            Detection((120, 0, 220, 200), 0.9, 0, "person"),
        ]


class _TwoHelmetDetector:
    def detect(self, _frame: np.ndarray) -> list[Detection]:
        return [
            Detection((20, 20, 80, 90), 0.9, 1, "no_helmet"),
            Detection((140, 20, 200, 90), 0.9, 1, "no_helmet"),
        ]


class _TwoFaceDetector:
    def detect(self, _frame: np.ndarray) -> list[FaceDetection]:
        first = _face()
        second = FaceDetection(
            bbox=(140, 20, 200, 90),
            score=0.95,
            landmarks=((155, 40), (185, 40), (170, 55), (158, 72), (182, 72)),
        )
        return [first, second]


def _png_bytes(color: tuple[int, int, int] = (220, 180, 140)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (100, 120), color).save(buffer, format="PNG")
    return buffer.getvalue()


class ProcessorTests(unittest.TestCase):
    def test_full_ui_config_is_filtered_and_aliased(self) -> None:
        full_ui_config = {
            "source_kind": "ip_webcam",
            "source": "http://phone/video",
            "source_name": "classroom",
            "ip_webcam_url": "http://phone/video",
            "output_dir": "outputs/classroom",
            "save_evidence": True,
            "model_path": "custom/helmet.onnx",
            "face_match_threshold": 0.48,
            "detection_confidence": 0.31,
            "device": "dml",
        }
        config = coerce_processor_config(full_ui_config)
        self.assertEqual(config.helmet_model, "custom/helmet.onnx")
        self.assertEqual(config.cosine_threshold, 0.48)
        self.assertEqual(config.person_confidence, 0.31)
        self.assertEqual(config.helmet_confidence, 0.31)
        self.assertEqual(
            config.providers,
            ("DmlExecutionProvider", "CPUExecutionProvider"),
        )

        with patch("src.classroom_demo.processor.ClassroomFrameProcessor") as constructor:
            create_default_processor(full_ui_config)
            passed = constructor.call_args.args[0]
            self.assertIsInstance(passed, ClassroomProcessorConfig)

    def test_register_bytes_before_and_after_processing_and_combined_events(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "data"
            data.mkdir()
            roster = data / "students.csv"
            roster.write_text(
                "student_id,full_name,active\n"
                "S1,Student One,1\n"
                "S2,Student Two,1\n"
                "S3,Student Three,1\n"
                "S4,Student Four,1\n"
                "S5,Student Five,1\n",
                encoding="utf-8-sig",
            )
            config = ClassroomProcessorConfig(
                project_root=root,
                roster_csv="data/students.csv",
                reference_root="data/students",
                identity_min_votes=2,
                status_min_votes=1,
                recognition_interval_frames=1,
                recognition_minimum_face_width=50,
                recognition_minimum_face_height=60,
            )
            processor = ClassroomFrameProcessor(
                config,
                person_detector=_PersonDetector(),
                helmet_detector=_HelmetDetector(),
                face_detector=_FaceDetector(),
                face_encoder=_FaceEncoder(),
            )
            self.assertEqual(len(processor.list_students()), 5)

            updated = processor.register_student(
                "S1",
                "Sinh viên Một",
                image_bytes=_png_bytes(),
            )
            self.assertEqual(updated["full_name"], "Sinh viên Một")
            self.assertEqual(updated["image_count"], 1)

            frame = np.zeros((240, 160, 3), dtype=np.uint8)
            first = processor.process_frame(frame)
            self.assertEqual(first["events"][0]["student_id"], None)
            self.assertEqual(first["events"][0]["helmet_status"], "Không mũ")

            second = processor.process_frame(frame)
            self.assertEqual(second["events"][0]["student_id"], "S1")
            self.assertEqual(second["events"][0]["helmet_status"], "Không mũ")
            self.assertIn("face_similarity", second["events"][0])
            self.assertIn("helmet_confidence", second["events"][0])

            added = processor.register_student(
                "S6",
                "Sinh viên Sáu",
                images=[_png_bytes((0, 0, 255))],
            )
            self.assertEqual(added["student_id"], "S6")
            self.assertEqual(len(processor.list_students()), 6)
            self.assertIn("S6", roster.read_text(encoding="utf-8-sig"))

    def test_sticky_identity_is_cleared_after_two_consecutive_rejections(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "data"
            data.mkdir()
            (data / "students.csv").write_text(
                "student_id,full_name,active\nS1,Student One,1\n",
                encoding="utf-8-sig",
            )
            config = ClassroomProcessorConfig(
                project_root=root,
                roster_csv="data/students.csv",
                reference_root="data/students",
                identity_min_votes=2,
                identity_min_ratio=1.0,
                identity_reject_reset_count=2,
                recognition_interval_frames=1,
                recognition_minimum_face_width=50,
                recognition_minimum_face_height=60,
                status_min_votes=1,
            )
            processor = ClassroomFrameProcessor(
                config,
                person_detector=_PersonDetector(),
                helmet_detector=_HelmetDetector(),
                face_detector=_FaceDetector(),
                face_encoder=_FaceEncoder(),
            )
            processor.register_student("S1", "Student One", image_bytes=_png_bytes())

            known_frame = np.zeros((240, 160, 3), dtype=np.uint8)
            self.assertIsNone(processor.process_frame(known_frame)["people"][0]["student_id"])
            self.assertEqual(
                processor.process_frame(known_frame)["people"][0]["student_id"],
                "S1",
            )

            rejected_frame = np.zeros((240, 160, 3), dtype=np.uint8)
            rejected_frame[:, :, 0] = 255
            first_rejection = processor.process_frame(rejected_frame)["people"][0]
            self.assertEqual(first_rejection["identity_current_reason"], "below_threshold")
            self.assertEqual(first_rejection["student_id"], "S1")

            second_rejection = processor.process_frame(rejected_frame)["people"][0]
            self.assertEqual(second_rejection["identity_current_reason"], "below_threshold")
            self.assertIsNone(second_rejection["student_id"])
            self.assertEqual(second_rejection["display_identity"], "Chưa xác định")

    def test_duplicate_registration_is_blocked_before_files_are_committed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "data"
            data.mkdir()
            roster = data / "students.csv"
            roster.write_text(
                "student_id,full_name,active\n"
                "S1,Student One,1\n"
                "S2,Student Two,1\n",
                encoding="utf-8-sig",
            )
            processor = ClassroomFrameProcessor(
                ClassroomProcessorConfig(
                    project_root=root,
                    roster_csv="data/students.csv",
                    reference_root="data/students",
                ),
                person_detector=_PersonDetector(),
                helmet_detector=_HelmetDetector(),
                face_detector=_FaceDetector(),
                face_encoder=_FaceEncoder(),
            )
            payload = _png_bytes()
            processor.register_student("S1", "Student One", image_bytes=payload)

            with self.assertRaisesRegex(
                ValueError,
                "too similar to registered student S1",
            ):
                processor.register_student("S2", "Student Two", image_bytes=payload)

            self.assertFalse((data / "students" / "S2").exists())
            self.assertEqual(set(processor.gallery.centroids), {"S1"})

    def test_one_student_id_cannot_be_assigned_to_two_live_tracks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "data"
            data.mkdir()
            (data / "students.csv").write_text(
                "student_id,full_name,active\nS1,Student One,1\n",
                encoding="utf-8-sig",
            )
            config = ClassroomProcessorConfig(
                project_root=root,
                roster_csv="data/students.csv",
                reference_root="data/students",
                identity_min_votes=1,
                identity_min_ratio=1.0,
                recognition_interval_frames=1,
                recognition_minimum_face_width=50,
                recognition_minimum_face_height=60,
                status_min_votes=1,
            )
            processor = ClassroomFrameProcessor(
                config,
                person_detector=_TwoPersonDetector(),
                helmet_detector=_TwoHelmetDetector(),
                face_detector=_FaceDetector(),
                face_encoder=_FaceEncoder(),
            )
            processor.register_student("S1", "Student One", image_bytes=_png_bytes())
            processor.face_detector = _TwoFaceDetector()

            result = processor.process_frame(np.zeros((240, 240, 3), dtype=np.uint8))

            self.assertEqual(len(result["people"]), 2)
            self.assertTrue(all(person["student_id"] is None for person in result["people"]))
            self.assertTrue(
                all(
                    person["identity_current_reason"] == "duplicate_identity_conflict"
                    for person in result["people"]
                )
            )
            self.assertEqual(result["stats"]["recognized_count"], 0)
            self.assertFalse(
                any(event.get("student_id") == "S1" for event in result["events"])
            )


if __name__ == "__main__":
    unittest.main()
