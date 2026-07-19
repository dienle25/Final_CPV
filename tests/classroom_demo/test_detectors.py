from __future__ import annotations

import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from src.classroom_demo.detectors import (
    NanoDetPersonDetector,
    YoloHelmetOnnxDetector,
    YuNetFaceDetector,
    create_ort_session_options,
)


class _Input:
    def __init__(self, shape: list[int]) -> None:
        self.name = "images"
        self.shape = shape


class _Session:
    def __init__(self, outputs: list[np.ndarray], shape: list[int]) -> None:
        self.outputs = outputs
        self.input = _Input(shape)
        self._guard = threading.Lock()
        self.active = 0
        self.max_active = 0

    def get_inputs(self) -> list[_Input]:
        return [self.input]

    def run(self, _names: object, feed: object) -> list[np.ndarray]:
        self.last_feed = feed
        with self._guard:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        time.sleep(0.005)
        with self._guard:
            self.active -= 1
        return self.outputs


class _Options:
    def __init__(self) -> None:
        self.execution_mode = None
        self.enable_mem_pattern = True
        self.inter_op_num_threads = 4


class _FakeOrt:
    class ExecutionMode:
        ORT_SEQUENTIAL = "sequential"

    SessionOptions = _Options


class DetectorTests(unittest.TestCase):
    @staticmethod
    def _yolo_output() -> np.ndarray:
        output = np.zeros((1, 7, 8400), dtype=np.float32)
        output[0, :, 0] = [320, 320, 200, 100, 0.90, 0.10, 0.05]
        output[0, :, 1] = [322, 320, 200, 100, 0.80, 0.10, 0.05]
        output[0, :, 2] = [100, 250, 50, 50, 0.05, 0.85, 0.05]
        return output

    def test_directml_safe_options(self) -> None:
        options = create_ort_session_options(_FakeOrt)
        self.assertEqual(options.execution_mode, "sequential")
        self.assertFalse(options.enable_mem_pattern)
        self.assertEqual(options.inter_op_num_threads, 1)

    def test_yolo_decode_letterbox_and_class_aware_nms(self) -> None:
        session = _Session([self._yolo_output()], [1, 3, 640, 640])
        detector = YoloHelmetOnnxDetector(session=session)
        frame = np.zeros((320, 640, 3), dtype=np.uint8)
        detections = detector.detect(frame)

        self.assertEqual([detection.label for detection in detections], ["helmet", "no_helmet"])
        self.assertTrue(np.allclose(detections[0].bbox, (220, 110, 420, 210), atol=1))
        self.assertTrue(np.allclose(detections[1].bbox, (75, 65, 125, 115), atol=1))

    def test_yolo_session_run_is_serialized(self) -> None:
        session = _Session([self._yolo_output()], [1, 3, 640, 640])
        detector = YoloHelmetOnnxDetector(session=session)
        frame = np.zeros((160, 320, 3), dtype=np.uint8)
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(lambda _index: detector.detect(frame), range(8)))
        self.assertTrue(all(results))
        self.assertEqual(session.max_active, 1)

    def test_nanodet_unordered_six_outputs(self) -> None:
        counts = (2704, 676, 169)
        class_outputs = {
            count: np.zeros((1, count, 80), dtype=np.float32) for count in counts
        }
        box_outputs = {
            count: np.zeros((1, count, 32), dtype=np.float32) for count in counts
        }
        prior_index = 6 * 13 + 6
        class_outputs[169][0, prior_index, 0] = 0.90
        logits = np.full((4, 8), -8.0, dtype=np.float32)
        logits[:, 3] = 8.0
        box_outputs[169][0, prior_index] = logits.reshape(-1)
        outputs = [
            box_outputs[2704],
            class_outputs[169],
            box_outputs[676],
            class_outputs[2704],
            class_outputs[676],
            box_outputs[169],
        ]
        session = _Session(outputs, [1, 3, 416, 416])
        detector = NanoDetPersonDetector(session=session)
        detections = detector.detect(np.zeros((416, 416, 3), dtype=np.uint8))
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0].label, "person")
        self.assertTrue(np.allclose(detections[0].bbox, (112, 112, 304, 304), atol=2))

    def test_yunet_parsing_with_injected_detector(self) -> None:
        class FakeYuNet:
            def setInputSize(self, size: tuple[int, int]) -> None:
                self.size = size

            def detect(self, _frame: np.ndarray) -> tuple[int, np.ndarray]:
                row = np.asarray(
                    [[10, 20, 40, 50, 20, 35, 40, 35, 30, 45, 22, 58, 38, 58, 0.95]],
                    dtype=np.float32,
                )
                return 1, row

        detector = YuNetFaceDetector("unused.onnx", detector=FakeYuNet())
        faces = detector.detect(np.zeros((100, 120, 3), dtype=np.uint8))
        self.assertEqual(len(faces), 1)
        self.assertEqual(faces[0].bbox, (10.0, 20.0, 50.0, 70.0))
        self.assertEqual(faces[0].landmarks[2], (30.0, 45.0))


if __name__ == "__main__":
    unittest.main()
