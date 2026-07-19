from __future__ import annotations

import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from pathlib import Path

from src.classroom_demo.sources import (
    CaptureWorker,
    _default_capture_factory,
    normalize_ip_webcam_url,
    parse_source,
)


def wait_until(predicate, timeout: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return bool(predicate())


class FakeCapture:
    def __init__(self, frames=(), *, opened: bool = True) -> None:
        self.frames = list(frames)
        self.opened = opened
        self.released = threading.Event()

    def isOpened(self) -> bool:  # noqa: N802
        return self.opened and not self.released.is_set()

    def read(self):
        if self.released.is_set() or not self.frames:
            return False, None
        return True, self.frames.pop(0)

    def release(self) -> None:
        self.released.set()


class SourceNormalizationTests(unittest.TestCase):
    def test_default_opencv_capture_enables_orientation_metadata(self) -> None:
        configured: list[tuple[int, int]] = []
        opened_with: list[tuple[object, ...]] = []
        capture = SimpleNamespace(set=lambda prop, value: configured.append((prop, value)))
        def video_capture(*args):
            opened_with.append(args)
            return capture

        fake_cv2 = SimpleNamespace(
            CAP_ANY=0,
            CAP_PROP_ORIENTATION_AUTO=42,
            CAP_PROP_OPEN_TIMEOUT_MSEC=43,
            CAP_PROP_READ_TIMEOUT_MSEC=44,
            VideoCapture=video_capture,
            error=RuntimeError,
        )
        with patch.dict("sys.modules", {"cv2": fake_cv2}):
            result = _default_capture_factory("portrait-phone.mp4")
        self.assertIs(result, capture)
        self.assertEqual(
            opened_with,
            [("portrait-phone.mp4", 0, [43, 5000, 44, 5000])],
        )
        self.assertEqual(configured, [(42, 1)])

    def test_ip_webcam_url_is_normalized_to_video_endpoint(self) -> None:
        self.assertEqual(
            normalize_ip_webcam_url("192.168.1.20:8080"),
            "http://192.168.1.20:8080/video",
        )
        self.assertEqual(
            normalize_ip_webcam_url(" https://user:pass@example.test:9000/ignored/?quality=80 "),
            "https://user:pass@example.test:9000/video?quality=80",
        )

    def test_source_types(self) -> None:
        self.assertEqual(parse_source(0).kind, "webcam")
        self.assertEqual(parse_source("2").capture_value, 2)
        self.assertEqual(parse_source(Path("data/demo.mp4")).kind, "file")

        phone = parse_source("http://10.0.0.5:8080")
        self.assertEqual(phone.kind, "ip_webcam")
        self.assertEqual(phone.capture_value, "http://10.0.0.5:8080/video")

        generic = parse_source("http://camera.test/mjpeg", ip_webcam=False)
        self.assertEqual(generic.kind, "stream")
        self.assertEqual(generic.capture_value, "http://camera.test/mjpeg")

    def test_network_source_label_redacts_secrets_but_capture_keeps_them(self) -> None:
        source_url = (
            "https://camera-user:camera-password@camera.test:8443/"
            "mjpeg?token=private-token"
        )

        source = parse_source(source_url, ip_webcam=False)

        self.assertEqual(source.capture_value, source_url)
        self.assertEqual(source.label, "https://camera.test:8443/mjpeg")
        self.assertNotIn("camera-user", source.label)
        self.assertNotIn("camera-password", source.label)
        self.assertNotIn("private-token", source.label)

    def test_invalid_sources_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            parse_source(-1)
        with self.assertRaises(ValueError):
            parse_source("ftp://camera.test/live")
        with self.assertRaises(ValueError):
            normalize_ip_webcam_url("")


class CaptureWorkerTests(unittest.TestCase):
    def test_live_source_reconnects_and_keeps_latest_frame(self) -> None:
        captures: list[FakeCapture] = []

        def factory(_source):
            if not captures:
                capture = FakeCapture(opened=False)
            elif len(captures) == 1:
                capture = FakeCapture(["older", "newest"])
            else:
                capture = FakeCapture(opened=False)
            captures.append(capture)
            return capture

        worker = CaptureWorker(
            "http://10.0.0.5:8080",
            capture_factory=factory,
            initial_backoff=0.005,
            max_backoff=0.02,
        ).start()
        try:
            self.assertTrue(wait_until(lambda: worker.sequence >= 2))
            packet = worker.get_latest(timeout=0.2)
            self.assertIsNotNone(packet)
            self.assertEqual(packet.frame, "newest")
            self.assertGreaterEqual(worker.reconnect_count, 1)
        finally:
            worker.stop()
        self.assertFalse(worker.running)
        self.assertTrue(all(capture.released.is_set() for capture in captures))

    def test_local_file_stops_at_eof_instead_of_looping(self) -> None:
        calls = 0

        def factory(_source):
            nonlocal calls
            calls += 1
            return FakeCapture(["only-frame"])

        worker = CaptureWorker(
            Path("demo.mp4"),
            capture_factory=factory,
            initial_backoff=0.001,
        ).start()
        self.assertTrue(wait_until(lambda: not worker.running))
        self.assertEqual(worker.sequence, 1)
        self.assertEqual(calls, 1)
        self.assertEqual(worker.latest.frame, "only-frame")
        self.assertIsNone(worker.last_error)

    def test_local_file_applies_backpressure_without_dropping_frames(self) -> None:
        worker = CaptureWorker(
            Path("demo.mp4"),
            capture_factory=lambda _source: FakeCapture(["one", "two", "three"]),
            initial_backoff=0.001,
        ).start()
        received: list[str] = []
        deadline = time.monotonic() + 2.0
        try:
            while len(received) < 3 and time.monotonic() < deadline:
                packet = worker.get_latest(timeout=0.2)
                if packet is not None:
                    received.append(packet.frame)
            self.assertEqual(received, ["one", "two", "three"])
            self.assertTrue(wait_until(lambda: not worker.running))
        finally:
            worker.stop()


if __name__ == "__main__":
    unittest.main()
