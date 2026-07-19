from __future__ import annotations

import importlib.util
import tempfile
import time
import unittest
from pathlib import Path

from src.classroom_demo.pipeline import (
    ClassroomDemoPipeline,
    PipelineState,
    _status_style,
    annotate_people,
)
from src.classroom_demo.sources import FramePacket, parse_source
from src.classroom_demo.storage import EventStore


def wait_until(predicate, timeout: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return bool(predicate())


class FakeCaptureWorker:
    def __init__(self, frames) -> None:
        self.source = parse_source(Path("fake.mp4"))
        captured_at = time.time()
        self._packets = [
            FramePacket(frame, index, captured_at + index * 0.01, time.monotonic(), self.source)
            for index, frame in enumerate(frames, start=1)
        ]
        self.running = False
        self.connected = False
        self.last_error = None
        self.reconnect_count = 0

    def start(self):
        self.running = True
        self.connected = True
        return self

    def stop(self, timeout=5.0):
        self.running = False
        self.connected = False

    def get_latest(self, timeout=None):
        if self._packets:
            return self._packets.pop(0)
        self.running = False
        self.connected = False
        return None


class FakeProcessor:
    def __init__(self) -> None:
        self.registrations = []

    def register_student(self, student_id, full_name, images):
        self.registrations.append((student_id, full_name, images))
        return {"registered": student_id, "image_count": len(images)}

    def process_frame(self, frame, frame_index, session_id):
        person = {
            "student_id": "CE182206",
            "name": "Nguyễn Thị Bích Tuyền",
            "status": "no_helmet",
            "confidence": 0.92,
            "bbox": (1, 2, 20, 30),
            "track_id": 7,
        }
        event = {**person, "snapshot": b"\xff\xd8event\xff\xd9"}
        return {
            "frame": f"processed:{frame}",
            "people": [person],
            "events": [event],
            "stats": {"processor_session": session_id, "processor_index": frame_index},
        }


class PipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        self.store = EventStore(
            root / "events.db",
            snapshot_dir=root / "snapshots",
            export_dir=root / "exports",
            cooldown_seconds=30.0,
        )

    def tearDown(self) -> None:
        self.store.close()
        self.temporary.cleanup()

    def test_background_state_ui_adapter_and_cooldown(self) -> None:
        processor = FakeProcessor()
        pipeline = ClassroomDemoPipeline(
            processor=processor,
            capture_worker=FakeCaptureWorker(["one", "two"]),
            store=self.store,
            annotator=lambda frame, people: f"annotated:{frame}:{people[0]['student_id']}",
        )
        registration = pipeline.register_student(
            "CE182206", "Nguyễn Thị Bích Tuyền", [b"face-one", b"face-two"]
        )
        self.assertEqual(registration["registered"], "CE182206")

        pipeline.start({"session_id": "demo-room"})
        self.assertTrue(wait_until(lambda: pipeline.state == PipelineState.STOPPED))
        snapshot = pipeline.snapshot()

        self.assertEqual(snapshot["frame"], "annotated:processed:two:CE182206")
        self.assertEqual(snapshot["stats"]["frames_processed"], 2)
        self.assertEqual(snapshot["stats"]["event_count"], 1)
        self.assertEqual(snapshot["stats"]["processor_session"], "demo-room")
        self.assertEqual(snapshot["people"][0]["student_id"], "CE182206")
        self.assertEqual(len(snapshot["events"]), 1)
        self.assertEqual(len(pipeline.history({"session_id": "demo-room"})), 1)
        self.assertTrue(pipeline.export_csv().exists())
        pipeline.stop()

    def test_processor_error_sets_failed_state(self) -> None:
        def broken_processor(_frame):
            raise RuntimeError("inference failed")

        pipeline = ClassroomDemoPipeline(
            processor=broken_processor,
            capture_worker=FakeCaptureWorker(["frame"]),
            store=self.store,
            annotator=lambda frame, people: frame,
        ).start({"session_id": "broken"})
        self.assertTrue(wait_until(lambda: pipeline.state == PipelineState.FAILED))
        self.assertIn("inference failed", pipeline.snapshot()["stats"]["last_error"])
        pipeline.stop()

    def test_owned_runtime_is_recreated_when_source_changes(self) -> None:
        created_sources: list[str] = []
        created_processors: list[FakeProcessor] = []

        def capture_factory(spec):
            created_sources.append(str(spec.capture_value))
            return FakeCaptureWorker([str(spec.capture_value)])

        def processor_factory(_config):
            processor = FakeProcessor()
            created_processors.append(processor)
            return processor

        pipeline = ClassroomDemoPipeline(
            processor_factory=processor_factory,
            capture_worker_factory=capture_factory,
            store=self.store,
            annotator=lambda frame, _people: frame,
        )
        pipeline.start({"source": "first.mp4", "session_id": "first"})
        self.assertTrue(wait_until(lambda: pipeline.state == PipelineState.STOPPED))
        pipeline.start({"source": "second.mp4", "session_id": "second"})
        self.assertTrue(wait_until(lambda: pipeline.state == PipelineState.STOPPED))

        self.assertEqual(created_sources, ["first.mp4", "second.mp4"])
        self.assertEqual(len(created_processors), 2)
        self.assertIn("second.mp4", str(pipeline.snapshot()["frame"]))
        pipeline.close()

    def test_default_store_history_survives_backend_recreation(self) -> None:
        root = Path(self.temporary.name) / "project"
        first = ClassroomDemoPipeline(project_root=root)
        assert first._store is not None
        first._store.record_event(
            session_id="persistent",
            student_id="CE182206",
            name="Nguyễn Thị Bích Tuyền",
            status="Không mũ",
            confidence=0.9,
            dedupe_key="student:CE182206",
        )
        first.close()

        second = ClassroomDemoPipeline(project_root=root)
        history = second.history({"session_id": "persistent"})
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["student_id"], "CE182206")
        second.close()

    def test_vietnamese_status_colors_are_mapped(self) -> None:
        self.assertEqual(_status_style("Đội đúng"), ((30, 180, 75), "Đội đúng"))
        self.assertEqual(_status_style("Đội sai"), ((245, 155, 35), "Đội sai"))
        self.assertEqual(_status_style("Không mũ"), ((220, 50, 55), "Không mũ"))

    def test_automatic_session_ids_are_unique_for_fast_restarts(self) -> None:
        pipeline = ClassroomDemoPipeline(store=self.store)
        pipeline._configure({})
        first = pipeline._session_id
        pipeline._configure({})
        second = pipeline._session_id
        self.assertNotEqual(first, second)
        self.assertRegex(first, r"^classroom_\d{8}_\d{6}_\d{6}$")

    def test_evidence_toggle_only_saves_violation_snapshots(self) -> None:
        packet = FramePacket(
            frame=b"raw",
            sequence=1,
            captured_at=time.time(),
            monotonic_at=time.monotonic(),
            source=parse_source(Path("fake.mp4")),
        )
        pipeline = ClassroomDemoPipeline(store=self.store)
        pipeline._configure({"save_event_snapshots": True, "session_id": "evidence-on"})
        events = [
            {"student_id": "S1", "status": "Đội đúng", "dedupe_key": "correct"},
            {"student_id": "S2", "status": "Chưa rõ", "dedupe_key": "unknown"},
            {"student_id": "S3", "status": "Đội sai", "dedupe_key": "wrong"},
            {"student_id": "S4", "status": "Không mũ", "dedupe_key": "none"},
        ]
        stored = pipeline._persist_events(events, b"\xff\xd8frame\xff\xd9", packet)
        paths = {item["student_id"]: item["snapshot_path"] for item in stored}
        self.assertTrue(all(item["source"].endswith("fake.mp4") for item in stored))
        self.assertEqual(paths["S1"], "")
        self.assertEqual(paths["S2"], "")
        self.assertTrue(Path(paths["S3"]).is_file())
        self.assertTrue(Path(paths["S4"]).is_file())

        pipeline._configure({"save_evidence": False, "session_id": "evidence-off"})
        disabled = pipeline._persist_events(
            [{"student_id": "S5", "status": "Không mũ", "dedupe_key": "disabled"}],
            b"\xff\xd8frame\xff\xd9",
            packet,
        )
        self.assertEqual(disabled[0]["snapshot_path"], "")


@unittest.skipUnless(
    importlib.util.find_spec("numpy") and importlib.util.find_spec("PIL"),
    "Pillow/numpy are optional in the infrastructure-only test environment",
)
class UnicodeAnnotationTests(unittest.TestCase):
    def test_pillow_annotation_changes_frame(self) -> None:
        import numpy as np

        frame = np.zeros((100, 180, 3), dtype=np.uint8)
        annotated = annotate_people(
            frame,
            [
                {
                    "student_id": "CE190579",
                    "status": "Đội sai",
                    "bbox": (10, 25, 160, 90),
                    "face_bbox": (50, 30, 90, 65),
                }
            ],
        )
        self.assertEqual(annotated.shape, frame.shape)
        self.assertGreater(int(annotated.sum()), 0)


if __name__ == "__main__":
    unittest.main()
