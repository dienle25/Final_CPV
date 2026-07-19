from __future__ import annotations

import importlib.util
import tempfile
import threading
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.classroom_demo.storage import EventStore


class EventStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.store = EventStore(
            self.root / "db" / "events.db",
            snapshot_dir=self.root / "Ảnh vi phạm",
            export_dir=self.root / "exports",
            cooldown_seconds=5.0,
        )

    def tearDown(self) -> None:
        self.store.close()
        self.temporary.cleanup()

    def test_cooldown_unicode_snapshot_and_csv_bom(self) -> None:
        started = datetime(2026, 7, 19, 8, 0, tzinfo=timezone.utc)
        first = self.store.record_event(
            session_id="Phòng mới",
            name="Nguyễn Thị Bích Tuyền",
            status="Không mũ",
            confidence=0.91,
            timestamp=started,
            snapshot=b"\xff\xd8demo\xff\xd9",
            dedupe_key="track:7",
        )
        duplicate = self.store.record_event(
            session_id="Phòng mới",
            name="Nguyễn Thị Bích Tuyền",
            status="Không mũ",
            confidence=0.95,
            timestamp=started + timedelta(seconds=2),
            dedupe_key="track:7",
        )
        changed_status = self.store.record_event(
            session_id="Phòng mới",
            name="Nguyễn Thị Bích Tuyền",
            status="Đội sai",
            confidence=0.80,
            timestamp=started + timedelta(seconds=2),
            dedupe_key="track:7",
        )
        after_cooldown = self.store.record_event(
            session_id="Phòng mới",
            name="Nguyễn Thị Bích Tuyền",
            status="Không mũ",
            confidence=0.96,
            timestamp=started + timedelta(seconds=6),
            dedupe_key="track:7",
        )

        self.assertIsNotNone(first)
        self.assertIsNone(duplicate)
        self.assertIsNotNone(changed_status)
        self.assertIsNotNone(after_cooldown)
        self.assertEqual(self.store.count(), 3)
        snapshot = Path(first["snapshot_path"])
        self.assertTrue(snapshot.exists())
        self.assertIn("Nguyễn Thị Bích Tuyền", snapshot.name)

        csv_path = self.store.export_csv()
        payload = csv_path.read_bytes()
        self.assertTrue(payload.startswith(b"\xef\xbb\xbf"))
        self.assertIn("Nguyễn Thị Bích Tuyền", payload.decode("utf-8-sig"))

    def test_concurrent_duplicate_writes_create_one_row(self) -> None:
        timestamp = datetime(2026, 7, 19, 9, 0, tzinfo=timezone.utc)
        barrier = threading.Barrier(8)
        inserted: list[dict] = []
        result_lock = threading.Lock()

        def write_event() -> None:
            barrier.wait()
            result = self.store.record_event(
                session_id="demo",
                student_id="CE182206",
                name="Nguyễn Thị Bích Tuyền",
                status="Đội đúng",
                confidence=0.9,
                timestamp=timestamp,
            )
            if result is not None:
                with result_lock:
                    inserted.append(result)

        threads = [threading.Thread(target=write_event) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=2.0)

        self.assertEqual(len(inserted), 1)
        self.assertEqual(self.store.count(), 1)

    @unittest.skipUnless(importlib.util.find_spec("openpyxl"), "openpyxl is optional")
    def test_xlsx_export_preserves_unicode(self) -> None:
        from openpyxl import load_workbook

        self.store.record_event(
            session_id="demo",
            student_id="CE190579",
            name="Lê Thanh Điền",
            status="Đội sai",
            confidence=0.75,
        )
        path = self.store.export_xlsx()
        workbook = load_workbook(path, read_only=True)
        sheet = workbook["Lịch sử"]
        values = list(sheet.values)
        workbook.close()
        self.assertIn("Lê Thanh Điền", values[1])


if __name__ == "__main__":
    unittest.main()

