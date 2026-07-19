from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.classroom_demo.ui_helpers import (
    REQUIRED_DEMO_MODELS,
    connection_view,
    inspect_demo_readiness,
)


class ConnectionViewTests(unittest.TestCase):
    def test_reports_truthful_connection_states(self) -> None:
        now = datetime(2026, 7, 19, 8, 0, tzinfo=timezone.utc)
        cases = (
            ({"state": "stopped"}, False, "stopped"),
            ({"state": "failed", "last_error": "boom"}, True, "failed"),
            ({"state": "starting"}, True, "connecting"),
            ({"state": "reconnecting", "reconnect_count": 1}, True, "reconnecting"),
            (
                {
                    "state": "running",
                    "connected": True,
                    "frame_sequence": 4,
                    "last_frame_at": (now - timedelta(seconds=1)).isoformat(),
                },
                True,
                "live",
            ),
            (
                {
                    "state": "running",
                    "connected": True,
                    "frame_sequence": 4,
                    "last_frame_at": (now - timedelta(seconds=4)).isoformat(),
                },
                True,
                "stale",
            ),
        )

        for stats, running, expected in cases:
            with self.subTest(expected=expected):
                view = connection_view(
                    stats,
                    monitor_running=running,
                    stale_after_seconds=3.0,
                    now=now,
                )
                self.assertEqual(view["code"], expected)

    def test_failed_state_keeps_backend_error_for_operator(self) -> None:
        view = connection_view(
            {"state": "failed", "last_error": "camera refused connection"},
            monitor_running=True,
        )

        self.assertEqual(view["code"], "failed")
        self.assertEqual(view["message"], "camera refused connection")


class ReadinessTests(unittest.TestCase):
    def test_readiness_counts_only_active_students_and_flags_missing_references(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            for relative in REQUIRED_DEMO_MODELS:
                model_path = root / relative
                model_path.parent.mkdir(parents=True, exist_ok=True)
                model_path.write_bytes(b"0" * 100_000)

            roster = root / "data/students.csv"
            roster.parent.mkdir(parents=True, exist_ok=True)
            roster.write_text(
                "student_id,full_name,active\n"
                "READY01,Ready Student,1\n"
                "MISSING01,Missing Student,true\n"
                "INACTIVE01,Inactive Student,0\n",
                encoding="utf-8",
            )
            ready_folder = root / "data/students/READY01"
            ready_folder.mkdir(parents=True, exist_ok=True)
            for index in range(5):
                (ready_folder / f"reference_{index}.jpg").write_bytes(b"image")

            result = inspect_demo_readiness(root, recommended_images=5)

        self.assertFalse(result["ready"])
        self.assertEqual(result["model_count"], len(REQUIRED_DEMO_MODELS))
        self.assertEqual(result["student_count"], 2)
        self.assertEqual(result["ready_student_count"], 1)
        self.assertEqual(result["reference_counts"], {"READY01": 5, "MISSING01": 0})
        self.assertTrue(any("MISSING01" in message for message in result["errors"]))
        self.assertTrue(any("MISSING01" in message for message in result["warnings"]))
        self.assertFalse(any("INACTIVE01" in message for message in result["errors"]))


if __name__ == "__main__":
    unittest.main()
