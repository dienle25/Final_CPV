"""Temporal confirmation and persistent violation logging.

A violation is committed only after the same tracked rider has a stable
``no_helmet`` status in several nearby frames. Evidence is saved as a complete
annotated frame, while SQLite and CSV provide simple, server-free persistence.
"""

from __future__ import annotations

import csv
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np


@dataclass
class TrackState:
    hits: int = 0
    last_frame: int = -1
    logged: bool = False
    best_confidence: float = 0.0
    best_frame: np.ndarray | None = None
    best_rider_bbox: tuple[int, int, int, int] | None = None
    best_head_bbox: tuple[int, int, int, int] | None = None
    best_plate_crop: np.ndarray | None = None
    best_plate_source: str = ""


class ViolationLogger:
    def __init__(
        self,
        db_path: str | Path = "outputs/db/violations.db",
        image_dir: str | Path = "outputs/violations",
        min_hits: int = 3,
        max_frame_gap: int = 2,
        source_name: str = "video",
    ) -> None:
        self.db_path = Path(db_path)
        self.image_dir = Path(image_dir)
        self.plate_dir = self.image_dir / "plates"
        self.min_hits = max(1, int(min_hits))
        self.max_frame_gap = max(1, int(max_frame_gap))
        self.source_name = source_name
        self.states: dict[int, TrackState] = {}
        self.lock = threading.Lock()

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.image_dir.mkdir(parents=True, exist_ok=True)
        self.plate_dir.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._create_schema()

    def _create_schema(self) -> None:
        with self.conn:
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS violations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    track_id INTEGER NOT NULL,
                    violation_type TEXT NOT NULL DEFAULT 'no_helmet',
                    detected_at_utc TEXT NOT NULL,
                    video_time_s REAL NOT NULL,
                    frame_index INTEGER NOT NULL,
                    confidence REAL NOT NULL,
                    plate_text TEXT NOT NULL,
                    plate_confidence REAL NOT NULL DEFAULT 0,
                    plate_source TEXT NOT NULL DEFAULT '',
                    image_path TEXT NOT NULL,
                    plate_image_path TEXT NOT NULL DEFAULT '',
                    source TEXT NOT NULL
                )
                """
            )
            columns = {
                row[1] for row in self.conn.execute("PRAGMA table_info(violations)").fetchall()
            }
            migrations = {
                "violation_type": "TEXT NOT NULL DEFAULT 'no_helmet'",
                "plate_image_path": "TEXT NOT NULL DEFAULT ''",
                "plate_source": "TEXT NOT NULL DEFAULT ''",
            }
            for column, definition in migrations.items():
                if column not in columns:
                    self.conn.execute(
                        f"ALTER TABLE violations ADD COLUMN {column} {definition}"
                    )
            self.conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_violations_time "
                "ON violations(detected_at_utc DESC)"
            )

    @staticmethod
    def _draw_full_frame_evidence(
        frame: np.ndarray,
        rider_bbox: tuple[int, int, int, int],
        head_bbox: tuple[int, int, int, int] | None,
        track_id: int,
        confidence: float,
    ) -> np.ndarray:
        """Return a complete annotated frame for human-review evidence."""
        evidence = frame.copy()
        height, width = evidence.shape[:2]

        def clip(box: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
            x1, y1, x2, y2 = box
            return (
                max(0, min(x1, width - 1)),
                max(0, min(y1, height - 1)),
                max(1, min(x2, width)),
                max(1, min(y2, height)),
            )

        rx1, ry1, rx2, ry2 = clip(rider_bbox)
        cv2.rectangle(evidence, (rx1, ry1), (rx2, ry2), (0, 0, 255), 3)
        if head_bbox is not None:
            hx1, hy1, hx2, hy2 = clip(head_bbox)
            cv2.rectangle(evidence, (hx1, hy1), (hx2, hy2), (0, 165, 255), 3)

        label = f"NO HELMET | Rider ID {track_id} | conf {confidence:.2f}"
        cv2.putText(
            evidence,
            label,
            (rx1, max(30, ry1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        return evidence

    def reject(self, track_id: int) -> None:
        """Reset an unlogged candidate after a contradictory helmet vote."""
        if track_id < 0:
            return
        with self.lock:
            state = self.states.get(track_id)
            if state is None or state.logged:
                return
            state.hits = 0
            state.last_frame = -1
            state.best_confidence = 0.0
            state.best_frame = None
            state.best_rider_bbox = None
            state.best_head_bbox = None
            state.best_plate_crop = None
            state.best_plate_source = ""

    def observe(
        self,
        *,
        track_id: int,
        frame_index: int,
        video_time_s: float,
        confidence: float,
        rider_bbox: tuple[int, int, int, int],
        head_bbox: tuple[int, int, int, int] | None,
        frame: np.ndarray,
        plate_crop: np.ndarray | None = None,
        plate_source: str = "",
        ocr_reader: Callable[[np.ndarray | None], Any] | None = None,
    ) -> dict[str, Any] | None:
        """Update temporal state and commit once a rider is confirmed."""
        if track_id < 0:
            return None

        with self.lock:
            state = self.states.setdefault(track_id, TrackState())
            if state.logged:
                return None

            consecutive = (
                state.last_frame >= 0
                and frame_index - state.last_frame <= self.max_frame_gap
            )
            state.hits = state.hits + 1 if consecutive else 1
            state.last_frame = frame_index

            if confidence >= state.best_confidence:
                state.best_confidence = float(confidence)
                state.best_frame = frame.copy()
                state.best_rider_bbox = rider_bbox
                state.best_head_bbox = head_bbox
                if plate_crop is not None and plate_crop.size > 0:
                    state.best_plate_crop = plate_crop.copy()
                    state.best_plate_source = plate_source

            if state.hits < self.min_hits:
                return None

            evidence_frame = state.best_frame if state.best_frame is not None else frame
            evidence_rider_bbox = (
                state.best_rider_bbox if state.best_rider_bbox is not None else rider_bbox
            )
            evidence = self._draw_full_frame_evidence(
                evidence_frame,
                rider_bbox=evidence_rider_bbox,
                head_bbox=state.best_head_bbox,
                track_id=track_id,
                confidence=state.best_confidence,
            )

            ocr_text, ocr_confidence = "UNREAD", 0.0
            if ocr_reader is not None:
                result = ocr_reader(state.best_plate_crop)
                ocr_text = str(getattr(result, "text", "UNREAD"))
                ocr_confidence = float(getattr(result, "confidence", 0.0))

            detected_at = datetime.now(timezone.utc)
            stamp = detected_at.strftime("%Y%m%dT%H%M%S_%fZ")
            image_path = self.image_dir / f"rider_{track_id}_{stamp}.jpg"
            plate_image_path = ""

            if state.best_plate_crop is not None and state.best_plate_crop.size > 0:
                plate_path = self.plate_dir / f"rider_{track_id}_{stamp}_candidate.jpg"
                if cv2.imwrite(str(plate_path), state.best_plate_crop):
                    plate_image_path = str(plate_path)

            if not cv2.imwrite(str(image_path), evidence):
                raise IOError(f"Could not save evidence image: {image_path}")

            with self.conn:
                cursor = self.conn.execute(
                    """
                    INSERT INTO violations (
                        track_id, violation_type, detected_at_utc, video_time_s,
                        frame_index, confidence, plate_text, plate_confidence,
                        plate_source, image_path, plate_image_path, source
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        track_id,
                        "no_helmet",
                        detected_at.isoformat(),
                        float(video_time_s),
                        int(frame_index),
                        float(state.best_confidence),
                        ocr_text,
                        ocr_confidence,
                        state.best_plate_source,
                        str(image_path),
                        plate_image_path,
                        self.source_name,
                    ),
                )
                violation_id = int(cursor.lastrowid)

            state.logged = True
            state.best_frame = None
            state.best_plate_crop = None
            return {
                "id": violation_id,
                "track_id": track_id,
                "violation_type": "no_helmet",
                "detected_at_utc": detected_at.isoformat(),
                "video_time_s": round(float(video_time_s), 3),
                "frame_index": int(frame_index),
                "confidence": round(float(state.best_confidence), 4),
                "plate_text": ocr_text,
                "plate_confidence": round(ocr_confidence, 4),
                "plate_source": state.best_plate_source,
                "image_path": str(image_path),
                "plate_image_path": plate_image_path,
                "source": self.source_name,
            }

    def purge_stale(self, current_frame: int, stale_after_frames: int = 600) -> None:
        with self.lock:
            stale = [
                track_id
                for track_id, state in self.states.items()
                if current_frame - state.last_frame > stale_after_frames
            ]
            for track_id in stale:
                self.states.pop(track_id, None)

    def recent(self, limit: int = 100) -> list[dict[str, Any]]:
        cursor = self.conn.execute(
            "SELECT * FROM violations ORDER BY id DESC LIMIT ?", (int(limit),)
        )
        return [dict(row) for row in cursor.fetchall()]

    def count(self) -> int:
        return int(self.conn.execute("SELECT COUNT(*) FROM violations").fetchone()[0])

    def export_csv(self, path: str | Path = "outputs/violations.csv") -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = self.recent(limit=1_000_000)
        fieldnames = [
            "id",
            "track_id",
            "violation_type",
            "detected_at_utc",
            "video_time_s",
            "frame_index",
            "confidence",
            "plate_text",
            "plate_confidence",
            "plate_source",
            "image_path",
            "plate_image_path",
            "source",
        ]
        with path.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(reversed(rows))
        return path

    def close(self) -> None:
        self.conn.close()
