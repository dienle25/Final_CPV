"""Thread-safe persistence and exports for classroom demo events."""

from __future__ import annotations

import csv
import json
import math
import os
import re
import sqlite3
import tempfile
import threading
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


EXPORT_COLUMNS = (
    "id",
    "session_id",
    "student_id",
    "name",
    "status",
    "confidence",
    "timestamp",
    "snapshot_path",
    "dedupe_key",
    "metadata",
)


def _as_utc(value: datetime | str | None, clock: Callable[[], datetime]) -> datetime:
    if value is None:
        result = clock()
    elif isinstance(value, datetime):
        result = value
    elif isinstance(value, str):
        normalized = value.strip().replace("Z", "+00:00")
        if not normalized:
            raise ValueError("timestamp must not be empty")
        result = datetime.fromisoformat(normalized)
    else:
        raise TypeError("timestamp must be a datetime, ISO string, or None")

    if result.tzinfo is None:
        result = result.replace(tzinfo=timezone.utc)
    return result.astimezone(timezone.utc)


def _iso_utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z"
    )


def _safe_path_component(value: str, *, fallback: str) -> str:
    """Keep Unicode names while removing characters illegal on Windows."""

    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", str(value).strip())
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .")
    if not cleaned:
        cleaned = fallback
    # Leave room for timestamp/UUID and for parent directories on Windows.
    return cleaned[:80]


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{path.stem}-",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as stream:
            temporary_name = stream.name
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except Exception:
        if temporary_name:
            Path(temporary_name).unlink(missing_ok=True)
        raise


def save_jpeg_unicode(
    image: Any,
    destination: str | Path,
    *,
    quality: int = 90,
    encoder: Callable[[str, Any, Sequence[int]], tuple[bool, Any]] | None = None,
) -> Path:
    """Save JPEG bytes or an image array to a Unicode path safely.

    OpenCV's ``imwrite`` has historically been unreliable with some Unicode
    paths on Windows.  Encoding in memory and writing with :class:`Path` avoids
    that limitation.
    """

    path = Path(destination)
    if not 1 <= int(quality) <= 100:
        raise ValueError("JPEG quality must be between 1 and 100")

    if isinstance(image, (bytes, bytearray, memoryview)):
        payload = bytes(image)
    else:
        if image is None or getattr(image, "size", 1) == 0:
            raise ValueError("Cannot save an empty snapshot")
        if encoder is None:
            import cv2

            encoder = cv2.imencode
            params = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
        else:
            # OpenCV uses integer key/value pairs.  Custom test encoders may
            # simply ignore this conventional quality pair.
            params = [1, int(quality)]
        ok, encoded = encoder(".jpg", image, params)
        if not ok:
            raise RuntimeError("JPEG encoding failed")
        payload = encoded.tobytes() if hasattr(encoded, "tobytes") else bytes(encoded)

    if not payload:
        raise ValueError("Cannot save an empty snapshot")
    _atomic_write_bytes(path, payload)
    return path


class EventStore:
    """SQLite event store safe for one capture thread plus UI readers.

    Cooldown de-duplication runs inside ``BEGIN IMMEDIATE`` so two producer
    threads cannot both insert the same person/status event concurrently.
    """

    def __init__(
        self,
        db_path: str | Path = "outputs/classroom/db/events.db",
        *,
        snapshot_dir: str | Path = "outputs/classroom/snapshots",
        export_dir: str | Path = "outputs/classroom/exports",
        cooldown_seconds: float = 5.0,
        jpeg_quality: int = 90,
        clock: Callable[[], datetime] | None = None,
        jpeg_encoder: Callable[[str, Any, Sequence[int]], tuple[bool, Any]] | None = None,
    ) -> None:
        if cooldown_seconds < 0:
            raise ValueError("cooldown_seconds must not be negative")
        if not 1 <= int(jpeg_quality) <= 100:
            raise ValueError("jpeg_quality must be between 1 and 100")

        self.db_path = Path(db_path)
        self.snapshot_dir = Path(snapshot_dir)
        self.export_dir = Path(export_dir)
        self.cooldown_seconds = float(cooldown_seconds)
        self.jpeg_quality = int(jpeg_quality)
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._jpeg_encoder = jpeg_encoder
        self._lock = threading.RLock()
        self._closed = False

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        self.export_dir.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(
            self.db_path,
            timeout=30.0,
            check_same_thread=False,
            isolation_level=None,
        )
        self._connection.row_factory = sqlite3.Row
        self._initialize_schema()

    def _initialize_schema(self) -> None:
        with self._lock:
            self._connection.execute("PRAGMA journal_mode=WAL")
            self._connection.execute("PRAGMA synchronous=NORMAL")
            self._connection.execute("PRAGMA busy_timeout=30000")
            self._connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS classroom_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    student_id TEXT NOT NULL DEFAULT '',
                    name TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    timestamp TEXT NOT NULL,
                    snapshot_path TEXT,
                    dedupe_key TEXT NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_classroom_events_recent
                    ON classroom_events(session_id, timestamp DESC, id DESC);
                CREATE INDEX IF NOT EXISTS idx_classroom_events_dedupe
                    ON classroom_events(session_id, dedupe_key, status, timestamp DESC);
                CREATE INDEX IF NOT EXISTS idx_classroom_events_student
                    ON classroom_events(student_id, timestamp DESC);
                """
            )

    def record_event(
        self,
        *,
        session_id: str,
        status: str,
        confidence: float,
        student_id: str = "",
        name: str = "",
        timestamp: datetime | str | None = None,
        snapshot: Any | None = None,
        dedupe_key: str | int | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Insert one event, returning ``None`` when cooldown suppresses it."""

        session_id = str(session_id).strip()
        student_id = str(student_id or "").strip()
        name = str(name or "").strip()
        status = str(status).strip()
        if not session_id:
            raise ValueError("session_id must not be empty")
        if not status:
            raise ValueError("status must not be empty")

        numeric_confidence = float(confidence)
        if not math.isfinite(numeric_confidence) or not 0.0 <= numeric_confidence <= 1.0:
            raise ValueError("confidence must be a finite number between 0 and 1")

        event_time = _as_utc(timestamp, self._clock)
        event_timestamp = _iso_utc(event_time)
        metadata_dict = dict(metadata or {})
        identity = self._dedupe_identity(
            dedupe_key=dedupe_key,
            student_id=student_id,
            name=name,
            metadata=metadata_dict,
        )
        metadata_json = json.dumps(
            metadata_dict,
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )

        created_snapshot: Path | None = None
        with self._lock:
            self._ensure_open()
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                latest = self._connection.execute(
                    """
                    SELECT timestamp
                    FROM classroom_events
                    WHERE session_id = ? AND dedupe_key = ? AND status = ?
                    ORDER BY timestamp DESC, id DESC
                    LIMIT 1
                    """,
                    (session_id, identity, status),
                ).fetchone()
                if latest is not None:
                    latest_time = _as_utc(str(latest["timestamp"]), self._clock)
                    if event_time <= latest_time + timedelta(seconds=self.cooldown_seconds):
                        self._connection.execute("ROLLBACK")
                        return None

                snapshot_path = ""
                if snapshot is not None:
                    created_snapshot = self._snapshot_destination(
                        session_id=session_id,
                        student_id=student_id,
                        name=name,
                        status=status,
                        event_time=event_time,
                    )
                    save_jpeg_unicode(
                        snapshot,
                        created_snapshot,
                        quality=self.jpeg_quality,
                        encoder=self._jpeg_encoder,
                    )
                    snapshot_path = str(created_snapshot)

                created_at = _iso_utc(_as_utc(None, self._clock))
                cursor = self._connection.execute(
                    """
                    INSERT INTO classroom_events (
                        session_id, student_id, name, status, confidence,
                        timestamp, snapshot_path, dedupe_key, metadata_json, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        session_id,
                        student_id,
                        name,
                        status,
                        numeric_confidence,
                        event_timestamp,
                        snapshot_path,
                        identity,
                        metadata_json,
                        created_at,
                    ),
                )
                event_id = int(cursor.lastrowid)
                self._connection.execute("COMMIT")
            except Exception:
                try:
                    self._connection.execute("ROLLBACK")
                except sqlite3.Error:
                    pass
                if created_snapshot is not None:
                    created_snapshot.unlink(missing_ok=True)
                raise

        return {
            "id": event_id,
            "session_id": session_id,
            "student_id": student_id,
            "name": name,
            "status": status,
            "confidence": numeric_confidence,
            "timestamp": event_timestamp,
            "snapshot_path": snapshot_path,
            "snapshot": snapshot_path,
            "dedupe_key": identity,
            "source": str(metadata_dict.get("source", "") or ""),
            "metadata": metadata_dict,
        }

    def history(
        self,
        filters: Mapping[str, Any] | None = None,
        *,
        limit: int | None = 200,
        newest_first: bool = True,
    ) -> list[dict[str, Any]]:
        """Return events using simple, parameterized UI-friendly filters."""

        filters = dict(filters or {})
        clauses: list[str] = []
        values: list[Any] = []
        filter_columns = {
            "session_id": "session_id",
            "student_id": "student_id",
            "name": "name",
            "status": "status",
            "dedupe_key": "dedupe_key",
        }
        for key, column in filter_columns.items():
            value = filters.get(key)
            if value not in (None, ""):
                clauses.append(f"{column} = ?")
                values.append(str(value))

        since = filters.get("since", filters.get("from_timestamp"))
        until = filters.get("until", filters.get("to_timestamp"))
        if since not in (None, ""):
            clauses.append("timestamp >= ?")
            values.append(_iso_utc(_as_utc(since, self._clock)))
        if until not in (None, ""):
            clauses.append("timestamp <= ?")
            values.append(_iso_utc(_as_utc(until, self._clock)))

        query = "SELECT * FROM classroom_events"
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY timestamp " + ("DESC" if newest_first else "ASC")
        query += ", id " + ("DESC" if newest_first else "ASC")
        if limit is not None:
            if int(limit) < 1:
                return []
            query += " LIMIT ?"
            values.append(int(limit))

        with self._lock:
            self._ensure_open()
            rows = self._connection.execute(query, values).fetchall()
        return [self._row_to_event(row) for row in rows]

    def recent(
        self,
        limit: int = 100,
        *,
        session_id: str | None = None,
    ) -> list[dict[str, Any]]:
        filters = {"session_id": session_id} if session_id else None
        return self.history(filters, limit=limit, newest_first=True)

    def count(self, filters: Mapping[str, Any] | None = None) -> int:
        # History is deliberately not used here, so large demo databases remain
        # inexpensive to count.
        filters = dict(filters or {})
        clauses: list[str] = []
        values: list[str] = []
        for key in ("session_id", "student_id", "name", "status", "dedupe_key"):
            value = filters.get(key)
            if value not in (None, ""):
                clauses.append(f"{key} = ?")
                values.append(str(value))
        query = "SELECT COUNT(*) FROM classroom_events"
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        with self._lock:
            self._ensure_open()
            return int(self._connection.execute(query, values).fetchone()[0])

    def export_csv(
        self,
        path: str | Path | None = None,
        *,
        filters: Mapping[str, Any] | None = None,
    ) -> Path:
        """Export chronological rows as UTF-8 with BOM for Vietnamese Excel."""

        destination = Path(path) if path is not None else self.export_dir / "classroom_events.csv"
        rows = self.history(filters, limit=None, newest_first=False)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=EXPORT_COLUMNS)
            writer.writeheader()
            for row in rows:
                writer.writerow(self._export_row(row))
        return destination

    def export_xlsx(
        self,
        path: str | Path | None = None,
        *,
        filters: Mapping[str, Any] | None = None,
    ) -> Path:
        """Export chronological rows to an XLSX workbook via openpyxl."""

        try:
            from openpyxl import Workbook
            from openpyxl.styles import Font
        except ImportError as exc:
            raise RuntimeError(
                "XLSX export requires openpyxl. Install it before exporting."
            ) from exc

        destination = Path(path) if path is not None else self.export_dir / "classroom_events.xlsx"
        rows = self.history(filters, limit=None, newest_first=False)
        workbook = Workbook()
        sheet = workbook.active
        sheet.title = "Lịch sử"
        sheet.append(list(EXPORT_COLUMNS))
        for cell in sheet[1]:
            cell.font = Font(bold=True)
        for row in rows:
            exported = self._export_row(row)
            sheet.append([exported[column] for column in EXPORT_COLUMNS])
        sheet.freeze_panes = "A2"
        sheet.auto_filter.ref = sheet.dimensions
        widths = {
            "A": 10,
            "B": 24,
            "C": 16,
            "D": 28,
            "E": 22,
            "F": 12,
            "G": 26,
            "H": 48,
            "I": 24,
            "J": 42,
        }
        for column, width in widths.items():
            sheet.column_dimensions[column].width = width
        destination.parent.mkdir(parents=True, exist_ok=True)
        workbook.save(destination)
        return destination

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._connection.close()
            self._closed = True

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("EventStore is closed")

    @staticmethod
    def _dedupe_identity(
        *,
        dedupe_key: str | int | None,
        student_id: str,
        name: str,
        metadata: Mapping[str, Any],
    ) -> str:
        if dedupe_key not in (None, ""):
            return str(dedupe_key)
        if student_id:
            return f"student:{student_id}"
        for key in ("track_id", "person_id", "face_id"):
            if metadata.get(key) not in (None, ""):
                return f"{key}:{metadata[key]}"
        if name:
            return f"name:{name.casefold()}"
        return "unknown"

    def _snapshot_destination(
        self,
        *,
        session_id: str,
        student_id: str,
        name: str,
        status: str,
        event_time: datetime,
    ) -> Path:
        session = _safe_path_component(session_id, fallback="session")
        subject = _safe_path_component(student_id or name, fallback="unknown")
        safe_status = _safe_path_component(status, fallback="status")
        stamp = event_time.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        filename = f"{stamp}_{subject}_{safe_status}_{uuid.uuid4().hex[:8]}.jpg"
        return self.snapshot_dir / session / filename

    @staticmethod
    def _row_to_event(row: sqlite3.Row) -> dict[str, Any]:
        try:
            metadata = json.loads(row["metadata_json"] or "{}")
        except (TypeError, json.JSONDecodeError):
            metadata = {}
        snapshot_path = str(row["snapshot_path"] or "")
        return {
            "id": int(row["id"]),
            "session_id": str(row["session_id"]),
            "student_id": str(row["student_id"] or ""),
            "name": str(row["name"] or ""),
            "status": str(row["status"]),
            "confidence": float(row["confidence"]),
            "timestamp": str(row["timestamp"]),
            "snapshot_path": snapshot_path,
            "snapshot": snapshot_path,
            "dedupe_key": str(row["dedupe_key"]),
            "source": str(metadata.get("source", "") or ""),
            "metadata": metadata,
        }

    @staticmethod
    def _export_row(row: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "id": row.get("id", ""),
            "session_id": row.get("session_id", ""),
            "student_id": row.get("student_id", ""),
            "name": row.get("name", ""),
            "status": row.get("status", ""),
            "confidence": row.get("confidence", ""),
            "timestamp": row.get("timestamp", ""),
            "snapshot_path": row.get("snapshot_path", row.get("snapshot", "")),
            "dedupe_key": row.get("dedupe_key", ""),
            "metadata": json.dumps(
                row.get("metadata", {}), ensure_ascii=False, sort_keys=True, default=str
            ),
        }

    def __enter__(self) -> "EventStore":
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _traceback: Any) -> None:
        self.close()


# A concise alias for consumers that prefer infrastructure-oriented naming.
ClassroomEventStore = EventStore
