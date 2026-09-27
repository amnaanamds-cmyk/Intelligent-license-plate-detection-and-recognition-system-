"""Event store: a SQLite record of every plate read, plus snapshot images
kept as evidence."""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
import numpy as np

from .ocr import clean_plate_text

SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    ts                    TEXT NOT NULL,              -- ISO-8601 UTC
    camera_id             TEXT NOT NULL,
    plate                 TEXT NOT NULL,
    raw_text              TEXT,
    confidence            REAL,
    detection_confidence  REAL,
    n_reads               INTEGER DEFAULT 1,
    format_valid          INTEGER,
    verification_status   TEXT,
    details               TEXT,                       -- JSON
    alerts                TEXT,                       -- JSON list
    frame_path            TEXT,
    crop_path             TEXT
);
CREATE INDEX IF NOT EXISTS idx_events_plate ON events(plate);
CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts);
CREATE INDEX IF NOT EXISTS idx_events_camera ON events(camera_id, ts);

CREATE TABLE IF NOT EXISTS watchlist (
    plate     TEXT PRIMARY KEY,
    reason    TEXT,
    added_ts  TEXT NOT NULL
);
"""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class EventStore:
    def __init__(self, database: str | Path = "data/lpr.db",
                 snapshot_dir: str | Path = "data/snapshots",
                 save_snapshots: bool = True):
        self.database = str(database)
        if self.database != ":memory:":
            Path(self.database).parent.mkdir(parents=True, exist_ok=True)
        self.snapshot_dir = Path(snapshot_dir)
        self.save_snapshots = save_snapshots
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.database, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    @classmethod
    def from_config(cls, cfg: dict) -> "EventStore":
        s = cfg["storage"]
        return cls(s["database"], s["snapshot_dir"], bool(s.get("save_snapshots", True)))

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # ------------------------------------------------------------ events --
    def _save_image(self, ts: str, camera_id: str, kind: str,
                    image: np.ndarray | None) -> str | None:
        if not self.save_snapshots or image is None or image.size == 0:
            return None
        day = ts[:10]
        safe_cam = "".join(c if c.isalnum() or c in "-_" else "_" for c in camera_id)
        folder = self.snapshot_dir / day / safe_cam
        folder.mkdir(parents=True, exist_ok=True)
        stamp = ts[11:23].replace(":", "").replace(".", "")
        path = folder / f"{stamp}_{kind}_{np.random.randint(1 << 30):08x}.jpg"
        cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, 90])
        return str(path)

    def add_event(self, camera_id: str, plate: str, *, raw_text: str = "",
                  confidence: float = 0.0, detection_confidence: float = 0.0,
                  n_reads: int = 1, format_valid: bool | None = None,
                  verification_status: str | None = None,
                  details: dict | None = None, alerts: list | None = None,
                  frame: np.ndarray | None = None, crop: np.ndarray | None = None,
                  ts: str | None = None) -> int:
        ts = ts or utc_now()
        frame_path = self._save_image(ts, camera_id, "frame", frame)
        crop_path = self._save_image(ts, camera_id, "plate", crop)
        with self._lock:
            cur = self._conn.execute(
                """INSERT INTO events (ts, camera_id, plate, raw_text, confidence,
                   detection_confidence, n_reads, format_valid, verification_status,
                   details, alerts, frame_path, crop_path)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (ts, camera_id, plate, raw_text, float(confidence),
                 float(detection_confidence), int(n_reads),
                 None if format_valid is None else int(format_valid),
                 verification_status, json.dumps(details or {}),
                 json.dumps(alerts or []), frame_path, crop_path))
            self._conn.commit()
            return int(cur.lastrowid)

    def set_alerts(self, event_id: int, alerts: list) -> None:
        with self._lock:
            self._conn.execute("UPDATE events SET alerts=? WHERE id=?",
                               (json.dumps(alerts), event_id))
            self._conn.commit()

    @staticmethod
    def _row(r: sqlite3.Row) -> dict:
        d = dict(r)
        d["details"] = json.loads(d["details"] or "{}")
        d["alerts"] = json.loads(d["alerts"] or "[]")
        if d["format_valid"] is not None:
            d["format_valid"] = bool(d["format_valid"])
        return d

    def get_event(self, event_id: int) -> dict | None:
        with self._lock:
            r = self._conn.execute("SELECT * FROM events WHERE id=?", (event_id,)).fetchone()
        return self._row(r) if r else None

    def query_events(self, plate: str | None = None, camera_id: str | None = None,
                     since: str | None = None, until: str | None = None,
                     alerts_only: bool = False, limit: int = 100,
                     offset: int = 0) -> list[dict]:
        """``plate`` matches partially (``LEB12`` finds ``LEB1234``). Times are
        ISO-8601 strings."""
        sql, args = "SELECT * FROM events WHERE 1=1", []
        if plate:
            sql += " AND plate LIKE ?"
            args.append(f"%{clean_plate_text(plate)}%")
        if camera_id:
            sql += " AND camera_id = ?"
            args.append(camera_id)
        if since:
            sql += " AND ts >= ?"
            args.append(since)
        if until:
            sql += " AND ts <= ?"
            args.append(until)
        if alerts_only:
            sql += " AND alerts != '[]'"
        sql += " ORDER BY ts DESC, id DESC LIMIT ? OFFSET ?"
        args += [int(limit), int(offset)]
        with self._lock:
            rows = self._conn.execute(sql, args).fetchall()
        return [self._row(r) for r in rows]

    def purge_older_than(self, days: int) -> int:
        """Delete events (and their images) older than ``days``. Returns the
        number of events deleted. Supports a data-retention policy."""
        if days <= 0:
            return 0
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        with self._lock:
            rows = self._conn.execute(
                "SELECT frame_path, crop_path FROM events WHERE ts < ?", (cutoff,)).fetchall()
            self._conn.execute("DELETE FROM events WHERE ts < ?", (cutoff,))
            self._conn.commit()
        for r in rows:
            for p in (r["frame_path"], r["crop_path"]):
                if p:
                    Path(p).unlink(missing_ok=True)
        return len(rows)

    # --------------------------------------------------------- watchlist --
    def add_watchlist(self, plate: str, reason: str = "") -> str:
        key = clean_plate_text(plate)
        if not key:
            raise ValueError("empty plate")
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO watchlist (plate, reason, added_ts) VALUES (?,?,?)",
                (key, reason, utc_now()))
            self._conn.commit()
        return key

    def remove_watchlist(self, plate: str) -> bool:
        with self._lock:
            cur = self._conn.execute("DELETE FROM watchlist WHERE plate=?",
                                     (clean_plate_text(plate),))
            self._conn.commit()
        return cur.rowcount > 0

    def list_watchlist(self) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM watchlist ORDER BY added_ts DESC").fetchall()
        return [dict(r) for r in rows]
