import json
import threading
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np

from lpr.alerts import AlertManager
from lpr.storage import EventStore


def store(tmp_path):
    return EventStore(tmp_path / "lpr.db", tmp_path / "snaps")


def test_events_roundtrip_and_query(tmp_path):
    s = store(tmp_path)
    img = np.zeros((40, 60, 3), np.uint8)
    eid = s.add_event("gate", "LEB1234", confidence=0.9, n_reads=5, format_valid=True,
                      details={"k": 1}, alerts=[{"type": "watchlist"}], frame=img, crop=img)
    s.add_event("exit", "ABC999", confidence=0.7)
    ev = s.get_event(eid)
    assert ev["plate"] == "LEB1234" and ev["format_valid"] is True and ev["details"] == {"k": 1}
    assert (tmp_path / "snaps").exists() and ev["frame_path"].endswith(".jpg")
    assert [e["plate"] for e in s.query_events(plate="leb-12")] == ["LEB1234"]
    assert [e["plate"] for e in s.query_events(camera_id="exit")] == ["ABC999"]
    assert [e["id"] for e in s.query_events(alerts_only=True)] == [eid]
    assert len(s.query_events(limit=1)) == 1


def test_retention_purge_deletes_images(tmp_path):
    s = store(tmp_path)
    old = (datetime.now(timezone.utc) - timedelta(days=100)).isoformat()
    eid = s.add_event("gate", "OLD1", frame=np.zeros((5, 5, 3), np.uint8), ts=old)
    path = s.get_event(eid)["frame_path"]
    s.add_event("gate", "NEW1")
    assert s.purge_older_than(90) == 1
    assert s.get_event(eid) is None
    assert not __import__("pathlib").Path(path).exists()
    assert len(s.query_events()) == 1


def test_watchlist_exact_and_fuzzy(tmp_path):
    s = store(tmp_path)
    s.add_watchlist("leb-1234", "stolen")
    a = AlertManager(s, watchlist_max_distance=1)
    assert a.watchlist_hits("LEB1234")[0]["match"] == "exact"
    assert a.watchlist_hits("LEB1284")[0]["match"] == "fuzzy"
    assert a.watchlist_hits("LEB1299") == []
    assert AlertManager(s, watchlist_max_distance=0).watchlist_hits("LEB1284") == []
    assert s.remove_watchlist("LEB1234") and not s.remove_watchlist("LEB1234")


def test_mismatch_alert_and_webhook(tmp_path):
    received = []

    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            received.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            self.send_response(200)
            self.end_headers()

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.handle_request, daemon=True).start()
    a = AlertManager(store(tmp_path),
                     webhook_url=f"http://127.0.0.1:{srv.server_address[1]}/hook")
    event = {"plate": "X1", "camera_id": "c",
             "verification": {"status": "mismatch", "mismatched": ["color"], "summary": "s"}}
    alerts = a.evaluate(event)
    assert alerts == [{"type": "mismatch", "mismatched": ["color"], "summary": "s"}]
    a.dispatch(event, alerts)
    for _ in range(50):
        if received:
            break
        threading.Event().wait(0.05)
    assert received and received[0]["alerts"][0]["type"] == "mismatch"
    srv.server_close()
