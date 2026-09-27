"""Alert rules and delivery.

Rules:
* watchlist - the plate is on the watchlist (stolen or wanted vehicles).
  With ``watchlist_max_distance`` = 1, a plate that differs by one character
  also alerts, marked as a fuzzy match, because a single OCR error should not
  let a wanted vehicle through.
* mismatch - vehicle-plate verification failed (a possible cloned plate).

Alerts are logged, stored with the event, and optionally POSTed as JSON to
a webhook (control-room dashboard, SMS gateway, chat bot, ...). Webhook
delivery runs on a background thread so a slow receiver never delays video
processing.
"""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
import urllib.request

from .metrics import levenshtein
from .ocr import clean_plate_text

log = logging.getLogger("lpr.alerts")


class AlertManager:
    def __init__(self, store, on_watchlist: bool = True, on_mismatch: bool = True,
                 watchlist_max_distance: int = 1, webhook_url: str | None = None,
                 webhook_timeout: float = 5.0):
        self.store = store
        self.on_watchlist = on_watchlist
        self.on_mismatch = on_mismatch
        self.max_distance = watchlist_max_distance
        self.webhook_url = webhook_url
        self.webhook_timeout = webhook_timeout
        self._queue: queue.Queue = queue.Queue(maxsize=1000)
        if webhook_url:
            threading.Thread(target=self._sender, name="lpr-webhook", daemon=True).start()

    @classmethod
    def from_config(cls, cfg: dict, store) -> "AlertManager":
        a = cfg["alerts"]
        return cls(store, a.get("on_watchlist", True), a.get("on_mismatch", True),
                   int(a.get("watchlist_max_distance", 1)), a.get("webhook_url"),
                   float(a.get("webhook_timeout", 5)))

    def watchlist_hits(self, plate: str) -> list[dict]:
        plate = clean_plate_text(plate)
        if not plate:
            return []
        hits = []
        for entry in self.store.list_watchlist():
            d = levenshtein(plate, entry["plate"])
            if d <= self.max_distance and (d == 0 or len(entry["plate"]) >= 5):
                hits.append({"type": "watchlist", "watch_plate": entry["plate"],
                             "reason": entry["reason"],
                             "match": "exact" if d == 0 else "fuzzy", "distance": d})
        hits.sort(key=lambda h: h["distance"])
        return hits

    def evaluate(self, event: dict) -> list[dict]:
        """Return the alerts triggered by an event dict (plate, verification...)."""
        alerts = []
        if self.on_watchlist:
            alerts += self.watchlist_hits(event.get("plate", ""))
        v = event.get("verification")
        if self.on_mismatch and v and v.get("status") == "mismatch":
            alerts.append({"type": "mismatch", "mismatched": v.get("mismatched", []),
                           "summary": v.get("summary", "")})
        return alerts

    def dispatch(self, event: dict, alerts: list[dict]) -> None:
        for a in alerts:
            log.warning("ALERT %s camera=%s plate=%s %s", a["type"],
                        event.get("camera_id"), event.get("plate"),
                        a.get("reason") or a.get("summary") or "")
        if alerts and self.webhook_url:
            payload = {"event": event, "alerts": alerts}
            try:
                self._queue.put_nowait(payload)
            except queue.Full:
                log.error("webhook queue full, alert dropped for %s", event.get("plate"))

    def _sender(self):
        while True:
            payload = self._queue.get()
            body = json.dumps(payload, default=str).encode()
            req = urllib.request.Request(self.webhook_url, data=body, method="POST",
                                         headers={"Content-Type": "application/json"})
            for attempt in range(3):
                try:
                    with urllib.request.urlopen(req, timeout=self.webhook_timeout):
                        break
                except Exception as e:  # network errors are logged and retried
                    log.error("webhook delivery failed (attempt %d): %s", attempt + 1, e)
                    time.sleep(2 ** attempt)
