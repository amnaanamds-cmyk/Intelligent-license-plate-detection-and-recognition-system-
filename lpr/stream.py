"""Live processing of camera streams (RTSP / HTTP / webcam / video file).

    FrameSource   reads frames. For a live stream a background thread always
                  keeps only the newest frame, so processing never falls
                  behind real time. It reconnects with backoff when the
                  camera drops.
    CameraWorker  one thread per camera: track plates -> OCR -> vote -> event.
    LPRSystem     builds everything from the config and owns the event
                  store, the alerts and the camera workers.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from .alerts import AlertManager
from .pipeline import LicensePlatePipeline, build_ocr, verification_dict
from .storage import EventStore, utc_now
from .tracking import PlateTrack, TrackVoter

log = logging.getLogger("lpr.stream")


def is_live_source(source) -> bool:
    s = str(source)
    return s.isdigit() or s.lower().startswith(("rtsp://", "rtmp://", "http://", "https://"))


class FrameSource:
    def __init__(self, source, reconnect_delay: float = 2.0, max_delay: float = 60.0):
        self.source = source
        self.live = is_live_source(source)
        self.reconnect_delay = reconnect_delay
        self.max_delay = max_delay
        self.connected = False
        self.finished = False
        self._cap = None
        self._frame = None
        self._frame_id = 0
        self._last_returned = 0
        self._cond = threading.Condition()
        self._stop = threading.Event()
        if self.live:
            threading.Thread(target=self._reader, daemon=True,
                             name=f"lpr-reader-{source}").start()
        else:
            if not Path(str(source)).exists():
                raise FileNotFoundError(source)
            self._cap = cv2.VideoCapture(str(source))
            self.connected = self._cap.isOpened()

    def _open(self):
        src = int(self.source) if str(self.source).isdigit() else self.source
        cap = cv2.VideoCapture(src)
        if isinstance(src, str) and src.startswith("rtsp"):
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def _reader(self):
        delay = self.reconnect_delay
        while not self._stop.is_set():
            cap = self._open()
            if not cap.isOpened():
                self.connected = False
                log.warning("cannot open %s, retrying in %.0fs", self.source, delay)
                self._stop.wait(delay)
                delay = min(delay * 2, self.max_delay)
                continue
            self.connected, delay = True, self.reconnect_delay
            log.info("connected to %s", self.source)
            while not self._stop.is_set():
                ok, frame = cap.read()
                if not ok:
                    log.warning("stream %s dropped, reconnecting", self.source)
                    break
                with self._cond:
                    self._frame, self._frame_id = frame, self._frame_id + 1
                    self._cond.notify_all()
            cap.release()
            self.connected = False

    def read(self, timeout: float = 5.0) -> np.ndarray | None:
        """Return the next frame, or None on timeout or at the end of a file."""
        if not self.live:
            ok, frame = self._cap.read()
            if not ok:
                self.finished = True
                self.connected = False
                return None
            return frame
        with self._cond:
            if not self._cond.wait_for(
                    lambda: self._frame_id != self._last_returned or self._stop.is_set(),
                    timeout):
                return None
            self._last_returned = self._frame_id
            return self._frame

    def close(self):
        self._stop.set()
        with self._cond:
            self._cond.notify_all()
        if self._cap is not None:
            self._cap.release()


class CameraWorker(threading.Thread):
    def __init__(self, camera_id: str, source, pipeline: LicensePlatePipeline,
                 voter: TrackVoter, on_track, stride: int = 1,
                 tracker: str = "bytetrack.yaml"):
        super().__init__(name=f"lpr-camera-{camera_id}", daemon=True)
        self.camera_id = camera_id
        self.source_spec = source
        self.pipeline = pipeline
        self.voter = voter
        self.on_track = on_track
        self.stride = max(1, int(stride))
        self.tracker = tracker
        self._stop_event = threading.Event()
        self.source: FrameSource | None = None
        self.frames = 0
        self.processed = 0
        self.events = 0
        self.fps = 0.0
        self.error: str | None = None

    def stop(self):
        self._stop_event.set()
        if self.source:
            self.source.close()

    def status(self) -> dict:
        return {"camera_id": self.camera_id, "source": _redact(self.source_spec),
                "running": self.is_alive(),
                "connected": bool(self.source and self.source.connected),
                "finished": bool(self.source and self.source.finished),
                "frames": self.frames, "processed": self.processed,
                "fps": round(self.fps, 2), "events": self.events,
                "active_tracks": len(self.voter.tracks), "error": self.error}

    def process_frame(self, frame: np.ndarray) -> None:
        idx = self.processed
        for det in self.pipeline.plate_detector.track(frame, self.tracker):
            if det.track_id is None:
                continue
            self.voter.seen(det.track_id, idx)
            if not (self.pipeline.readable(det) and self.voter.needs_ocr(det.track_id)):
                continue
            reading = self.pipeline.read_plate(frame, det)
            if reading is None or not reading.plate_text:
                continue
            if self.pipeline.reject_invalid and reading.format_valid is False:
                continue
            self.voter.add(det.track_id, idx, reading, frame,
                           bool(reading.format_valid) if reading.format_valid is not None
                           else False)
        self.processed += 1
        for track in self.voter.collect(self.processed):
            self._emit(track)

    def _emit(self, track: PlateTrack):
        try:
            self.on_track(self.camera_id, track, self.pipeline)
            self.events += 1
        except Exception:
            log.exception("failed to handle track %s on %s", track.track_id, self.camera_id)

    def run(self):
        try:
            self.source = FrameSource(self.source_spec)
        except Exception as e:
            self.error = str(e)
            log.error("camera %s: %s", self.camera_id, e)
            return
        t0, n0 = time.time(), 0
        while not self._stop_event.is_set():
            frame = self.source.read()
            if frame is None:
                if self.source.finished:
                    break
                continue
            self.frames += 1
            if self.frames % self.stride:
                continue
            try:
                self.process_frame(frame)
            except Exception as e:
                self.error = str(e)
                log.exception("camera %s: frame processing failed", self.camera_id)
            now = time.time()
            if now - t0 >= 2.0:
                self.fps, t0, n0 = (self.processed - n0) / (now - t0), now, self.processed
        for track in self.voter.flush():
            self._emit(track)
        self.source.close()
        log.info("camera %s stopped (%d frames, %d events)", self.camera_id,
                 self.frames, self.events)


def _redact(source) -> str:
    """Hide the password in rtsp://user:pass@host URLs."""
    s = str(source)
    if "://" in s and "@" in s:
        scheme, rest = s.split("://", 1)
        creds, host = rest.rsplit("@", 1)
        return f"{scheme}://{creds.split(':')[0]}:***@{host}"
    return s


class LPRSystem:
    def __init__(self, cfg: dict, store: EventStore | None = None,
                 alerts: AlertManager | None = None, ocr=None):
        self.cfg = cfg
        self.store = store or EventStore.from_config(cfg)
        self.alerts = alerts or AlertManager.from_config(cfg, self.store)
        self._ocr = ocr
        self.cameras: dict[str, CameraWorker] = {}
        self.recent: deque = deque(maxlen=200)
        self._image_pipeline: LicensePlatePipeline | None = None
        self._image_lock = threading.Lock()
        self._stop = threading.Event()
        self._listeners: list = []
        self.model_error: str | None = None
        self.camera_errors: dict[str, str] = {}

    # ------------------------------------------------------------ models --
    @property
    def ocr(self):
        if self._ocr is None:
            self._ocr = build_ocr(self.cfg)
        return self._ocr

    def make_pipeline(self) -> LicensePlatePipeline:
        """Each camera gets its own detector, because the tracker state lives
        in the model. The OCR engine is shared."""
        weights = self.cfg["detector"]["weights"]
        if not weights.startswith(("yolo", "http")) and not Path(weights).exists():
            raise FileNotFoundError(
                f"plate detector weights not found: {weights}. Train the model "
                f"(scripts/train.py) or set detector.weights in the config")
        return LicensePlatePipeline.from_config(self.cfg, ocr=self.ocr)

    @property
    def image_pipeline(self) -> LicensePlatePipeline:
        if self._image_pipeline is None:
            self._image_pipeline = self.make_pipeline()
        return self._image_pipeline

    def load_models(self) -> bool:
        """Load the models now. On failure (missing weights, OCR models not
        downloadable, ...) the error is kept in ``model_error`` and the
        service keeps running, so the web app can show what is wrong."""
        try:
            with self._image_lock:
                self.image_pipeline
            self.model_error = None
        except Exception as e:
            self.model_error = f"{type(e).__name__}: {e}"
            if "network" in str(e).lower() or "hosting" in str(e).lower():
                self.model_error += (" - the first start downloads the PaddleOCR models, so it "
                                     "needs internet access once (or set ocr.det_model_dir / "
                                     "ocr.rec_model_dir to local copies)")
            log.error("model loading failed: %s", self.model_error)
        return self.model_error is None

    @property
    def models_loaded(self) -> bool:
        return self._image_pipeline is not None

    def recognize_image(self, image: np.ndarray) -> list:
        if self._image_pipeline is None and not self.load_models():
            raise RuntimeError(self.model_error)
        with self._image_lock:
            return self.image_pipeline.process(image)

    # ------------------------------------------------------------ events --
    def add_listener(self, fn) -> None:
        """``fn(event_dict)`` is called for every stored event."""
        self._listeners.append(fn)

    def record(self, camera_id: str, plate: str, *, confidence: float,
               detection_confidence: float, raw_text: str = "", n_reads: int = 1,
               format_valid: bool | None = None, verification: dict | None = None,
               details: dict | None = None, frame=None, crop=None) -> dict:
        event = {"camera_id": camera_id, "plate": plate, "ts": utc_now(),
                 "confidence": round(float(confidence), 4),
                 "detection_confidence": round(float(detection_confidence), 4),
                 "n_reads": n_reads, "format_valid": format_valid,
                 "verification": verification}
        alerts = self.alerts.evaluate(event)
        event["id"] = self.store.add_event(
            camera_id, plate, raw_text=raw_text, confidence=confidence,
            detection_confidence=detection_confidence, n_reads=n_reads,
            format_valid=format_valid,
            verification_status=verification["status"] if verification else None,
            details={**(details or {}), **({"verification": verification}
                                           if verification else {})},
            alerts=alerts, frame=frame, crop=crop, ts=event["ts"])
        event["alerts"] = alerts
        self.alerts.dispatch(event, alerts)
        self.recent.appendleft(event)
        for fn in list(self._listeners):
            try:
                fn(event)
            except Exception:
                log.exception("event listener failed")
        log.info("event #%d %s plate=%s conf=%.2f reads=%d alerts=%d", event["id"],
                 camera_id, plate, confidence, n_reads, len(alerts))
        return event

    def handle_track(self, camera_id: str, track: PlateTrack,
                     pipeline: LicensePlatePipeline) -> dict | None:
        text, conf, agreement = track.consensus()
        best = track.best_read
        if not text or best is None:
            return None
        frame = track.best_frame
        verification = None
        if pipeline.verification_enabled and frame is not None:
            _, _, v = pipeline.verify(frame, best.detection.box, text)
            verification = verification_dict(v)
        valid = pipeline.formatter.format(text).valid if pipeline.formatter else None
        annotated = frame
        if frame is not None:
            annotated = frame.copy()
            x1, y1, x2, y2 = best.detection.box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 200, 0), 2)
        return self.record(
            camera_id, text, confidence=conf,
            detection_confidence=best.detection.confidence,
            raw_text=best.ocr.text, n_reads=len(track.reads), format_valid=valid,
            verification=verification,
            details={"track_id": track.track_id, "agreement": round(agreement, 3),
                     "reads": [r.text for r in track.reads][:30]},
            frame=annotated, crop=best.crop)

    # ----------------------------------------------------------- cameras --
    def start_camera(self, cam: dict) -> CameraWorker:
        cid = str(cam["id"])
        if cid in self.cameras and self.cameras[cid].is_alive():
            raise ValueError(f"camera {cid} is already running")
        t = self.cfg["tracking"]
        worker = CameraWorker(
            cid, cam["source"], self.make_pipeline(),
            TrackVoter(t["min_reads"], t["agreement"], t["max_reads"], t["lost_frames"]),
            self.handle_track, stride=cam.get("stride", 1), tracker=t["tracker"])
        self.cameras[cid] = worker
        worker.start()
        return worker

    def start(self) -> None:
        for cam in self.cfg.get("cameras") or []:
            if cam.get("enabled", True):
                try:
                    self.start_camera(cam)
                    self.camera_errors.pop(str(cam["id"]), None)
                except Exception as e:
                    # A broken camera or missing model must not take down the
                    # service. The error is reported in status().
                    self.camera_errors[str(cam.get("id"))] = str(e)
                    log.error("camera %s could not start: %s", cam.get("id"), e)
        days = int(self.cfg["storage"].get("retention_days") or 0)
        if days > 0:
            threading.Thread(target=self._retention_loop, args=(days,),
                             daemon=True, name="lpr-retention").start()

    def _retention_loop(self, days: int):
        while not self._stop.is_set():
            try:
                n = self.store.purge_older_than(days)
                if n:
                    log.info("retention: purged %d events older than %d days", n, days)
            except Exception:
                log.exception("retention purge failed")
            self._stop.wait(6 * 3600)

    def stop(self) -> None:
        self._stop.set()
        for w in self.cameras.values():
            w.stop()
        for w in self.cameras.values():
            w.join(timeout=10)

    def status(self) -> list[dict]:
        out = [w.status() for w in self.cameras.values()]
        for cid, err in self.camera_errors.items():
            if cid not in self.cameras:
                out.append({"camera_id": cid, "running": False, "connected": False,
                            "finished": False, "frames": 0, "processed": 0, "fps": 0.0,
                            "events": 0, "active_tracks": 0, "error": err, "source": ""})
        return out
