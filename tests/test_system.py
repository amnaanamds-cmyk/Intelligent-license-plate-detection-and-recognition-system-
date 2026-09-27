"""Video processing and the REST API, tested with stub models."""
import cv2
import numpy as np
import pytest

from lpr.config import load_config
from lpr.detector import Detection
from lpr.ocr import OCRResult
from lpr.pipeline import LicensePlatePipeline
from lpr.postprocess import PlateFormatter
from lpr.stream import CameraWorker, LPRSystem, _redact
from lpr.tracking import TrackVoter


class StubTracker:
    """Pretends to track one plate that moves across the frame."""

    def __init__(self):
        self.n = 0

    def track(self, frame, tracker="bytetrack.yaml"):
        self.n += 1
        if self.n > 8:  # car has left
            return []
        x = 50 + 10 * self.n
        return [Detection((x, 200, x + 150, 240), 0.9, track_id=1)]

    detect = track


class NoisyOCR:
    """Mostly right, sometimes wrong, the way real per-frame OCR behaves."""

    def __init__(self, reads):
        self.reads, self.i = reads, 0

    def recognize(self, img):
        t = self.reads[self.i % len(self.reads)]
        self.i += 1
        return OCRResult(t, 0.9)


def make_video(path, n=20):
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (640, 480))
    for i in range(n):
        f = np.full((480, 640, 3), 90, np.uint8)
        cv2.putText(f, str(i), (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        w.write(f)
    w.release()


def system(tmp_path, **over):
    cfg = load_config(overrides={"storage": {"database": str(tmp_path / "db.sqlite"),
                                             "snapshot_dir": str(tmp_path / "snaps")},
                                 "detector": {"min_plate_width": 0}, **over})
    return LPRSystem(cfg)


def stub_pipeline(reads):
    return LicensePlatePipeline(StubTracker(), NoisyOCR(reads),
                                formatter=PlateFormatter(["LLLDDDD"], ["PUNJAB"]))


def test_camera_worker_votes_and_stores_one_event(tmp_path):
    video = tmp_path / "v.avi"
    make_video(video)
    sys_ = system(tmp_path)
    sys_.store.add_watchlist("LEB1234", "stolen")
    w = CameraWorker("gate", str(video), stub_pipeline(["LEB1234", "LEB1Z34", "LE81234"]),
                     TrackVoter(min_reads=3, lost_frames=5), sys_.handle_track)
    w.start()
    w.join(timeout=30)
    assert not w.is_alive()
    events = sys_.store.query_events()
    assert len(events) == 1
    ev = events[0]
    # "LEB1Z34" / "LE81234" are fixed by the format rules, so every read agrees
    assert ev["plate"] == "LEB1234" and ev["format_valid"] is True
    assert ev["alerts"][0]["type"] == "watchlist"
    assert ev["frame_path"] and ev["crop_path"]
    assert w.status()["events"] == 1 and w.status()["finished"]


def test_redact_rtsp_password():
    assert _redact("rtsp://admin:secret@10.0.0.5:554/s") == "rtsp://admin:***@10.0.0.5:554/s"
    assert _redact("video.mp4") == "video.mp4"


@pytest.fixture
def client(tmp_path):
    from fastapi.testclient import TestClient

    from service.api import create_app

    s = system(tmp_path, api={"api_key": "k"})
    s._image_pipeline = stub_pipeline(["PUNJAB LEB 1234"])
    with TestClient(create_app(s, start_cameras=False)) as c:
        c.headers["X-API-Key"] = "k"
        yield c


def test_api_auth(client):
    assert client.get("/health").status_code == 200
    assert client.get("/events", headers={"X-API-Key": "wrong"}).status_code == 401


def test_api_recognize_watchlist_and_events(client):
    assert client.post("/watchlist", json={"plate": "leb-1234", "reason": "stolen"}).status_code == 201
    ok, buf = cv2.imencode(".jpg", np.full((480, 640, 3), 90, np.uint8))
    r = client.post("/recognize?save=true&camera_id=desk",
                    files={"image": ("car.jpg", buf.tobytes(), "image/jpeg")})
    assert r.status_code == 200, r.text
    [p] = r.json()["plates"]
    assert p["plate"] == "LEB1234" and p["format_valid"] and p["alerts"][0]["match"] == "exact"
    ev = client.get(f"/events/{p['event_id']}").json()
    assert ev["camera_id"] == "desk"
    assert client.get(f"/events/{p['event_id']}/crop").headers["content-type"] == "image/jpeg"
    assert len(client.get("/events", params={"plate": "LEB"}).json()) == 1
    assert client.get("/events/999").status_code == 404
    assert client.delete("/watchlist/LEB1234").status_code == 200
    bad = client.post("/recognize", files={"image": ("x.jpg", b"notanimage", "image/jpeg")})
    assert bad.status_code == 400
