"""Accounts, roles, dashboard stats, CSV export and the web app."""
import cv2
import numpy as np
import pytest

from lpr.auth import AuthManager, hash_password, verify_password
from lpr.storage import EventStore
from test_system import stub_pipeline, system


def test_password_hashing():
    h = hash_password("correct horse")
    assert h.startswith("pbkdf2_sha256$") and "correct" not in h
    assert verify_password("correct horse", h)
    assert not verify_password("wrong", h)
    assert not verify_password("x", "garbage")
    assert hash_password("same") != hash_password("same")  # salted


def test_auth_manager_setup_mode_and_login(tmp_path):
    s = EventStore(tmp_path / "db", tmp_path / "snaps")
    a = AuthManager(s)
    assert not a.enabled and a.authenticate().role == "admin"  # fresh install
    with pytest.raises(ValueError):
        a.create_user("bob", "short", "viewer")
    a.create_user("bob", "longenough", "viewer")
    assert a.enabled and a.authenticate() is None
    assert a.login("bob", "nope") is None
    token, p = a.login("bob", "longenough")
    assert p.role == "viewer" and a.authenticate(token=token).username == "bob"
    a.create_user("root", "rootpass1", "admin")
    a.delete_user("bob")
    assert a.authenticate(token=token) is None  # deleting a user ends their sessions
    with pytest.raises(ValueError):
        a.delete_user("root")                    # never delete the last admin
    s.delete_user("root")                        # even if all users vanish...
    assert a.enabled and a.authenticate() is None  # ...it never reopens setup mode


def test_login_rate_limit(tmp_path):
    a = AuthManager(EventStore(tmp_path / "db", tmp_path / "s"))
    a.create_user("eve", "password1")
    for _ in range(5):
        assert a.login("eve", "bad") is None
    with pytest.raises(PermissionError):
        a.login("eve", "password1")


@pytest.fixture
def app_client(tmp_path):
    from fastapi.testclient import TestClient

    from service.api import create_app

    s = system(tmp_path)
    s._image_pipeline = stub_pipeline(["LEB 1234"])
    with TestClient(create_app(s, start_cameras=False)) as c:
        yield c, s


def login(c, user, pw):
    r = c.post("/auth/login", json={"username": user, "password": pw})
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['token']}"}


def jpeg():
    return cv2.imencode(".jpg", np.full((480, 640, 3), 90, np.uint8))[1].tobytes()


def test_web_app_served(app_client):
    c, _ = app_client
    r = c.get("/")
    assert r.status_code == 200 and "PlateVision" in r.text
    assert c.get("/static/app.js").status_code == 200
    assert c.get("/manifest.webmanifest").json()["short_name"] == "PlateVision"
    assert c.get("/static/icon-192.png").headers["content-type"] == "image/png"


def test_first_run_setup_then_roles(app_client):
    c, s = app_client
    assert c.get("/auth/me").json() == {"authenticated": True, "username": "setup",
                                        "role": "admin", "setup_required": True}
    # first visit creates the admin, after which login is required
    assert c.post("/users", json={"username": "admin", "password": "adminpass1",
                                  "role": "admin"}).status_code == 201
    assert c.get("/events").status_code == 401
    assert c.get("/auth/me").json()["setup_required"] is False
    admin = login(c, "admin", "adminpass1")
    assert c.post("/users", headers=admin, json={"username": "viewer1", "password": "viewerpass",
                                                 "role": "viewer"}).status_code == 201
    assert c.post("/users", headers=admin, json={"username": "op", "password": "operator1",
                                                 "role": "operator"}).status_code == 201
    viewer, op = login(c, "viewer1", "viewerpass"), login(c, "op", "operator1")

    # viewers read, but cannot scan, edit the watchlist or manage users
    assert c.get("/events", headers=viewer).status_code == 200
    assert c.post("/recognize", headers=viewer,
                  files={"image": ("a.jpg", jpeg(), "image/jpeg")}).status_code == 403
    assert c.post("/watchlist", headers=viewer, json={"plate": "X1"}).status_code == 403
    assert c.get("/users", headers=viewer).status_code == 403
    # operators scan and manage the watchlist, but not users
    assert c.post("/watchlist", headers=op, json={"plate": "LEB1234", "reason": "t"}).status_code == 201
    r = c.post("/recognize?save=true&images=true", headers=op,
               files={"image": ("a.jpg", jpeg(), "image/jpeg")})
    assert r.status_code == 200
    body = r.json()
    assert body["annotated_image"].startswith("data:image/jpeg;base64,")
    assert body["plates"][0]["crop_image"].startswith("data:image/jpeg")
    assert body["plates"][0]["alerts"][0]["type"] == "watchlist"
    assert c.get("/audit", headers=op).status_code == 403
    # admin sees who did what
    actions = [a["action"] for a in c.get("/audit", headers=admin).json()]
    for expected in ("login", "user_add", "watchlist_add", "recognize"):
        assert expected in actions
    assert c.delete("/users/admin", headers=admin).status_code == 400  # not yourself
    # token in query string works for <img> tags
    eid = body["plates"][0]["event_id"]
    token = viewer["Authorization"][7:]
    assert c.get(f"/events/{eid}/crop?token={token}").status_code == 200
    assert c.get(f"/events/{eid}/crop?token=bad").status_code == 401
    # logout ends the session
    c.post("/auth/logout", headers=viewer)
    assert c.get("/events", headers=viewer).status_code == 401


def test_stats_and_csv(app_client):
    c, s = app_client
    s.store.add_watchlist("ABC123")
    s.record("gate", "ABC123", confidence=0.9, detection_confidence=0.9)
    s.record("gate", "XYZ789", confidence=0.8, detection_confidence=0.9)
    s.record("exit", "ABC123", confidence=0.95, detection_confidence=0.9)
    st = c.get("/stats").json()
    assert st["events"] == 3 and st["unique_plates"] == 2 and st["alerts"] == 2
    assert st["per_camera"][0] == {"camera_id": "gate", "n": 2}
    assert sum(h["count"] for h in st["hourly"]) == 3
    assert st["top_plates"][0] == {"plate": "ABC123", "n": 2}
    r = c.get("/events.csv?plate=ABC")
    assert r.headers["content-type"].startswith("text/csv")
    lines = r.text.strip().splitlines()
    assert lines[0].startswith("id,timestamp_utc,camera,plate") and len(lines) == 3
    assert "watchlist" in lines[1]


def test_csv_formula_injection_neutralised(app_client):
    c, s = app_client
    s.record("=HYPERLINK(\"http://x\")", "ABC123", confidence=0.9, detection_confidence=0.9)
    line = c.get("/events.csv").text.splitlines()[1]
    assert "'=HYPERLINK" in line


def test_missing_model_reported_not_crashing(tmp_path):
    from fastapi.testclient import TestClient

    from service.api import create_app

    s = system(tmp_path, detector={"weights": str(tmp_path / "missing.pt"), "min_plate_width": 0})
    assert not s.load_models() and "not found" in s.model_error
    with TestClient(create_app(s, start_cameras=False)) as c:
        h = c.get("/health").json()
        assert h["models_loaded"] is False and "not found" in h["model_error"]
        r = c.post("/recognize", files={"image": ("a.jpg", jpeg(), "image/jpeg")})
        assert r.status_code == 503 and "not found" in r.json()["detail"]


def test_camera_start_failure_is_reported(tmp_path):
    s = system(tmp_path, detector={"weights": str(tmp_path / "missing.pt")},
               cameras=[{"id": "gate", "source": "rtsp://10.0.0.1/x"}])
    s.start()  # must not raise
    [st] = s.status()
    assert st["camera_id"] == "gate" and st["error"] and not st["running"]
    s.stop()
