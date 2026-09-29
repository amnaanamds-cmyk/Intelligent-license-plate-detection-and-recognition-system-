"""REST API and web app for the LPR system.

Web app (phone and desktop):  GET /            (open in any browser)

    POST   /auth/login                  {"username", "password"} -> bearer token
    POST   /auth/logout
    GET    /auth/me                     current user + whether setup is needed
    GET    /health                      liveness, model and camera status (no auth)
    POST   /recognize                   upload an image, get plate reads   [operator]
    GET    /events                      search the event log               [viewer]
    GET    /events.csv                  same filters, CSV download         [viewer]
    GET    /events/{id}                 one event                          [viewer]
    GET    /events/{id}/frame|crop      evidence images                    [viewer]
    GET    /stats                       dashboard numbers                  [viewer]
    GET    /watchlist                                                      [viewer]
    POST   /watchlist                   {"plate", "reason"}                [operator]
    DELETE /watchlist/{plate}                                              [operator]
    GET    /cameras                     per-camera status                  [viewer]
    POST   /cameras                     start {"id", "source", "stride"}   [operator]
    DELETE /cameras/{id}                                                   [operator]
    GET    /users   POST /users   DELETE /users/{name}                     [admin]
    GET    /audit                       who did what, when                 [admin]

Credentials are accepted as ``Authorization: Bearer <token>`` (from
/auth/login), as the ``X-API-Key`` header (config ``api.api_key``, admin
rights, for integrations), or as a ``?token=`` query parameter so that
<img> tags and download links work in the browser. Interactive docs: /docs.
"""
from __future__ import annotations

import base64
import csv
import io
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
import numpy as np
from fastapi import Depends, FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from lpr.auth import ROLES, AuthManager, Principal, role_at_least
from lpr.pipeline import draw_readings
from lpr.stream import LPRSystem

MAX_UPLOAD_BYTES = 20 * 1024 * 1024
WEB_DIR = Path(__file__).parent / "web"


class WatchlistItem(BaseModel):
    plate: str = Field(min_length=1, max_length=20)
    reason: str = Field(default="", max_length=500)


class CameraSpec(BaseModel):
    id: str = Field(min_length=1, max_length=64)
    source: str = Field(min_length=1, max_length=1000)
    stride: int = Field(default=1, ge=1, le=100)


class LoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1, max_length=256)


class NewUser(BaseModel):
    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1, max_length=256)
    role: str = Field(default="operator")


def _jpeg_b64(img: np.ndarray, max_side: int = 1280) -> str:
    h, w = img.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode() if ok else ""


def _csv_safe(value):
    """Stop spreadsheet formula injection: a cell starting with = + - @ would
    run as a formula when the export is opened in Excel."""
    if isinstance(value, str) and value[:1] in ("=", "+", "-", "@", "\t", "\r"):
        return "'" + value
    return value


def _local_day_start_utc() -> str:
    """Midnight of the server's local day, as ISO UTC (for "today" stats)."""
    now = datetime.now().astimezone()
    return now.replace(hour=0, minute=0, second=0, microsecond=0).astimezone(
        timezone.utc).isoformat()


def create_app(system: LPRSystem, start_cameras: bool = True,
               auth: AuthManager | None = None) -> FastAPI:
    auth = auth or AuthManager(system.store, system.cfg["api"].get("api_key"),
                               ttl_hours=float(system.cfg["api"].get("session_hours", 12)))

    def principal(request: Request) -> Principal:
        header = request.headers.get("authorization", "")
        token = header[7:] if header.lower().startswith("bearer ") else None
        token = token or request.query_params.get("token")
        p = auth.authenticate(request.headers.get("x-api-key"), token)
        if p is None:
            raise HTTPException(401, "login required")
        return p

    def need(role: str):
        def dep(p: Principal = Depends(principal)) -> Principal:
            if not role_at_least(p.role, role):
                raise HTTPException(403, f"requires the {role} role")
            return p
        return dep

    viewer, operator, admin = need("viewer"), need("operator"), need("admin")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if start_cameras:
            system.start()
        yield
        system.stop()

    app = FastAPI(title="License Plate Recognition", version="1.1", lifespan=lifespan)

    # ------------------------------------------------------------ web app --
    if WEB_DIR.exists():
        app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

        @app.get("/", include_in_schema=False)
        def index():
            return FileResponse(WEB_DIR / "index.html")

        @app.get("/manifest.webmanifest", include_in_schema=False)
        def manifest():
            return FileResponse(WEB_DIR / "manifest.webmanifest",
                                media_type="application/manifest+json")

        @app.get("/sw.js", include_in_schema=False)
        def service_worker():
            return FileResponse(WEB_DIR / "sw.js", media_type="application/javascript")

    # --------------------------------------------------------------- auth --
    @app.post("/auth/login")
    def login(body: LoginRequest):
        try:
            res = auth.login(body.username, body.password)
        except PermissionError as e:
            raise HTTPException(429, str(e))
        if res is None:
            system.store.audit(body.username, "login_failed")
            raise HTTPException(401, "wrong username or password")
        token, p = res
        system.store.audit(p.username, "login")
        return {"token": token, "username": p.username, "role": p.role}

    @app.post("/auth/logout")
    def logout(request: Request):
        header = request.headers.get("authorization", "")
        if header.lower().startswith("bearer "):
            auth.logout(header[7:])
        return {"ok": True}

    @app.get("/auth/me")
    def me(request: Request):
        setup = not auth.enabled
        try:
            p = principal(request)
        except HTTPException:
            return {"authenticated": False, "setup_required": setup}
        return {"authenticated": True, "username": p.username, "role": p.role,
                "setup_required": setup}

    # ------------------------------------------------------------- health --
    @app.get("/health")
    def health():
        cams = system.status()
        return {"status": "ok", "models_loaded": system.models_loaded,
                "model_error": system.model_error,
                "verification": bool(system.cfg["verification"].get("enabled")),
                "cameras": len(cams),
                "cameras_connected": sum(bool(c["connected"]) for c in cams)}

    # --------------------------------------------------------- recognize --
    @app.post("/recognize")
    async def recognize(image: UploadFile = File(...),
                        camera_id: str = Query("api-upload", max_length=64),
                        save: bool = Query(False, description="store as events"),
                        images: bool = Query(False, description="include base64 images"),
                        p: Principal = Depends(operator)):
        data = await image.read()
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, "image too large (max 20 MB)")
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(400, "could not decode image")
        try:
            readings = system.recognize_image(img)
        except RuntimeError as e:
            raise HTTPException(503, f"models not available: {e}")
        results = [r.as_dict() for r in readings]
        for r, res in zip(readings, results):
            if save and r.plate_text:
                ev = system.record(
                    camera_id, r.plate_text, confidence=r.ocr.confidence,
                    detection_confidence=r.detection.confidence, raw_text=r.ocr.text,
                    format_valid=r.format_valid, verification=res.get("verification"),
                    details={"source": "upload", "user": p.username},
                    frame=draw_readings(img, [r]), crop=r.crop)
                res["event_id"], res["alerts"] = ev["id"], ev["alerts"]
            else:
                res["alerts"] = system.alerts.evaluate(
                    {"plate": r.plate_text, "verification": res.get("verification")})
            if images:
                res["crop_image"] = _jpeg_b64(r.crop, 480)
                res["enhanced_image"] = _jpeg_b64(r.enhanced, 480)
        out = {"plates": results, "count": len(results)}
        if images:
            out["annotated_image"] = _jpeg_b64(draw_readings(img, readings))
        system.store.audit(p.username, "recognize",
                           f"{len(results)} plate(s) {[r['plate'] for r in results]}"
                           f"{' saved' if save else ''}")
        return out

    # ------------------------------------------------------------- events --
    def _search(plate, camera_id, since, until, alerts_only, limit, offset):
        return system.store.query_events(plate, camera_id, since, until,
                                         alerts_only, limit, offset)

    @app.get("/events")
    def events(plate: str | None = Query(None, max_length=20),
               camera_id: str | None = None,
               since: str | None = Query(None, description="ISO-8601, e.g. 2026-01-31T00:00:00"),
               until: str | None = None, alerts_only: bool = False,
               limit: int = Query(100, ge=1, le=1000), offset: int = Query(0, ge=0),
               p: Principal = Depends(viewer)):
        if plate:
            system.store.audit(p.username, "search", f"plate={plate}")
        return _search(plate, camera_id, since, until, alerts_only, limit, offset)

    @app.get("/events.csv")
    def events_csv(plate: str | None = Query(None, max_length=20),
                   camera_id: str | None = None, since: str | None = None,
                   until: str | None = None, alerts_only: bool = False,
                   limit: int = Query(10000, ge=1, le=100000),
                   p: Principal = Depends(viewer)):
        rows = _search(plate, camera_id, since, until, alerts_only, limit, 0)
        system.store.audit(p.username, "export_csv", f"{len(rows)} rows plate={plate or ''}")
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["id", "timestamp_utc", "camera", "plate", "confidence", "reads",
                    "format_valid", "verification", "alerts", "raw_text"])
        for e in rows:
            w.writerow([_csv_safe(v) for v in (
                e["id"], e["ts"], e["camera_id"], e["plate"],
                f"{e['confidence']:.3f}", e["n_reads"], e["format_valid"],
                e["verification_status"] or "",
                "; ".join(a["type"] for a in e["alerts"]), e["raw_text"])])
        name = f"lpr_events_{datetime.now():%Y%m%d_%H%M}.csv"
        return StreamingResponse(iter([buf.getvalue()]), media_type="text/csv",
                                 headers={"Content-Disposition": f'attachment; filename="{name}"'})

    def _event(event_id: int) -> dict:
        ev = system.store.get_event(event_id)
        if ev is None:
            raise HTTPException(404, "event not found")
        return ev

    @app.get("/events/{event_id}")
    def event(event_id: int, p: Principal = Depends(viewer)):
        return _event(event_id)

    @app.get("/events/{event_id}/{kind}")
    def event_image(event_id: int, kind: str, p: Principal = Depends(viewer)):
        if kind not in ("frame", "crop"):
            raise HTTPException(404, "unknown image kind")
        path = _event(event_id).get(f"{kind}_path")
        if not path or not Path(path).exists():
            raise HTTPException(404, "image not stored")
        return FileResponse(path, media_type="image/jpeg")

    @app.get("/events-recent")
    def recent(limit: int = Query(20, ge=1, le=200), p: Principal = Depends(viewer)):
        return list(system.recent)[:limit]

    @app.get("/stats")
    def stats(p: Principal = Depends(viewer)):
        day_start = _local_day_start_utc()
        last_24h = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()
        s = system.store.stats(day_start, hours_since=last_24h)
        cams = system.status()
        s.update({"since": day_start, "cameras": len(cams),
                  "cameras_online": sum(bool(c["connected"]) for c in cams)})
        return s

    # ---------------------------------------------------------- watchlist --
    @app.get("/watchlist")
    def watchlist(p: Principal = Depends(viewer)):
        return system.store.list_watchlist()

    @app.post("/watchlist", status_code=201)
    def add_watch(item: WatchlistItem, p: Principal = Depends(operator)):
        try:
            plate = system.store.add_watchlist(item.plate, item.reason)
        except ValueError as e:
            raise HTTPException(400, str(e))
        system.store.audit(p.username, "watchlist_add", f"{plate}: {item.reason}")
        return {"plate": plate}

    @app.delete("/watchlist/{plate}")
    def remove_watch(plate: str, p: Principal = Depends(operator)):
        if not system.store.remove_watchlist(plate):
            raise HTTPException(404, "plate not on watchlist")
        system.store.audit(p.username, "watchlist_remove", plate)
        return {"removed": plate}

    # ------------------------------------------------------------ cameras --
    @app.get("/cameras")
    def cameras(p: Principal = Depends(viewer)):
        return system.status()

    @app.post("/cameras", status_code=201)
    def start_camera(spec: CameraSpec, p: Principal = Depends(operator)):
        try:
            w = system.start_camera(spec.model_dump())
        except ValueError as e:
            raise HTTPException(409, str(e))
        except Exception as e:
            raise HTTPException(503, f"camera could not start: {e}")
        system.store.audit(p.username, "camera_start", spec.id)
        return w.status()

    @app.delete("/cameras/{camera_id}")
    def stop_camera(camera_id: str, p: Principal = Depends(operator)):
        w = system.cameras.get(camera_id)
        if w is None:
            if system.camera_errors.pop(camera_id, None) is not None:
                return {"stopped": camera_id}
            raise HTTPException(404, "unknown camera")
        w.stop()
        w.join(timeout=10)
        del system.cameras[camera_id]
        system.store.audit(p.username, "camera_stop", camera_id)
        return {"stopped": camera_id}

    # -------------------------------------------------------------- admin --
    @app.get("/users")
    def users(p: Principal = Depends(admin)):
        return system.store.list_users()

    @app.post("/users", status_code=201)
    def add_user(body: NewUser, p: Principal = Depends(admin)):
        if body.role not in ROLES:
            raise HTTPException(400, f"role must be one of {list(ROLES)}")
        if system.store.get_user(body.username.strip()):
            raise HTTPException(409, "user already exists")
        try:
            auth.create_user(body.username, body.password, body.role)
        except ValueError as e:
            raise HTTPException(400, str(e))
        system.store.audit(p.username, "user_add", f"{body.username} ({body.role})")
        return {"username": body.username.strip(), "role": body.role}

    @app.delete("/users/{username}")
    def delete_user(username: str, p: Principal = Depends(admin)):
        if username == p.username:
            raise HTTPException(400, "you cannot delete your own account")
        try:
            deleted = auth.delete_user(username)
        except ValueError as e:
            raise HTTPException(400, str(e))
        if not deleted:
            raise HTTPException(404, "unknown user")
        system.store.audit(p.username, "user_delete", username)
        return {"deleted": username}

    @app.get("/audit")
    def audit(limit: int = Query(200, ge=1, le=5000), username: str | None = None,
              p: Principal = Depends(admin)):
        return system.store.query_audit(limit, username)

    return app
