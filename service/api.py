"""REST API for integrating the LPR system with other software (toll,
parking, gate barrier, control room).

    GET    /health                      liveness + camera status
    POST   /recognize                   upload an image, get plate reads
    GET    /events                      search the event log
    GET    /events/{id}                 one event
    GET    /events/{id}/frame|crop      evidence images
    GET    /watchlist                   list watchlisted plates
    POST   /watchlist                   {"plate": "...", "reason": "..."}
    DELETE /watchlist/{plate}
    GET    /cameras                     per-camera status (fps, connection, ...)
    POST   /cameras                     start a camera {"id", "source", "stride"}
    DELETE /cameras/{id}                stop a camera

Interactive documentation is served at /docs. When ``api.api_key`` is set,
every request except /health must send the ``X-API-Key`` header.
"""
from __future__ import annotations

import secrets
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
from fastapi import Depends, FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from lpr.stream import LPRSystem

MAX_UPLOAD_BYTES = 20 * 1024 * 1024


class WatchlistItem(BaseModel):
    plate: str = Field(min_length=1, max_length=20)
    reason: str = Field(default="", max_length=500)


class CameraSpec(BaseModel):
    id: str = Field(min_length=1, max_length=64)
    source: str
    stride: int = Field(default=1, ge=1, le=100)


def create_app(system: LPRSystem, start_cameras: bool = True) -> FastAPI:
    api_key = system.cfg["api"].get("api_key")

    def check_key(request: Request):
        if api_key and not secrets.compare_digest(
                request.headers.get("x-api-key", ""), str(api_key)):
            raise HTTPException(401, "invalid or missing X-API-Key")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if start_cameras:
            system.start()
        yield
        system.stop()

    app = FastAPI(title="License Plate Recognition API", version="1.0",
                  lifespan=lifespan)
    auth = [Depends(check_key)]

    @app.get("/health")
    def health():
        cams = system.status()
        return {"status": "ok", "cameras": len(cams),
                "cameras_connected": sum(c["connected"] for c in cams)}

    @app.post("/recognize", dependencies=auth)
    async def recognize(image: UploadFile = File(...),
                        camera_id: str = Query("api-upload"),
                        save: bool = Query(False, description="store as events")):
        data = await image.read()
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, "image too large")
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(400, "could not decode image")
        readings = system.recognize_image(img)
        results = [r.as_dict() for r in readings]
        if save:
            for r, res in zip(readings, results):
                if not r.plate_text:
                    continue
                ev = system.record(
                    camera_id, r.plate_text, confidence=r.ocr.confidence,
                    detection_confidence=r.detection.confidence, raw_text=r.ocr.text,
                    format_valid=r.format_valid, verification=res.get("verification"),
                    frame=img, crop=r.crop)
                res["event_id"], res["alerts"] = ev["id"], ev["alerts"]
        else:
            for r, res in zip(readings, results):
                res["alerts"] = system.alerts.evaluate(
                    {"plate": r.plate_text, "verification": res.get("verification")})
        return {"plates": results, "count": len(results)}

    @app.get("/events", dependencies=auth)
    def events(plate: str | None = None, camera_id: str | None = None,
               since: str | None = Query(None, description="ISO-8601, e.g. 2026-01-31T00:00:00"),
               until: str | None = None, alerts_only: bool = False,
               limit: int = Query(100, ge=1, le=1000), offset: int = Query(0, ge=0)):
        return system.store.query_events(plate, camera_id, since, until,
                                         alerts_only, limit, offset)

    def _event(event_id: int) -> dict:
        ev = system.store.get_event(event_id)
        if ev is None:
            raise HTTPException(404, "event not found")
        return ev

    @app.get("/events/{event_id}", dependencies=auth)
    def event(event_id: int):
        return _event(event_id)

    @app.get("/events/{event_id}/{kind}", dependencies=auth)
    def event_image(event_id: int, kind: str):
        if kind not in ("frame", "crop"):
            raise HTTPException(404, "unknown image kind")
        path = _event(event_id).get(f"{kind}_path")
        if not path or not Path(path).exists():
            raise HTTPException(404, "image not stored")
        return FileResponse(path, media_type="image/jpeg")

    @app.get("/watchlist", dependencies=auth)
    def watchlist():
        return system.store.list_watchlist()

    @app.post("/watchlist", dependencies=auth, status_code=201)
    def add_watch(item: WatchlistItem):
        try:
            return {"plate": system.store.add_watchlist(item.plate, item.reason)}
        except ValueError as e:
            raise HTTPException(400, str(e))

    @app.delete("/watchlist/{plate}", dependencies=auth)
    def remove_watch(plate: str):
        if not system.store.remove_watchlist(plate):
            raise HTTPException(404, "plate not on watchlist")
        return {"removed": plate}

    @app.get("/cameras", dependencies=auth)
    def cameras():
        return system.status()

    @app.post("/cameras", dependencies=auth, status_code=201)
    def start_camera(spec: CameraSpec):
        try:
            return system.start_camera(spec.model_dump()).status()
        except ValueError as e:
            raise HTTPException(409, str(e))

    @app.delete("/cameras/{camera_id}", dependencies=auth)
    def stop_camera(camera_id: str):
        w = system.cameras.get(camera_id)
        if w is None:
            raise HTTPException(404, "unknown camera")
        w.stop()
        w.join(timeout=10)
        del system.cameras[camera_id]
        return {"stopped": camera_id}

    @app.get("/events-recent", dependencies=auth)
    def recent(limit: int = Query(20, ge=1, le=200)):
        return list(system.recent)[:limit]

    return app
