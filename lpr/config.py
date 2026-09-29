"""System configuration: YAML file merged over defaults."""

from __future__ import annotations

import copy
from pathlib import Path

DEFAULTS: dict = {
    "detector": {"weights": "weights/plate_yolo11.pt", "conf": 0.35, "iou": 0.7,
                 "imgsz": 640, "device": None, "min_plate_width": 60},
    "ocr": {"lang": "en", "use_gpu": False, "min_line_confidence": 0.5,
            "det_model_dir": None, "rec_model_dir": None, "rec_model_name": None},
    "enhancement": {"enabled": True, "deskew": True},
    "plate_format": {"enabled": True, "templates": [], "ignore_words": [],
                     "max_substitutions": 2, "reject_invalid": False},
    "tracking": {"tracker": "bytetrack.yaml", "min_reads": 3, "agreement": 0.6,
                 "max_reads": 12, "lost_frames": 30},
    "verification": {"enabled": False, "vehicle_weights": "yolo11n.pt",
                     "registry": None, "body_type_weights": None,
                     "make_model_weights": None, "min_mismatches": 1},
    "storage": {"database": "data/lpr.db", "snapshot_dir": "data/snapshots",
                "save_snapshots": True, "retention_days": 90},
    "alerts": {"on_watchlist": True, "on_mismatch": True,
               "watchlist_max_distance": 1, "webhook_url": None,
               "webhook_timeout": 5},
    "api": {"host": "0.0.0.0", "port": 8000, "api_key": None, "session_hours": 12},
    "cameras": [],
}


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> dict:
    data = {}
    if path:
        import yaml

        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return deep_merge(deep_merge(DEFAULTS, data), overrides or {})
