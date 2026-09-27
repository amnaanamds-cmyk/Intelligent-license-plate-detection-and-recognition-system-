"""Multi-modal vehicle-plate verification (Section 8).

This branch runs next to the plate pipeline:

    vehicle crop -> attribute classifier -> (colour, body type, make/model)
    plate string -> registration record  -> registered attributes
    comparator   -> MATCH / MISMATCH (with the attributes that differ)

A mismatch is a flag for an operator to review. It is not an automatic
violation, because the attribute classifiers can be wrong.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import cv2
import numpy as np

from .ocr import clean_plate_text

# COCO class ids that count as vehicles in the pretrained YOLOv11 model.
COCO_VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

ATTRIBUTES = ("color", "body_type", "make_model")


# --------------------------------------------------------------------------- #
# Attribute prediction
# --------------------------------------------------------------------------- #

@dataclass
class AttributePrediction:
    value: str
    confidence: float


@dataclass
class VehicleAttributes:
    color: AttributePrediction | None = None
    body_type: AttributePrediction | None = None
    make_model: AttributePrediction | None = None

    def as_dict(self) -> dict[str, dict | None]:
        return {a: (vars(getattr(self, a)) if getattr(self, a) else None)
                for a in ATTRIBUTES}


# Hue ranges in OpenCV units (0-179), checked in order.
_HUE_BANDS = (
    ("red", 0, 10), ("orange", 10, 22), ("yellow", 22, 35),
    ("green", 35, 85), ("blue", 85, 130), ("purple", 130, 160),
    ("red", 160, 180),
)


def classify_pixel_colors(hsv: np.ndarray, sat_thresh: int = 60,
                          dark_thresh: int = 50,
                          white_thresh: int = 190) -> np.ndarray:
    """Give every HSV pixel a colour name. Returns an array of strings."""
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    out = np.full(h.shape, "grey", dtype=object)
    achromatic = s < sat_thresh
    out[achromatic & (v >= white_thresh)] = "white"
    out[achromatic & (v < white_thresh) & (v >= 120)] = "silver"
    out[achromatic & (v < 120) & (v >= dark_thresh)] = "grey"
    out[v < dark_thresh] = "black"
    chromatic = (~achromatic) & (v >= dark_thresh)
    for name, lo, hi in _HUE_BANDS:
        out[chromatic & (h >= lo) & (h < hi)] = name
    # Dark, saturated orange is closer to brown.
    out[chromatic & (h >= 5) & (h < 22) & (v < 130)] = "brown"
    return out


class ColorClassifier:
    """Estimates body colour from the vehicle crop using HSV pixel votes.

    Only the central part of the crop is used, because the edges mostly
    contain road and background. The central band also skips most of the
    windscreen at the top and the tyres at the bottom. This needs no
    training, and a learned colour head can be used instead through
    ``LearnedAttributeClassifier``.
    """

    def __init__(self, center_fraction: tuple[float, float] = (0.6, 0.45),
                 vertical_offset: float = 0.1):
        self.center_fraction = center_fraction
        self.vertical_offset = vertical_offset

    def _body_region(self, crop: np.ndarray) -> np.ndarray:
        h, w = crop.shape[:2]
        fw, fh = self.center_fraction
        cx, cy = w / 2, h / 2 + self.vertical_offset * h
        x1, x2 = int(cx - fw * w / 2), int(cx + fw * w / 2)
        y1, y2 = int(cy - fh * h / 2), int(cy + fh * h / 2)
        region = crop[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
        return region if region.size else crop

    def predict(self, crop: np.ndarray) -> AttributePrediction:
        region = self._body_region(crop)
        region = cv2.resize(region, (64, 64), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        labels = classify_pixel_colors(hsv).ravel()
        names, counts = np.unique(labels, return_counts=True)
        best = int(np.argmax(counts))
        return AttributePrediction(str(names[best]), float(counts[best] / counts.sum()))


class LearnedAttributeClassifier:
    """An Ultralytics classification model (for example a fine-tuned
    ``yolo11n-cls.pt``) for one attribute such as body type or make/model
    group. Train it with ``scripts/train_attribute_classifier.py``."""

    def __init__(self, weights: str, device: str | None = None, imgsz: int = 224):
        from ultralytics import YOLO

        self.model = YOLO(weights)
        self.device = device
        self.imgsz = imgsz

    def predict(self, crop: np.ndarray) -> AttributePrediction:
        r = self.model.predict(crop, imgsz=self.imgsz, device=self.device,
                               verbose=False)[0]
        idx = int(r.probs.top1)
        return AttributePrediction(str(self.model.names[idx]), float(r.probs.top1conf))


class VehicleAttributeClassifier:
    """Combines one predictor per attribute. A missing predictor means that
    attribute is not predicted, and it is then skipped during comparison."""

    def __init__(self, color=None, body_type=None, make_model=None):
        self.color = color if color is not None else ColorClassifier()
        self.body_type = body_type
        self.make_model = make_model

    def predict(self, crop: np.ndarray) -> VehicleAttributes:
        return VehicleAttributes(
            color=self.color.predict(crop) if self.color else None,
            body_type=self.body_type.predict(crop) if self.body_type else None,
            make_model=self.make_model.predict(crop) if self.make_model else None,
        )


# --------------------------------------------------------------------------- #
# Registration records
# --------------------------------------------------------------------------- #

@dataclass
class RegistrationRecord:
    plate: str
    color: str | None = None
    body_type: str | None = None
    make_model: str | None = None


def _norm(value) -> str | None:
    if value is None:
        return None
    v = str(value).strip().lower().replace(" ", "_").replace("-", "_")
    return v or None


class VehicleRegistry:
    """Looks up registered attributes by plate number.

    In production this would query the licensing authority's database. Here
    it reads a JSON object (``{plate: {color, body_type, make_model}}``) or a
    CSV with the columns ``plate,color,body_type,make_model``. Plates are
    normalised with ``clean_plate_text``, so "ABC-1234" and "abc 1234" are
    the same key.
    """

    def __init__(self, records: dict[str, RegistrationRecord] | None = None):
        self._records: dict[str, RegistrationRecord] = {}
        for rec in (records or {}).values():
            self.add(rec)

    def add(self, record: RegistrationRecord) -> None:
        key = clean_plate_text(record.plate)
        self._records[key] = RegistrationRecord(
            plate=key, color=_norm(record.color),
            body_type=_norm(record.body_type), make_model=_norm(record.make_model),
        )

    def lookup(self, plate: str) -> RegistrationRecord | None:
        return self._records.get(clean_plate_text(plate))

    def __len__(self) -> int:
        return len(self._records)

    @classmethod
    def from_file(cls, path: str | Path) -> "VehicleRegistry":
        path = Path(path)
        reg = cls()
        if path.suffix.lower() == ".json":
            data = json.loads(path.read_text(encoding="utf-8"))
            for plate, attrs in data.items():
                reg.add(RegistrationRecord(plate=plate, **{
                    k: attrs.get(k) for k in ATTRIBUTES}))
        elif path.suffix.lower() == ".csv":
            with path.open(newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    reg.add(RegistrationRecord(plate=row["plate"], **{
                        k: row.get(k) for k in ATTRIBUTES}))
        else:
            raise ValueError(f"unsupported registry format: {path.suffix}")
        return reg


# --------------------------------------------------------------------------- #
# Comparison
# --------------------------------------------------------------------------- #

class VerificationStatus(str, Enum):
    MATCH = "match"
    MISMATCH = "mismatch"
    NOT_REGISTERED = "not_registered"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


# Colours that the classifier often mixes up under different lighting.
# Two colours in the same group are treated as consistent.
DEFAULT_COLOR_GROUPS = (
    frozenset({"white", "silver"}),
    frozenset({"silver", "grey"}),
    frozenset({"grey", "black"}),
    frozenset({"red", "orange"}),
    frozenset({"orange", "brown"}),
    frozenset({"brown", "red"}),
    frozenset({"blue", "purple"}),
)


@dataclass
class AttributeCheck:
    attribute: str
    registered: str | None
    predicted: str | None
    confidence: float
    consistent: bool | None  # None means not checked (no data / low confidence)


@dataclass
class VerificationResult:
    plate: str
    status: VerificationStatus
    checks: list[AttributeCheck] = field(default_factory=list)

    @property
    def mismatched_attributes(self) -> list[str]:
        return [c.attribute for c in self.checks if c.consistent is False]

    def summary(self) -> str:
        if self.status is VerificationStatus.MISMATCH:
            parts = [f"{c.attribute}: registered '{c.registered}', "
                     f"detected '{c.predicted}' ({c.confidence:.0%})"
                     for c in self.checks if c.consistent is False]
            return "MISMATCH - " + "; ".join(parts)
        if self.status is VerificationStatus.MATCH:
            ok = [c.attribute for c in self.checks if c.consistent]
            return "MATCH - consistent on " + ", ".join(ok)
        if self.status is VerificationStatus.NOT_REGISTERED:
            return "NOT REGISTERED - plate has no registration record"
        return "UNVERIFIED - not enough confident attribute predictions"


class AttributeComparator:
    """Rule-based comparison of predicted and registered attributes.

    An attribute is checked only when both sides have a value and the
    prediction confidence reaches that attribute's threshold. The result is
    MISMATCH when at least ``min_mismatches`` checked attributes disagree.
    """

    def __init__(self, thresholds: dict[str, float] | None = None,
                 color_groups=DEFAULT_COLOR_GROUPS, min_mismatches: int = 1):
        self.thresholds = {"color": 0.35, "body_type": 0.6, "make_model": 0.7}
        self.thresholds.update(thresholds or {})
        self.color_groups = color_groups
        self.min_mismatches = min_mismatches

    def _consistent(self, attribute: str, registered: str, predicted: str) -> bool:
        if registered == predicted:
            return True
        if attribute == "color":
            return any(registered in g and predicted in g for g in self.color_groups)
        return False

    def compare(self, plate: str, record: RegistrationRecord | None,
                predicted: VehicleAttributes) -> VerificationResult:
        plate = clean_plate_text(plate)
        if record is None:
            return VerificationResult(plate, VerificationStatus.NOT_REGISTERED)
        checks: list[AttributeCheck] = []
        for attr in ATTRIBUTES:
            reg_val = getattr(record, attr)
            pred = getattr(predicted, attr)
            pred_val = _norm(pred.value) if pred else None
            conf = pred.confidence if pred else 0.0
            consistent = None
            if reg_val and pred_val and conf >= self.thresholds.get(attr, 0.5):
                consistent = self._consistent(attr, reg_val, pred_val)
            checks.append(AttributeCheck(attr, reg_val, pred_val, conf, consistent))
        checked = [c for c in checks if c.consistent is not None]
        if not checked:
            status = VerificationStatus.INSUFFICIENT_EVIDENCE
        elif sum(c.consistent is False for c in checked) >= self.min_mismatches:
            status = VerificationStatus.MISMATCH
        else:
            status = VerificationStatus.MATCH
        return VerificationResult(plate, status, checks)


def associate_plate_to_vehicle(plate_box, vehicle_boxes) -> int | None:
    """Return the index of the vehicle box that contains most of the plate box.

    If the plate lies inside several vehicle boxes (overlapping cars), the
    smallest one wins, since that is usually the nearer vehicle.
    """
    px1, py1, px2, py2 = plate_box
    parea = max(1, (px2 - px1) * (py2 - py1))
    best, best_key = None, None
    for i, (vx1, vy1, vx2, vy2) in enumerate(vehicle_boxes):
        iw = max(0, min(px2, vx2) - max(px1, vx1))
        ih = max(0, min(py2, vy2) - max(py1, vy1))
        coverage = iw * ih / parea
        if coverage < 0.5:
            continue
        key = (round(coverage, 2), -(vx2 - vx1) * (vy2 - vy1))
        if best_key is None or key > best_key:
            best, best_key = i, key
    return best
