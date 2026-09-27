"""End-to-end pipeline: detect -> crop -> [deskew] -> enhance -> recognise
-> format check [-> verify]."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

import cv2
import numpy as np

from .detector import Detection, YOLODetector, crop_box
from .enhancement import EnhancementConfig, PlateEnhancer
from .geometry import deskew
from .ocr import OCRResult, PlateOCR, clean_plate_text
from .postprocess import FormattedPlate, PlateFormatter
from .verification import (
    COCO_VEHICLE_CLASSES,
    AttributeComparator,
    LearnedAttributeClassifier,
    VehicleAttributeClassifier,
    VehicleAttributes,
    VehicleRegistry,
    VerificationResult,
    associate_plate_to_vehicle,
)


@dataclass
class PlateReading:
    detection: Detection
    crop: np.ndarray
    enhanced: np.ndarray
    ocr: OCRResult
    stages: dict[str, np.ndarray] = field(default_factory=dict)
    formatted: FormattedPlate | None = None
    skew_angle: float = 0.0
    vehicle: Detection | None = None
    vehicle_attributes: VehicleAttributes | None = None
    verification: VerificationResult | None = None

    @property
    def plate_text(self) -> str:
        if self.formatted is not None:
            return self.formatted.text
        return clean_plate_text(self.ocr.text)

    @property
    def format_valid(self) -> bool | None:
        return None if self.formatted is None else self.formatted.valid

    def as_dict(self) -> dict:
        d = {
            "plate": self.plate_text,
            "raw_text": self.ocr.text,
            "ocr_confidence": round(self.ocr.confidence, 4),
            "detection_confidence": round(self.detection.confidence, 4),
            "box": list(self.detection.box),
        }
        if self.formatted is not None:
            d["format_valid"] = self.formatted.valid
            d["format_template"] = self.formatted.template
            d["corrections"] = self.formatted.substitutions
        if self.detection.track_id is not None:
            d["track_id"] = self.detection.track_id
        if self.vehicle is not None:
            d["vehicle_box"] = list(self.vehicle.box)
            d["vehicle_class"] = self.vehicle.label
        if self.vehicle_attributes is not None:
            d["vehicle_attributes"] = self.vehicle_attributes.as_dict()
        if self.verification is not None:
            d["verification"] = verification_dict(self.verification)
        return d


def verification_dict(v: VerificationResult) -> dict:
    return {"status": v.status.value, "mismatched": v.mismatched_attributes,
            "summary": v.summary()}


class LicensePlatePipeline:
    def __init__(self, plate_detector: YOLODetector, ocr: PlateOCR,
                 enhancer: PlateEnhancer | None = None, use_enhancement: bool = True,
                 crop_padding: float = 0.05, max_plates: int = 10,
                 use_deskew: bool = False, formatter: PlateFormatter | None = None,
                 reject_invalid: bool = False, min_plate_width: int = 0,
                 vehicle_detector: YOLODetector | None = None,
                 attribute_classifier: VehicleAttributeClassifier | None = None,
                 registry: VehicleRegistry | None = None,
                 comparator: AttributeComparator | None = None):
        self.plate_detector = plate_detector
        self.ocr = ocr
        self.enhancer = enhancer or PlateEnhancer()
        self.use_enhancement = use_enhancement
        self.crop_padding = crop_padding
        self.max_plates = max_plates
        self.use_deskew = use_deskew
        self.formatter = formatter
        self.reject_invalid = reject_invalid
        self.min_plate_width = min_plate_width
        self.vehicle_detector = vehicle_detector
        self.attribute_classifier = attribute_classifier
        self.registry = registry
        self.comparator = comparator or AttributeComparator()
        # PaddleOCR and the vehicle models may be shared by several camera
        # threads. They are not documented as thread-safe, so calls are serialised.
        self._ocr_lock = threading.Lock()
        self._verify_lock = threading.Lock()

    # ----------------------------------------------------------- builders --
    @classmethod
    def from_weights(cls, plate_weights: str, conf: float = 0.25,
                     device: str | None = None, use_gpu_ocr: bool = False,
                     enhancement: EnhancementConfig | None = None,
                     use_enhancement: bool = True,
                     vehicle_weights: str | None = None,
                     registry_path: str | None = None,
                     body_type_weights: str | None = None,
                     make_model_weights: str | None = None) -> "LicensePlatePipeline":
        """Build the pipeline from file paths.

        Verification is turned on only when ``vehicle_weights`` and
        ``registry_path`` are both given.
        """
        vehicle_detector, attr_clf, registry = _build_verification(
            vehicle_weights, registry_path, body_type_weights, make_model_weights, device)
        return cls(
            plate_detector=YOLODetector(plate_weights, conf=conf, device=device),
            ocr=PlateOCR(use_gpu=use_gpu_ocr),
            enhancer=PlateEnhancer(enhancement),
            use_enhancement=use_enhancement,
            vehicle_detector=vehicle_detector,
            attribute_classifier=attr_clf,
            registry=registry,
        )

    @classmethod
    def from_config(cls, cfg: dict, ocr: PlateOCR | None = None,
                    plate_detector: YOLODetector | None = None) -> "LicensePlatePipeline":
        """Build from a config dict (see ``lpr.config``). Pass ``ocr`` to share
        one OCR engine between several pipelines, for example one per camera."""
        d, e = cfg["detector"], dict(cfg["enhancement"])
        pf, v = cfg["plate_format"], cfg["verification"]
        use_enh, use_deskew = e.pop("enabled", True), e.pop("deskew", True)
        enh_fields = EnhancementConfig.__dataclass_fields__
        enh_cfg = EnhancementConfig(**{k: (tuple(val) if isinstance(val, list) else val)
                                       for k, val in e.items() if k in enh_fields})
        vehicle_detector = attr_clf = registry = None
        if v.get("enabled"):
            vehicle_detector, attr_clf, registry = _build_verification(
                v["vehicle_weights"], v["registry"], v.get("body_type_weights"),
                v.get("make_model_weights"), d.get("device"))
        return cls(
            plate_detector=plate_detector or YOLODetector(
                d["weights"], conf=d["conf"], iou=d["iou"], imgsz=d["imgsz"],
                device=d.get("device")),
            ocr=ocr or build_ocr(cfg),
            enhancer=PlateEnhancer(enh_cfg),
            use_enhancement=use_enh,
            use_deskew=use_deskew,
            formatter=PlateFormatter.from_config(pf) if pf.get("enabled") else None,
            reject_invalid=bool(pf.get("reject_invalid")),
            min_plate_width=int(d.get("min_plate_width") or 0),
            vehicle_detector=vehicle_detector,
            attribute_classifier=attr_clf,
            registry=registry,
            comparator=AttributeComparator(min_mismatches=int(v.get("min_mismatches", 1))),
        )

    @property
    def verification_enabled(self) -> bool:
        return (self.vehicle_detector is not None
                and self.attribute_classifier is not None
                and self.registry is not None)

    # ---------------------------------------------------------- inference --
    def read_plate(self, image: np.ndarray, det: Detection) -> PlateReading | None:
        """Run crop -> deskew -> enhance -> OCR -> format on one detection."""
        crop = crop_box(image, det.box, self.crop_padding)
        if crop.size == 0:
            return None
        angle = 0.0
        if self.use_deskew:
            crop, angle = deskew(crop)
        if self.use_enhancement:
            stages = self.enhancer.enhance_with_stages(crop)
            enhanced = stages["output"]
        else:
            stages = {"input": crop}
            enhanced = crop
        with self._ocr_lock:
            ocr = self.ocr.recognize(enhanced)
        formatted = self.formatter.format(ocr.text) if self.formatter else None
        return PlateReading(det, crop, enhanced, ocr, stages, formatted, angle)

    def readable(self, det: Detection) -> bool:
        x1, _, x2, _ = det.box
        return (x2 - x1) >= self.min_plate_width

    def process(self, image: np.ndarray,
                detections: list[Detection] | None = None) -> list[PlateReading]:
        if image is None or image.size == 0:
            raise ValueError("empty input image")
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if detections is None:
            detections = self.plate_detector.detect(image)
        detections = detections[: self.max_plates]
        vehicles = (self.detect_vehicles(image)
                    if self.verification_enabled and detections else [])

        readings: list[PlateReading] = []
        for det in detections:
            if not self.readable(det):
                continue
            reading = self.read_plate(image, det)
            if reading is None:
                continue
            if self.reject_invalid and reading.format_valid is False:
                continue
            if self.verification_enabled and reading.plate_text:
                (reading.vehicle, reading.vehicle_attributes,
                 reading.verification) = self.verify(image, det.box,
                                                     reading.plate_text, vehicles)
            readings.append(reading)
        return readings

    def detect_vehicles(self, image: np.ndarray) -> list[Detection]:
        with self._verify_lock:
            return self.vehicle_detector.detect(image)

    def verify(self, image: np.ndarray, plate_box, plate_text: str,
               vehicles: list[Detection] | None = None):
        """Returns (vehicle detection or None, predicted attributes, result)."""
        if vehicles is None:
            vehicles = self.detect_vehicles(image)
        idx = associate_plate_to_vehicle(plate_box, [v.box for v in vehicles])
        record = self.registry.lookup(plate_text)
        vehicle = None
        if idx is None:
            # No vehicle box found, so the plate can only be checked for
            # registration. There are no attributes to compare.
            attrs = VehicleAttributes()
        else:
            vehicle = vehicles[idx]
            with self._verify_lock:
                attrs = self.attribute_classifier.predict(crop_box(image, vehicle.box))
        return vehicle, attrs, self.comparator.compare(plate_text, record, attrs)


def build_ocr(cfg: dict) -> PlateOCR:
    o = cfg["ocr"]
    return PlateOCR(lang=o.get("lang", "en"), use_gpu=bool(o.get("use_gpu")),
                    min_line_confidence=float(o.get("min_line_confidence", 0.3)),
                    det_model_dir=o.get("det_model_dir"),
                    rec_model_dir=o.get("rec_model_dir"),
                    rec_model_name=o.get("rec_model_name"))


def _build_verification(vehicle_weights, registry_path, body_type_weights,
                        make_model_weights, device):
    if not (vehicle_weights and registry_path):
        return None, None, None
    vehicle_detector = YOLODetector(vehicle_weights, conf=0.3, device=device,
                                    classes=list(COCO_VEHICLE_CLASSES))
    attr_clf = VehicleAttributeClassifier(
        body_type=LearnedAttributeClassifier(body_type_weights, device)
        if body_type_weights else None,
        make_model=LearnedAttributeClassifier(make_model_weights, device)
        if make_model_weights else None,
    )
    return vehicle_detector, attr_clf, VehicleRegistry.from_file(registry_path)


def draw_readings(image: np.ndarray, readings: list[PlateReading]) -> np.ndarray:
    """Draw plate boxes (green, or red on a mismatch) and vehicle boxes."""
    out = image.copy()
    for r in readings:
        mismatch = (r.verification is not None
                    and r.verification.status.value == "mismatch")
        color = (0, 0, 255) if mismatch else (0, 200, 0)
        if r.vehicle is not None:
            vx1, vy1, vx2, vy2 = r.vehicle.box
            cv2.rectangle(out, (vx1, vy1), (vx2, vy2), (255, 160, 0), 2)
        x1, y1, x2, y2 = r.detection.box
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"{r.plate_text or '?'} {r.ocr.confidence:.2f}"
        if mismatch:
            label += " !MISMATCH"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        ty = max(th + 4, y1 - 4)
        cv2.rectangle(out, (x1, ty - th - 4), (x1 + tw + 4, ty + 2), color, -1)
        cv2.putText(out, label, (x1 + 2, ty - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2, cv2.LINE_AA)
    return out
