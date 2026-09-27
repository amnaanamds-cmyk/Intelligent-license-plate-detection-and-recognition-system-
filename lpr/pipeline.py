"""End-to-end pipeline: detect -> crop -> enhance -> recognise [-> verify]."""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .detector import Detection, YOLODetector, crop_box
from .enhancement import EnhancementConfig, PlateEnhancer
from .ocr import OCRResult, PlateOCR, clean_plate_text
from .verification import (
    COCO_VEHICLE_CLASSES,
    AttributeComparator,
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
    vehicle: Detection | None = None
    vehicle_attributes: VehicleAttributes | None = None
    verification: VerificationResult | None = None

    @property
    def plate_text(self) -> str:
        return clean_plate_text(self.ocr.text)

    def as_dict(self) -> dict:
        d = {
            "plate": self.plate_text,
            "raw_text": self.ocr.text,
            "ocr_confidence": round(self.ocr.confidence, 4),
            "detection_confidence": round(self.detection.confidence, 4),
            "box": list(self.detection.box),
        }
        if self.vehicle is not None:
            d["vehicle_box"] = list(self.vehicle.box)
            d["vehicle_class"] = self.vehicle.label
        if self.vehicle_attributes is not None:
            d["vehicle_attributes"] = self.vehicle_attributes.as_dict()
        if self.verification is not None:
            d["verification"] = {
                "status": self.verification.status.value,
                "mismatched": self.verification.mismatched_attributes,
                "summary": self.verification.summary(),
            }
        return d


class LicensePlatePipeline:
    def __init__(self, plate_detector: YOLODetector, ocr: PlateOCR,
                 enhancer: PlateEnhancer | None = None, use_enhancement: bool = True,
                 crop_padding: float = 0.05, max_plates: int = 10,
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
        self.vehicle_detector = vehicle_detector
        self.attribute_classifier = attribute_classifier
        self.registry = registry
        self.comparator = comparator or AttributeComparator()

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
        from .verification import LearnedAttributeClassifier

        vehicle_detector = attr_clf = registry = None
        if vehicle_weights and registry_path:
            vehicle_detector = YOLODetector(vehicle_weights, conf=0.3, device=device,
                                            classes=list(COCO_VEHICLE_CLASSES))
            attr_clf = VehicleAttributeClassifier(
                body_type=LearnedAttributeClassifier(body_type_weights, device)
                if body_type_weights else None,
                make_model=LearnedAttributeClassifier(make_model_weights, device)
                if make_model_weights else None,
            )
            registry = VehicleRegistry.from_file(registry_path)
        return cls(
            plate_detector=YOLODetector(plate_weights, conf=conf, device=device),
            ocr=PlateOCR(use_gpu=use_gpu_ocr),
            enhancer=PlateEnhancer(enhancement),
            use_enhancement=use_enhancement,
            vehicle_detector=vehicle_detector,
            attribute_classifier=attr_clf,
            registry=registry,
        )

    @property
    def verification_enabled(self) -> bool:
        return (self.vehicle_detector is not None
                and self.attribute_classifier is not None
                and self.registry is not None)

    def process(self, image: np.ndarray) -> list[PlateReading]:
        if image is None or image.size == 0:
            raise ValueError("empty input image")
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        detections = self.plate_detector.detect(image)[: self.max_plates]
        vehicles = self.vehicle_detector.detect(image) if self.verification_enabled else []

        readings: list[PlateReading] = []
        for det in detections:
            crop = crop_box(image, det.box, self.crop_padding)
            if crop.size == 0:
                continue
            if self.use_enhancement:
                stages = self.enhancer.enhance_with_stages(crop)
                enhanced = stages["output"]
            else:
                stages = {"input": crop}
                enhanced = crop
            reading = PlateReading(det, crop, enhanced, self.ocr.recognize(enhanced), stages)
            if self.verification_enabled and reading.plate_text:
                self._verify(image, reading, vehicles)
            readings.append(reading)
        return readings

    def _verify(self, image: np.ndarray, reading: PlateReading,
                vehicles: list[Detection]) -> None:
        idx = associate_plate_to_vehicle(reading.detection.box, [v.box for v in vehicles])
        record = self.registry.lookup(reading.plate_text)
        if idx is None:
            # No vehicle box found, so the plate can only be checked for
            # registration. There are no attributes to compare.
            reading.vehicle_attributes = VehicleAttributes()
        else:
            reading.vehicle = vehicles[idx]
            vcrop = crop_box(image, reading.vehicle.box)
            reading.vehicle_attributes = self.attribute_classifier.predict(vcrop)
        reading.verification = self.comparator.compare(
            reading.plate_text, record, reading.vehicle_attributes)


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
