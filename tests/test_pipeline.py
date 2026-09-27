"""Pipeline wiring tested with stub models (no YOLO / PaddleOCR required)."""
import numpy as np

from lpr.detector import Detection, crop_box
from lpr.ocr import OCRResult
from lpr.pipeline import LicensePlatePipeline, draw_readings
from lpr.verification import (
    AttributePrediction, RegistrationRecord, VehicleAttributes, VehicleRegistry,
)


class StubDetector:
    def __init__(self, dets):
        self.dets = dets

    def detect(self, image):
        return list(self.dets)


class StubOCR:
    def __init__(self, text="ab-c 123"):
        self.text = text
        self.seen = []

    def recognize(self, image):
        self.seen.append(image)
        return OCRResult(self.text, 0.9)


class StubAttrs:
    def __init__(self, color, body):
        self.color, self.body = color, body

    def predict(self, crop):
        return VehicleAttributes(AttributePrediction(self.color, 0.9),
                                 AttributePrediction(self.body, 0.9))


IMG = np.random.default_rng(0).integers(0, 255, (480, 640, 3), dtype=np.uint8)
PLATE = Detection((200, 300, 360, 350), 0.95)


def test_crop_box_padding_is_clipped():
    assert crop_box(IMG, (0, 0, 100, 50), pad_ratio=0.1).shape == (55, 110, 3)


def test_process_enhanced_vs_raw():
    ocr = StubOCR()
    pipe = LicensePlatePipeline(StubDetector([PLATE]), ocr)
    [r] = pipe.process(IMG)
    assert r.plate_text == "ABC123"
    assert ocr.seen[-1].shape[0] == 96 and "clahe" in r.stages
    pipe.use_enhancement = False
    [r] = pipe.process(IMG)
    assert r.enhanced is r.crop and "clahe" not in r.stages
    assert r.verification is None


def test_verification_branch_flags_mismatch():
    reg = VehicleRegistry({"x": RegistrationRecord("ABC123", color="white", body_type="sedan")})
    pipe = LicensePlatePipeline(
        StubDetector([PLATE]), StubOCR(),
        vehicle_detector=StubDetector([Detection((100, 100, 500, 450), 0.9, 2, "car")]),
        attribute_classifier=StubAttrs("red", "suv"), registry=reg)
    [r] = pipe.process(IMG)
    assert r.vehicle.label == "car"
    assert r.verification.status.value == "mismatch"
    d = r.as_dict()
    assert d["verification"]["mismatched"] == ["color", "body_type"]
    assert draw_readings(IMG, [r]).shape == IMG.shape


def test_verification_without_vehicle_box():
    reg = VehicleRegistry({"x": RegistrationRecord("ABC123", color="white")})
    pipe = LicensePlatePipeline(
        StubDetector([PLATE]), StubOCR(), vehicle_detector=StubDetector([]),
        attribute_classifier=StubAttrs("red", "suv"), registry=reg)
    [r] = pipe.process(IMG)
    assert r.vehicle is None
    assert r.verification.status.value == "insufficient_evidence"
