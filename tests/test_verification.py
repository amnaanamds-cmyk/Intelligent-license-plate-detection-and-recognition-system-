import json

import cv2
import numpy as np

from lpr.verification import (
    AttributeComparator, AttributePrediction, ColorClassifier, RegistrationRecord,
    VehicleAttributes, VehicleRegistry, VerificationStatus, associate_plate_to_vehicle,
)


def solid(bgr, size=(120, 200)):
    img = np.zeros((*size, 3), np.uint8)
    img[:] = bgr
    return img


def test_color_classifier_basic_colours():
    clf = ColorClassifier()
    assert clf.predict(solid((0, 0, 200))).value == "red"
    assert clf.predict(solid((200, 60, 0))).value == "blue"
    assert clf.predict(solid((245, 245, 245))).value == "white"
    assert clf.predict(solid((15, 15, 15))).value == "black"
    assert clf.predict(solid((150, 150, 150))).value == "silver"


def test_color_classifier_ignores_background_border():
    img = solid((0, 180, 0), (200, 300))       # green background
    img[40:180, 50:250] = (0, 0, 200)          # red car body in the middle
    assert ColorClassifier().predict(img).value == "red"


def test_registry_normalises_plate_and_values(tmp_path):
    p = tmp_path / "reg.json"
    p.write_text(json.dumps({"abc-1234": {"color": "White", "body_type": "Sedan"}}))
    reg = VehicleRegistry.from_file(p)
    rec = reg.lookup("ABC 1234")
    assert rec is not None and rec.color == "white" and rec.body_type == "sedan"
    assert reg.lookup("ZZZ999") is None


def test_registry_csv(tmp_path):
    p = tmp_path / "reg.csv"
    p.write_text("plate,color,body_type,make_model\nPES5678,black,SUV,Toyota Fortuner\n")
    rec = VehicleRegistry.from_file(p).lookup("pes5678")
    assert rec.make_model == "toyota_fortuner" and rec.body_type == "suv"


def attrs(color=None, body=None, cconf=0.9, bconf=0.9):
    return VehicleAttributes(
        color=AttributePrediction(color, cconf) if color else None,
        body_type=AttributePrediction(body, bconf) if body else None,
    )


REC = RegistrationRecord("ABC1234", color="white", body_type="sedan")


def test_match():
    r = AttributeComparator().compare("ABC1234", REC, attrs("white", "sedan"))
    assert r.status is VerificationStatus.MATCH


def test_mismatch_reports_attributes():
    # Paper example: registered white sedan vs detected red SUV
    r = AttributeComparator().compare("ABC1234", REC, attrs("red", "suv"))
    assert r.status is VerificationStatus.MISMATCH
    assert r.mismatched_attributes == ["color", "body_type"]
    assert "registered 'white'" in r.summary()


def test_similar_colours_are_tolerated():
    r = AttributeComparator().compare("ABC1234", REC, attrs("silver", "sedan"))
    assert r.status is VerificationStatus.MATCH


def test_low_confidence_predictions_are_not_used():
    r = AttributeComparator().compare("ABC1234", REC, attrs("red", "suv", 0.1, 0.2))
    assert r.status is VerificationStatus.INSUFFICIENT_EVIDENCE


def test_not_registered():
    r = AttributeComparator().compare("XYZ", None, attrs("red"))
    assert r.status is VerificationStatus.NOT_REGISTERED


def test_min_mismatches():
    comp = AttributeComparator(min_mismatches=2)
    assert comp.compare("ABC1234", REC, attrs("red", "sedan")).status is VerificationStatus.MATCH
    assert comp.compare("ABC1234", REC, attrs("red", "suv")).status is VerificationStatus.MISMATCH


def test_associate_plate_to_vehicle():
    vehicles = [(0, 0, 1000, 800), (400, 300, 700, 600), (800, 0, 1000, 200)]
    assert associate_plate_to_vehicle((500, 500, 600, 540), vehicles) == 1
    assert associate_plate_to_vehicle((50, 700, 150, 740), vehicles) == 0
    assert associate_plate_to_vehicle((2000, 2000, 2100, 2040), vehicles) is None
