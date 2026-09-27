import cv2
import numpy as np

from lpr.geometry import deskew, estimate_skew_angle, rotate


def plate():
    img = np.full((80, 260, 3), 230, np.uint8)
    cv2.rectangle(img, (4, 4), (255, 75), (0, 0, 0), 3)
    cv2.putText(img, "ABC 1234", (18, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 0), 3)
    return img


def test_level_plate_untouched():
    out, angle = deskew(plate())
    assert angle == 0.0 and out.shape == plate().shape


def test_rotation_is_detected_and_corrected():
    for true in (8.0, -12.0):
        tilted = rotate(plate(), true)
        est = estimate_skew_angle(tilted)
        assert abs(est - true) < 2.0, (true, est)
        fixed, corrected = deskew(tilted)
        assert abs(estimate_skew_angle(fixed)) < 2.0
        assert abs(corrected - true) < 2.0


def test_tiny_or_blank_crop():
    assert estimate_skew_angle(np.zeros((5, 5, 3), np.uint8)) == 0.0
    assert estimate_skew_angle(np.full((60, 200, 3), 128, np.uint8)) == 0.0
