"""Rotation correction for tilted plate crops.

The tilt angle comes from the long, near-horizontal edges in the crop:
the plate border and the tops and bottoms of the characters. The crop is
then rotated to level it. This handles in-plane rotation from camera roll
or a plate mounted crooked. It does not fix strong perspective distortion,
which is better avoided by camera placement (docs/DEPLOYMENT.md).
"""

from __future__ import annotations

import cv2
import numpy as np


def estimate_skew_angle(crop: np.ndarray, max_angle: float = 30.0) -> float:
    """Return the tilt of the crop in degrees (positive = counter-clockwise),
    or 0.0 when no dominant horizontal edge is found."""
    gray = crop if crop.ndim == 2 else cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    if h < 10 or w < 20:
        return 0.0
    edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 360, threshold=max(15, w // 6),
                            minLineLength=w // 3, maxLineGap=max(3, w // 30))
    if lines is None:
        return 0.0
    angles, weights = [], []
    for x1, y1, x2, y2 in lines[:, 0]:
        dx, dy = x2 - x1, y2 - y1
        if dx == 0:
            continue
        a = np.degrees(np.arctan2(-dy, dx))  # image y axis points down
        if abs(a) <= max_angle:
            angles.append(a)
            weights.append(np.hypot(dx, dy))
    if not angles:
        return 0.0
    # Length-weighted median, so a few short spurious lines have no effect.
    order = np.argsort(angles)
    a, wts = np.asarray(angles)[order], np.asarray(weights)[order]
    return float(a[np.searchsorted(np.cumsum(wts), wts.sum() / 2)])


def rotate(image: np.ndarray, angle: float) -> np.ndarray:
    """Rotate counter-clockwise by ``angle`` degrees, enlarging the canvas."""
    h, w = image.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    cos, sin = abs(m[0, 0]), abs(m[0, 1])
    nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
    m[0, 2] += nw / 2 - w / 2
    m[1, 2] += nh / 2 - h / 2
    return cv2.warpAffine(image, m, (nw, nh), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)


def deskew(crop: np.ndarray, min_angle: float = 1.5,
           max_angle: float = 30.0) -> tuple[np.ndarray, float]:
    """Level the crop. Returns (image, angle corrected in degrees)."""
    angle = estimate_skew_angle(crop, max_angle)
    if abs(angle) < min_angle:
        return crop, 0.0
    return rotate(crop, -angle), angle
