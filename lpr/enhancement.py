"""Pre-OCR image enhancement pipeline (Section 5.3).

A cropped plate goes through four stages in a fixed order:

1. CLAHE             - local contrast for uneven light, glare and shadow.
2. Bilateral filter  - removes noise but keeps character edges.
3. Normalisation     - stretches intensities to the full 0-255 range.
4. Unsharp masking   - sharpens strokes again after the denoising step.

The result is then resized to a fixed height so that PaddleOCR sees
characters at the same scale, however far away the camera was.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class EnhancementConfig:
    clahe_clip_limit: float = 2.0
    clahe_tile_grid: tuple[int, int] = (8, 8)
    bilateral_d: int = 9
    bilateral_sigma_color: float = 75.0
    bilateral_sigma_space: float = 75.0
    unsharp_sigma: float = 1.0
    unsharp_amount: float = 1.5
    # OCR input size. Only the height is fixed. The width follows the aspect
    # ratio and is capped at max_width, so multi-line plates are not squashed.
    target_height: int = 96
    max_width: int = 480
    # When True the output stays 3-channel BGR, which PaddleOCR expects.
    output_bgr: bool = True
    enabled_stages: tuple[str, ...] = field(
        default=("clahe", "bilateral", "normalize", "unsharp")
    )


def to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def apply_clahe(gray: np.ndarray, clip_limit: float = 2.0,
                tile_grid: tuple[int, int] = (8, 8)) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    return clahe.apply(gray)


def apply_bilateral(gray: np.ndarray, d: int = 9, sigma_color: float = 75.0,
                    sigma_space: float = 75.0) -> np.ndarray:
    return cv2.bilateralFilter(gray, d, sigma_color, sigma_space)


def normalize_intensity(gray: np.ndarray) -> np.ndarray:
    """Min-max stretch to [0, 255]. A flat image is returned unchanged."""
    lo, hi = int(gray.min()), int(gray.max())
    if hi <= lo:
        return gray.copy()
    return cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def unsharp_mask(gray: np.ndarray, sigma: float = 1.0,
                 amount: float = 1.5) -> np.ndarray:
    """sharpened = original + amount * (original - blurred)"""
    blurred = cv2.GaussianBlur(gray, (0, 0), sigma)
    return cv2.addWeighted(gray, 1.0 + amount, blurred, -amount, 0)


def resize_for_ocr(image: np.ndarray, target_height: int = 96,
                   max_width: int = 480) -> np.ndarray:
    h, w = image.shape[:2]
    if h == 0 or w == 0:
        raise ValueError("cannot resize an empty image")
    scale = target_height / h
    new_w = max(1, min(max_width, int(round(w * scale))))
    interp = cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA
    return cv2.resize(image, (new_w, target_height), interpolation=interp)


class PlateEnhancer:
    """Runs the four enhancement stages and the final resize.

    ``enhance`` returns the final image. ``enhance_with_stages`` also returns
    each intermediate step, which the GUI uses to show the stages next to
    each other.
    """

    STAGE_ORDER = ("clahe", "bilateral", "normalize", "unsharp")

    def __init__(self, config: EnhancementConfig | None = None):
        self.config = config or EnhancementConfig()
        unknown = set(self.config.enabled_stages) - set(self.STAGE_ORDER)
        if unknown:
            raise ValueError(f"unknown enhancement stages: {sorted(unknown)}")

    def _run_stage(self, name: str, img: np.ndarray) -> np.ndarray:
        c = self.config
        if name == "clahe":
            return apply_clahe(img, c.clahe_clip_limit, c.clahe_tile_grid)
        if name == "bilateral":
            return apply_bilateral(img, c.bilateral_d, c.bilateral_sigma_color,
                                   c.bilateral_sigma_space)
        if name == "normalize":
            return normalize_intensity(img)
        if name == "unsharp":
            return unsharp_mask(img, c.unsharp_sigma, c.unsharp_amount)
        raise ValueError(name)

    def enhance_with_stages(self, crop: np.ndarray) -> dict[str, np.ndarray]:
        if crop is None or crop.size == 0:
            raise ValueError("empty plate crop")
        stages: dict[str, np.ndarray] = {"input": crop}
        img = to_gray(crop)
        stages["gray"] = img
        for name in self.STAGE_ORDER:
            if name in self.config.enabled_stages:
                img = self._run_stage(name, img)
                stages[name] = img
        img = resize_for_ocr(img, self.config.target_height, self.config.max_width)
        if self.config.output_bgr:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        stages["output"] = img
        return stages

    def enhance(self, crop: np.ndarray) -> np.ndarray:
        return self.enhance_with_stages(crop)["output"]
