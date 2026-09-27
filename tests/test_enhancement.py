import cv2
import numpy as np
import pytest

from lpr.enhancement import (
    EnhancementConfig, PlateEnhancer, apply_clahe, normalize_intensity,
    resize_for_ocr, unsharp_mask,
)


def synthetic_plate(dark=True):
    """A low-contrast, noisy plate with text on it."""
    img = np.full((40, 160, 3), 70 if dark else 200, np.uint8)
    cv2.putText(img, "ABC123", (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                (95, 95, 95) if dark else (170, 170, 170), 2)
    rng = np.random.default_rng(0)
    noise = rng.normal(0, 6, img.shape)
    return np.clip(img + noise, 0, 255).astype(np.uint8)


def test_normalize_stretches_to_full_range():
    g = np.array([[50, 100], [120, 150]], np.uint8)
    out = normalize_intensity(g)
    assert out.min() == 0 and out.max() == 255


def test_normalize_flat_image_unchanged():
    g = np.full((5, 5), 42, np.uint8)
    assert np.array_equal(normalize_intensity(g), g)


def test_clahe_increases_contrast_on_dark_plate():
    gray = cv2.cvtColor(synthetic_plate(), cv2.COLOR_BGR2GRAY)
    assert apply_clahe(gray).std() > gray.std()


def test_unsharp_mask_increases_edge_energy():
    gray = cv2.GaussianBlur(cv2.cvtColor(synthetic_plate(), cv2.COLOR_BGR2GRAY), (5, 5), 0)
    lap = lambda x: cv2.Laplacian(x, cv2.CV_64F).var()
    assert lap(unsharp_mask(gray)) > lap(gray)


def test_resize_keeps_aspect_and_caps_width():
    img = np.zeros((20, 100), np.uint8)
    assert resize_for_ocr(img, 96, 1000).shape == (96, 480)
    assert resize_for_ocr(img, 96, 300).shape == (96, 300)


def test_pipeline_output_shape_and_stages():
    stages = PlateEnhancer().enhance_with_stages(synthetic_plate())
    assert list(stages) == ["input", "gray", "clahe", "bilateral", "normalize", "unsharp", "output"]
    out = stages["output"]
    assert out.shape[0] == 96 and out.ndim == 3 and out.dtype == np.uint8


def test_pipeline_improves_text_background_contrast():
    plate = synthetic_plate()
    out = cv2.cvtColor(PlateEnhancer().enhance(plate), cv2.COLOR_BGR2GRAY)
    before = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
    assert (out.max() - out.min()) > (before.max() - before.min())
    assert out.std() > before.std()


def test_stage_selection_and_validation():
    enh = PlateEnhancer(EnhancementConfig(enabled_stages=("clahe",), output_bgr=False))
    stages = enh.enhance_with_stages(synthetic_plate())
    assert "bilateral" not in stages and stages["output"].ndim == 2
    with pytest.raises(ValueError):
        PlateEnhancer(EnhancementConfig(enabled_stages=("sharpen",)))
    with pytest.raises(ValueError):
        PlateEnhancer().enhance(np.zeros((0, 0, 3), np.uint8))
