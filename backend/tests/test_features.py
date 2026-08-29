"""
test_features.py
================
Unit tests for the feature extractor. Each test creates a controlled synthetic
image and verifies that the extracted features fall in the expected range.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import cv2
import pytest
from app.services.feature_extractor import extract_features, FeatureVector


def _solid(h=256, w=256, value=128) -> np.ndarray:
    """Solid gray image."""
    return np.full((h, w, 3), value, dtype=np.uint8)


def _gradient(h=256, w=256) -> np.ndarray:
    """Horizontal gradient from 0 to 255."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    for c in range(w):
        img[:, c] = int(c / w * 255)
    return img


def _checkerboard(h=256, w=256, sq=16) -> np.ndarray:
    """Black-and-white checkerboard — high Laplacian variance."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    for r in range(0, h, sq):
        for c in range(0, w, sq):
            if (r // sq + c // sq) % 2 == 0:
                img[r:r+sq, c:c+sq] = 255
    return img


def _noisy(h=256, w=256, sigma=40) -> np.ndarray:
    base = _gradient(h, w).astype(np.float32)
    noise = np.random.normal(0, sigma, base.shape)
    return np.clip(base + noise, 0, 255).astype(np.uint8)


def _blurred(h=256, w=256, sigma=8) -> np.ndarray:
    img = _gradient(h, w)
    return cv2.GaussianBlur(img, (31, 31), sigma)


def _dark(h=256, w=256) -> np.ndarray:
    return np.full((h, w, 3), 10, dtype=np.uint8)


def _bright(h=256, w=256) -> np.ndarray:
    return np.full((h, w, 3), 250, dtype=np.uint8)


# ── Tests ──────────────────────────────────────────────────────────────────────

def test_returns_feature_vector():
    fv = extract_features(_gradient())
    assert isinstance(fv, FeatureVector)


def test_feature_vector_has_correct_model_array_length():
    fv = extract_features(_gradient())
    arr = fv.to_model_array()
    assert arr.shape == (22,)


def test_sharpness_high_for_sharp_image():
    # Checkerboard has many sharp edges → high Laplacian variance
    sharp = _checkerboard()
    fv = extract_features(sharp)
    assert fv.laplacian_variance > 100, (
        f"Checkerboard should have high Laplacian variance, got {fv.laplacian_variance:.2f}"
    )


def test_sharpness_low_for_blurred_image():
    sharp   = _checkerboard()  # sharp edges
    blurred = cv2.GaussianBlur(sharp, (31, 31), 10)  # blur the checkerboard
    fv_b = extract_features(blurred)
    fv_s = extract_features(sharp)
    assert fv_b.laplacian_variance < fv_s.laplacian_variance, (
        "Blurred checkerboard should have lower Laplacian variance than sharp"
    )


def test_underexposure_detected_for_dark_image():
    dark = _dark()
    fv = extract_features(dark)
    assert fv.underexposed_ratio > 0.90, "Nearly all pixels should be underexposed"
    assert fv.brightness_mean < 30


def test_overexposure_detected_for_bright_image():
    bright = _bright()
    fv = extract_features(bright)
    assert fv.overexposed_ratio > 0.90, "Nearly all pixels should be overexposed"
    assert fv.brightness_mean > 200


def test_noise_estimate_higher_for_noisy_image():
    clean = _gradient()
    noisy = _noisy(sigma=40)
    fv_c = extract_features(clean)
    fv_n = extract_features(noisy)
    assert fv_n.noise_estimate > fv_c.noise_estimate, (
        "Noisy image should have higher noise estimate"
    )


def test_rms_contrast_low_for_solid_image():
    solid = _solid(value=128)
    fv = extract_features(solid)
    assert fv.rms_contrast < 0.02, "Solid image should have near-zero RMS contrast"


def test_rms_contrast_higher_for_gradient():
    fv = extract_features(_gradient())
    assert fv.rms_contrast > 0.1, "Gradient should have significant RMS contrast"


def test_metadata_fields_correct():
    img = np.zeros((240, 320, 3), dtype=np.uint8)
    fv = extract_features(img)
    assert fv.width == 320
    assert fv.height == 240
    assert fv.channels == 3
    assert abs(fv.aspect_ratio - (320 / 240)) < 0.01


def test_features_from_bytes_returns_none_for_invalid():
    from app.services.feature_extractor import features_from_bytes
    result = features_from_bytes(b"not an image")
    assert result is None


def test_features_from_bytes_works_for_valid_jpeg():
    from app.services.feature_extractor import features_from_bytes
    img = _gradient()
    _, enc = cv2.imencode(".jpg", img)
    fv = features_from_bytes(enc.tobytes())
    assert fv is not None
    assert fv.width > 0


def test_to_dict_has_all_required_keys():
    fv = extract_features(_gradient())
    d = fv.to_dict()
    required = [
        "laplacian_variance", "sobel_mean", "sobel_std", "tenengrad_score", "sharpness_score",
        "brightness_mean", "brightness_std", "underexposed_ratio", "overexposed_ratio", "histogram_entropy",
        "noise_estimate", "snr_db", "rms_contrast", "michelson_contrast",
        "saturation_mean", "saturation_std", "colorfulness_index",
        "glcm_energy", "glcm_homogeneity", "glcm_correlation",
        "jpeg_artifact_score", "blocking_score",
        "width", "height", "channels", "aspect_ratio",
    ]
    for key in required:
        assert key in d, f"Missing feature key: {key}"
