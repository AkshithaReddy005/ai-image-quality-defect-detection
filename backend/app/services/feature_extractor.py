"""
feature_extractor.py
====================
Extracts a 27-dimensional feature vector from an image for quality analysis.

Features extracted:
  Sharpness (4):    laplacian_variance, sobel_mean, sobel_std, tenengrad_score
  Exposure (5):     brightness_mean, brightness_std, underexposed_ratio,
                    overexposed_ratio, histogram_entropy
  Noise (2):        noise_estimate (MAD on HF wavelet), snr_db
  Contrast (2):     rms_contrast, michelson_contrast
  Color/Texture (6):saturation_mean, saturation_std, colorfulness_index,
                    glcm_energy, glcm_homogeneity, glcm_correlation
  Corruption (2):   jpeg_artifact_score, blocking_score
  Meta (6):         width, height, channels, aspect_ratio (not used in model)
"""

import cv2
import numpy as np
from dataclasses import dataclass, asdict
from typing import Optional


UNDEREXPOSED_THRESHOLD = 30
OVEREXPOSED_THRESHOLD  = 225
BLOCK_SIZE             = 8   # JPEG DCT block size


@dataclass
class FeatureVector:
    # Sharpness
    laplacian_variance: float = 0.0
    sobel_mean: float = 0.0
    sobel_std: float = 0.0
    tenengrad_score: float = 0.0
    sharpness_score: float = 0.0  # normalised composite

    # Exposure
    brightness_mean: float = 0.0
    brightness_std: float = 0.0
    underexposed_ratio: float = 0.0
    overexposed_ratio: float = 0.0
    histogram_entropy: float = 0.0

    # Noise
    noise_estimate: float = 0.0
    snr_db: float = 0.0

    # Contrast
    rms_contrast: float = 0.0
    michelson_contrast: float = 0.0

    # Color / Texture
    saturation_mean: float = 0.0
    saturation_std: float = 0.0
    colorfulness_index: float = 0.0
    glcm_energy: float = 0.0
    glcm_homogeneity: float = 0.0
    glcm_correlation: float = 0.0

    # Corruption
    jpeg_artifact_score: float = 0.0
    blocking_score: float = 0.0

    # Metadata (not fed into model)
    width: int = 0
    height: int = 0
    channels: int = 0
    aspect_ratio: float = 0.0

    def to_model_array(self) -> np.ndarray:
        """Return the 22 features used as ML input (excludes metadata)."""
        return np.array([
            self.laplacian_variance,
            self.sobel_mean,
            self.sobel_std,
            self.tenengrad_score,
            self.sharpness_score,
            self.brightness_mean,
            self.brightness_std,
            self.underexposed_ratio,
            self.overexposed_ratio,
            self.histogram_entropy,
            self.noise_estimate,
            self.snr_db,
            self.rms_contrast,
            self.michelson_contrast,
            self.saturation_mean,
            self.saturation_std,
            self.colorfulness_index,
            self.glcm_energy,
            self.glcm_homogeneity,
            self.glcm_correlation,
            self.jpeg_artifact_score,
            self.blocking_score,
        ], dtype=np.float32)

    def to_dict(self) -> dict:
        return asdict(self)


# ── Helper functions ──────────────────────────────────────────────────────────

def _to_gray(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)


def _sharpness_features(gray: np.ndarray) -> dict:
    # Ensure uint8 for OpenCV operations
    if gray.dtype != np.uint8:
        gray = gray.astype(np.uint8)

    lap = cv2.Laplacian(gray, cv2.CV_64F)
    lap_var = float(lap.var())

    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_mag = np.hypot(sobelx, sobely)
    sobel_mean = float(sobel_mag.mean())
    sobel_std  = float(sobel_mag.std())

    # Tenengrad: sum of squared Sobel gradients above a threshold
    tenengrad = float((sobel_mag ** 2).mean())

    # Composite normalised sharpness [0, 1]  (sigmoid on log variance)
    sharpness_score = float(1.0 / (1.0 + np.exp(-0.003 * (lap_var - 300))))

    return dict(
        laplacian_variance=lap_var,
        sobel_mean=sobel_mean,
        sobel_std=sobel_std,
        tenengrad_score=tenengrad,
        sharpness_score=sharpness_score,
    )


def _exposure_features(gray: np.ndarray) -> dict:
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    total_px = gray.size

    brightness_mean = float(gray.mean())
    brightness_std  = float(gray.std())
    underexposed    = float(hist[:UNDEREXPOSED_THRESHOLD].sum()  / total_px)
    overexposed     = float(hist[OVEREXPOSED_THRESHOLD:].sum()   / total_px)

    # Shannon entropy of histogram
    hist_norm = hist / (hist.sum() + 1e-9)
    entropy   = float(-np.sum(hist_norm * np.log2(hist_norm + 1e-9)))

    return dict(
        brightness_mean=brightness_mean,
        brightness_std=brightness_std,
        underexposed_ratio=underexposed,
        overexposed_ratio=overexposed,
        histogram_entropy=entropy,
    )


def _noise_features(gray: np.ndarray) -> dict:
    """
    Estimate noise using the median absolute deviation of high-frequency
    components extracted via a simple 2D Haar-like wavelet decomposition.
    """
    # Downsample to speed up; noise estimate is scale-invariant
    h, w = gray.shape
    if max(h, w) > 512:
        scale = 512 / max(h, w)
        gray = cv2.resize(gray, (int(w * scale), int(h * scale)))

    # High-pass by subtracting blurred version
    blurred = cv2.GaussianBlur(gray.astype(np.float32), (5, 5), 0)
    hf = gray.astype(np.float32) - blurred
    mad = float(np.median(np.abs(hf - np.median(hf))))
    sigma_est = mad / 0.6745  # consistent estimator

    signal_power = float((gray.astype(np.float32) ** 2).mean())
    noise_power  = sigma_est ** 2 + 1e-9
    snr_db = float(10 * np.log10(signal_power / noise_power + 1e-9))

    return dict(noise_estimate=sigma_est, snr_db=snr_db)


def _contrast_features(gray: np.ndarray) -> dict:
    g = gray.astype(np.float32) / 255.0
    rms = float(np.sqrt(((g - g.mean()) ** 2).mean()))
    g_min, g_max = float(g.min()), float(g.max())
    michelson = float((g_max - g_min) / (g_max + g_min + 1e-9))
    return dict(rms_contrast=rms, michelson_contrast=michelson)


def _color_texture_features(img_bgr: np.ndarray, gray: np.ndarray) -> dict:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1].astype(np.float32) / 255.0
    sat_mean = float(sat.mean())
    sat_std  = float(sat.std())

    # Colorfulness index (Hasler & Süsstrunk, 2003)
    B, G, R = img_bgr[:, :, 0].astype(float), img_bgr[:, :, 1].astype(float), img_bgr[:, :, 2].astype(float)
    rg = R - G
    yb = 0.5 * (R + G) - B
    colorfulness = float(
        np.sqrt(rg.std()**2 + yb.std()**2) + 0.3 * np.sqrt(rg.mean()**2 + yb.mean()**2)
    )

    # GLCM features (computed on downscaled gray)
    h, w = gray.shape
    if max(h, w) > 256:
        scale = 256 / max(h, w)
        g256 = cv2.resize(gray, (int(w * scale), int(h * scale)))
    else:
        g256 = gray.copy()

    glcm = _compute_glcm(g256, levels=32)
    energy       = float((glcm ** 2).sum())
    i_idx, j_idx = np.indices(glcm.shape)
    homogeneity  = float((glcm / (1 + np.abs(i_idx - j_idx))).sum())

    mu_i = float((i_idx * glcm).sum())
    mu_j = float((j_idx * glcm).sum())
    std_i = float(np.sqrt(((i_idx - mu_i) ** 2 * glcm).sum()))
    std_j = float(np.sqrt(((j_idx - mu_j) ** 2 * glcm).sum()))
    if std_i * std_j < 1e-9:
        correlation = 0.0
    else:
        correlation = float(((i_idx - mu_i) * (j_idx - mu_j) * glcm).sum() / (std_i * std_j))

    return dict(
        saturation_mean=sat_mean,
        saturation_std=sat_std,
        colorfulness_index=colorfulness,
        glcm_energy=energy,
        glcm_homogeneity=homogeneity,
        glcm_correlation=correlation,
    )


def _compute_glcm(gray: np.ndarray, levels: int = 32, distance: int = 1) -> np.ndarray:
    """Vectorised horizontal GLCM (offset = [0, distance])."""
    g = (gray.astype(np.float32) / 256.0 * levels).clip(0, levels - 1).astype(np.int32)
    i_vals = g[:, :-distance].ravel()
    j_vals = g[:, distance:].ravel()
    glcm = np.zeros((levels, levels), dtype=np.float64)
    np.add.at(glcm, (i_vals, j_vals), 1)
    glcm_sym = glcm + glcm.T
    glcm_sym /= (glcm_sym.sum() + 1e-9)
    return glcm_sym


def _corruption_features(gray: np.ndarray, img_bgr: np.ndarray) -> dict:
    """
    Estimate JPEG artifact severity by looking at DCT block boundaries.
    Blocking score: measure discontinuity at 8x8 block edges.
    """
    h, w = gray.shape
    bh = (h // BLOCK_SIZE) * BLOCK_SIZE
    bw = (w // BLOCK_SIZE) * BLOCK_SIZE
    g = gray[:bh, :bw].astype(np.float32)

    # Horizontal block edges
    h_edges = []
    for row in range(BLOCK_SIZE, bh, BLOCK_SIZE):
        diff = np.abs(g[row, :] - g[row - 1, :]).mean()
        interior = np.abs(g[row - 1, :] - g[row - 2, :]).mean() if row >= 2 else diff
        h_edges.append(float(diff / (interior + 1e-9)))

    # Vertical block edges
    v_edges = []
    for col in range(BLOCK_SIZE, bw, BLOCK_SIZE):
        diff = np.abs(g[:, col] - g[:, col - 1]).mean()
        interior = np.abs(g[:, col - 1] - g[:, col - 2]).mean() if col >= 2 else diff
        v_edges.append(float(diff / (interior + 1e-9)))

    blocking_score = float(np.mean(h_edges + v_edges)) if (h_edges or v_edges) else 0.0
    # Clamp: a perfect image has ratio ≈ 1; heavy JPEG has ratio >> 1
    blocking_score = min(blocking_score, 10.0)

    # JPEG artifact score: high-frequency energy in block-border rows/cols
    border_mask = np.zeros_like(g, dtype=bool)
    for row in range(0, bh, BLOCK_SIZE):
        border_mask[row, :] = True
    for col in range(0, bw, BLOCK_SIZE):
        border_mask[:, col] = True

    hf = cv2.Laplacian(g.astype(np.uint8), cv2.CV_64F)
    jpeg_score = float(np.abs(hf[border_mask]).mean() / (np.abs(hf).mean() + 1e-9))

    return dict(jpeg_artifact_score=jpeg_score, blocking_score=blocking_score)


# ── Public API ────────────────────────────────────────────────────────────────

def extract_features(img_bgr: np.ndarray) -> FeatureVector:
    """
    Given a BGR image array (as loaded by cv2.imread), compute the full
    feature vector.
    """
    gray = _to_gray(img_bgr)
    h, w = gray.shape
    channels = img_bgr.shape[2] if img_bgr.ndim == 3 else 1

    fv = FeatureVector()
    fv.width      = w
    fv.height     = h
    fv.channels   = channels
    fv.aspect_ratio = float(w / (h + 1e-9))

    fv.__dict__.update(_sharpness_features(gray))
    fv.__dict__.update(_exposure_features(gray))
    fv.__dict__.update(_noise_features(gray))
    fv.__dict__.update(_contrast_features(gray))
    fv.__dict__.update(_color_texture_features(img_bgr, gray))
    fv.__dict__.update(_corruption_features(gray, img_bgr))

    return fv


def features_from_bytes(image_bytes: bytes) -> Optional[FeatureVector]:
    """Decode image bytes and return feature vector, or None on failure."""
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return None
    return extract_features(img)
