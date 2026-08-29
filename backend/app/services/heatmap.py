"""
heatmap.py
==========
Generates a sliding-window saliency map showing which regions of the image
contribute most to quality degradation.

The approach:
  1. Divide the image into overlapping patches.
  2. For each patch, extract features and compute a local quality score.
  3. Build a score grid; invert so that degraded regions → high activation.
  4. Upsample and apply a colour-map (JET) to produce a visual heatmap.
  5. Overlay on the original image with alpha blending.
"""

import cv2
import numpy as np
import uuid
import logging
from pathlib import Path
from app.config import settings
from app.services.feature_extractor import extract_features
from app.services.model_inference import run_inference

logger = logging.getLogger(__name__)

PATCH_SIZE   = 128
STRIDE       = 64
ALPHA        = 0.5     # overlay transparency


def generate_heatmap(img_bgr: np.ndarray, analysis_id: str) -> str | None:
    """
    Generate a quality-saliency heatmap for img_bgr.
    Saves the overlay PNG and returns the relative path, or None on failure.
    """
    try:
        heatmap_dir = Path(settings.HEATMAP_DIR)
        heatmap_dir.mkdir(parents=True, exist_ok=True)

        h, w = img_bgr.shape[:2]
        score_grid = np.zeros((h, w), dtype=np.float32)
        count_grid = np.zeros((h, w), dtype=np.float32)

        # Sample patches across the image
        rows = list(range(0, h - PATCH_SIZE + 1, STRIDE))
        cols = list(range(0, w - PATCH_SIZE + 1, STRIDE))

        # Ensure at least a 1×1 grid for small images
        if not rows:
            rows = [0]
        if not cols:
            cols = [0]

        for r in rows:
            for c in cols:
                pr = min(r + PATCH_SIZE, h)
                pc = min(c + PATCH_SIZE, w)
                patch = img_bgr[r:pr, c:pc]

                if patch.shape[0] < 8 or patch.shape[1] < 8:
                    continue

                try:
                    fv = extract_features(patch)
                    result = run_inference(fv)
                    patch_score = result["quality_score"]  # 0–100
                except Exception:
                    patch_score = 50.0

                # Degradation activation: low score → high activation
                activation = (100.0 - patch_score) / 100.0

                score_grid[r:pr, c:pc] += activation
                count_grid[r:pr, c:pc] += 1.0

        # Normalise
        valid = count_grid > 0
        score_grid[valid] /= count_grid[valid]
        score_grid = (score_grid * 255).astype(np.uint8)

        # Smooth
        score_grid = cv2.GaussianBlur(score_grid, (31, 31), 0)

        # Colormap
        heatmap_colored = cv2.applyColorMap(score_grid, cv2.COLORMAP_JET)

        # Overlay on resized original
        overlay = cv2.addWeighted(img_bgr, 1 - ALPHA, heatmap_colored, ALPHA, 0)

        out_path = heatmap_dir / f"{analysis_id}.png"
        cv2.imwrite(str(out_path), overlay)
        logger.info(f"Heatmap saved: {out_path}")
        return str(out_path)

    except Exception as e:
        logger.error(f"Heatmap generation failed: {e}")
        return None
