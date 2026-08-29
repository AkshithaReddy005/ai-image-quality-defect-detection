"""
generate_dataset.py
===================
Generates a synthetic image quality dataset by applying controlled degradations
to clean seed images and saving labeled samples.

Degradations applied:
  - Gaussian blur (various σ values)
  - Brightness reduction (underexposure)
  - Brightness increase (overexposure)
  - Gaussian noise (various σ values)
  - JPEG compression artifacts (low quality)
  - Combined degradations (multiple issues at once)

Output:
  data/synthetic_dataset/
    images/      — degraded image files
    labels.csv   — columns: filename, label, issues (JSON list)
"""

import cv2
import numpy as np
import json
import csv
import random
import uuid
import argparse
import logging
from pathlib import Path
from typing import Callable

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SEED          = 42
random.seed(SEED)
np.random.seed(SEED)

DATASET_DIR   = Path("data/synthetic_dataset")
IMAGES_DIR    = DATASET_DIR / "images"
LABELS_FILE   = DATASET_DIR / "labels.csv"


# ── Seed image generation (no external dataset needed) ────────────────────────

def _create_seed_images(n: int = 200) -> list[np.ndarray]:
    """
    Generate synthetic clean seed images using structured patterns,
    gradients, and random textures — no external images required.
    """
    images = []
    for i in range(n):
        h, w = random.choice([(256, 256), (320, 240), (224, 224)])
        kind = i % 5

        if kind == 0:
            # Random structured gradient
            img = np.zeros((h, w, 3), dtype=np.uint8)
            x = np.linspace(60, 200, w, dtype=np.uint8)
            y = np.linspace(60, 200, h, dtype=np.uint8)
            img[:, :, 0] = np.outer(y, np.ones(w, dtype=np.uint8)) // 1
            img[:, :, 1] = np.outer(np.ones(h, dtype=np.uint8), x) // 1
            img[:, :, 2] = np.random.randint(80, 160, (h, w), dtype=np.uint8)

        elif kind == 1:
            # Checkerboard texture
            sq = random.choice([16, 32])
            img = np.zeros((h, w, 3), dtype=np.uint8)
            for r in range(0, h, sq):
                for c in range(0, w, sq):
                    if (r // sq + c // sq) % 2 == 0:
                        img[r:r+sq, c:c+sq] = np.random.randint(100, 220, 3, dtype=np.uint8)
                    else:
                        img[r:r+sq, c:c+sq] = np.random.randint(30, 120, 3, dtype=np.uint8)

        elif kind == 2:
            # Smooth noise (good texture image)
            base = np.random.randint(80, 180, (h // 8, w // 8, 3), dtype=np.uint8)
            img = cv2.resize(base, (w, h), interpolation=cv2.INTER_CUBIC)
            img = cv2.GaussianBlur(img, (5, 5), 0)

        elif kind == 3:
            # Concentric circles
            img = np.full((h, w, 3), 128, dtype=np.uint8)
            cx, cy = w // 2, h // 2
            color = tuple(int(x) for x in np.random.randint(50, 220, 3))
            for r in range(20, min(h, w) // 2, 20):
                cv2.circle(img, (cx, cy), r, color, 2)

        else:
            # Random rectangles
            img = np.full((h, w, 3), 200, dtype=np.uint8)
            for _ in range(random.randint(5, 15)):
                x1, y1 = random.randint(0, w - 1), random.randint(0, h - 1)
                x2, y2 = random.randint(x1, w), random.randint(y1, h)
                color = tuple(int(x) for x in np.random.randint(30, 220, 3))
                cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)

        images.append(img)

    return images


# ── Degradation functions ─────────────────────────────────────────────────────

def degrade_blur(img: np.ndarray) -> tuple[np.ndarray, str, str]:
    sigma = random.uniform(3, 12)
    k = int(sigma * 3) | 1  # odd kernel
    degraded = cv2.GaussianBlur(img, (k, k), sigma)
    severity = "high" if sigma > 7 else "medium"
    return degraded, "DEGRADED", json.dumps([{"type": "blur", "severity": severity}])


def degrade_underexposure(img: np.ndarray) -> tuple[np.ndarray, str, str]:
    factor = random.uniform(0.15, 0.45)
    degraded = np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)
    severity = "high" if factor < 0.25 else "medium"
    return degraded, "DEGRADED", json.dumps([{"type": "underexposure", "severity": severity}])


def degrade_overexposure(img: np.ndarray) -> tuple[np.ndarray, str, str]:
    shift = random.uniform(80, 160)
    degraded = np.clip(img.astype(np.float32) + shift, 0, 255).astype(np.uint8)
    severity = "high" if shift > 120 else "medium"
    return degraded, "DEGRADED", json.dumps([{"type": "overexposure", "severity": severity}])


def degrade_noise(img: np.ndarray) -> tuple[np.ndarray, str, str]:
    sigma = random.uniform(15, 55)
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    degraded = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    severity = "high" if sigma > 35 else "medium"
    return degraded, "DEGRADED", json.dumps([{"type": "noise", "severity": severity}])


def degrade_jpeg(img: np.ndarray) -> tuple[np.ndarray, str, str]:
    quality = random.randint(1, 15)
    _, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    degraded = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return degraded, "DEGRADED", json.dumps([{"type": "jpeg_artifacts", "severity": "high"}])


def degrade_combined(img: np.ndarray) -> tuple[np.ndarray, str, str]:
    """Apply 2–3 degradations simultaneously."""
    fns = [degrade_blur, degrade_noise, degrade_underexposure, degrade_overexposure, degrade_jpeg]
    selected = random.sample(fns, random.randint(2, 3))
    degraded = img.copy()
    issue_types = []
    for fn in selected:
        degraded, _, issues_json = fn(degraded)
        issues = json.loads(issues_json)
        issue_types.extend(issues)
    return degraded, "DEFECTIVE", json.dumps(issue_types)


def clean_image(img: np.ndarray) -> tuple[np.ndarray, str, str]:
    """Slightly process a clean image (minor sharpening) — label GOOD."""
    kernel = np.array([[0, -0.5, 0], [-0.5, 3, -0.5], [0, -0.5, 0]], dtype=np.float32)
    # Use uint8 image with filter2D to avoid OpenCV float dtype issues
    sharpened = cv2.filter2D(img, -1, kernel)
    out = np.clip(sharpened, 0, 255).astype(np.uint8)
    return out, "GOOD", json.dumps([])


# ── Dataset generation ────────────────────────────────────────────────────────

def generate_dataset(n_seeds: int = 200, samples_per_seed: int = 25) -> None:
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    logger.info(f"Generating {n_seeds} seed images...")
    seeds = _create_seed_images(n_seeds)

    degraders: list[Callable] = [
        clean_image,           # 1×  → ~20% good
        degrade_blur,          # 1×
        degrade_underexposure, # 1×
        degrade_overexposure,  # 1×
        degrade_noise,         # 1×
        degrade_jpeg,          # 1×
        degrade_combined,      # 2× → more defective samples
        degrade_combined,
    ]

    rows = []
    total = 0

    for seed_img in seeds:
        for fn in degraders:
            try:
                degraded, label, issues_json = fn(seed_img)
                fname = f"{uuid.uuid4().hex}.jpg"
                fpath = IMAGES_DIR / fname
                cv2.imwrite(str(fpath), degraded, [cv2.IMWRITE_JPEG_QUALITY, 95])
                rows.append({"filename": fname, "label": label, "issues": issues_json})
                total += 1
            except Exception as e:
                logger.warning(f"Failed to generate sample: {e}")

    with LABELS_FILE.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "label", "issues"])
        writer.writeheader()
        writer.writerows(rows)

    logger.info(f"Dataset generated: {total} samples → {DATASET_DIR}")

    # Print class distribution
    from collections import Counter
    labels = [r["label"] for r in rows]
    dist = Counter(labels)
    for label, count in sorted(dist.items()):
        logger.info(f"  {label}: {count} ({count/total*100:.1f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic image quality dataset")
    parser.add_argument("--seeds", type=int, default=200, help="Number of seed images")
    parser.add_argument("--samples-per-seed", type=int, default=8, help="Degradations per seed")
    args = parser.parse_args()
    generate_dataset(n_seeds=args.seeds, samples_per_seed=args.samples_per_seed)
