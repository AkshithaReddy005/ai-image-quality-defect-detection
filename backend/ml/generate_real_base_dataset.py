"""
generate_real_base_dataset.py
=============================
Downloads high-quality real-world base photographs from Picsum Photos and
applies standard parameterized degradations (matching LIVE/TID2013 benchmarks)
to construct a real-world image quality dataset containing 200 samples.
"""

import urllib.request
import csv
import logging
import random
import time
import json
from pathlib import Path
import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DATASET_DIR = Path("data/real_dataset")
IMAGES_DIR  = DATASET_DIR / "images"
LABELS_FILE = DATASET_DIR / "labels.csv"

# Picsum Photos IDs for beautiful real-world camera photographs
PICSUM_IDS = [
    10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
    20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
    30, 31, 32, 33, 34, 35, 36, 37, 38, 39
]


def download_base_image(image_id: int, dest: Path) -> bool:
    """Download a real photograph from Picsum Photos."""
    url = f"https://picsum.photos/id/{image_id}/640/480"
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "ImageQualityAssessmentTool/1.0 (contact: akshithareddy005@gmail.com)"}
    )
    try:
        # Retry up to 3 times for robustness
        for attempt in range(3):
            try:
                with urllib.request.urlopen(req, timeout=15) as response:
                    dest.write_bytes(response.read())
                return True
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(1.0)
    except Exception as e:
        logger.error(f"Failed to download Picsum ID {image_id}: {e}")
        return False


# Degradation functions (applied to real base images)

def clean_image(img: np.ndarray) -> np.ndarray:
    """Keep the real image clean, apply minor sharpening."""
    kernel = np.array([[0, -0.2, 0], [-0.2, 1.8, -0.2], [0, -0.2, 0]], dtype=np.float32)
    sharpened = cv2.filter2D(img, -1, kernel)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def blur_image(img: np.ndarray, severity: int) -> np.ndarray:
    """Apply Gaussian Blur."""
    ksize = severity * 4 + 1
    return cv2.GaussianBlur(img, (ksize, ksize), severity)


def underexpose_image(img: np.ndarray, severity: float) -> np.ndarray:
    """Scale down brightness."""
    return np.clip(img.astype(np.float32) * severity, 0, 255).astype(np.uint8)


def overexpose_image(img: np.ndarray, severity: float) -> np.ndarray:
    """Scale up brightness, clipping highlights."""
    return np.clip(img.astype(np.float32) + severity, 0, 255).astype(np.uint8)


def noise_image(img: np.ndarray, severity: float) -> np.ndarray:
    """Add Gaussian noise."""
    noise = np.random.normal(0, severity, img.shape).astype(np.float32)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def jpeg_compress_image(img: np.ndarray, quality: int) -> np.ndarray:
    """Compress image with low JPEG quality."""
    _, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)


def main():
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    temp_dir = DATASET_DIR / "temp_seeds"
    temp_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Download base images
    logger.info("Downloading real-world base photographs from Picsum Photos...")
    downloaded_seeds = []
    
    for img_id in PICSUM_IDS:
        seed_path = temp_dir / f"seed_{img_id}.jpg"
        if not seed_path.exists():
            logger.info(f"  Downloading seed photo ID: {img_id}")
            if download_base_image(img_id, seed_path):
                downloaded_seeds.append(seed_path)
            # Sleep to be polite to the Picsum CDN
            time.sleep(0.5)
        else:
            downloaded_seeds.append(seed_path)

    if not downloaded_seeds:
        logger.error("No base images could be downloaded. Check internet connection.")
        return

    logger.info(f"Downloaded {len(downloaded_seeds)} base photographs.")

    # Step 2: Generate degraded dataset
    rows = []
    logger.info("Applying quality degradation matrices to generate final dataset...")

    for i, seed_path in enumerate(downloaded_seeds):
        img = cv2.imread(str(seed_path))
        if img is None:
            continue

        # Save Clean (GOOD)
        name_clean = f"real_{i}_clean.jpg"
        cv2.imwrite(str(IMAGES_DIR / name_clean), clean_image(img))
        rows.append({"filename": name_clean, "label": "GOOD"})

        # Save Blurry (DEGRADED)
        name_blur = f"real_{i}_blur.jpg"
        cv2.imwrite(str(IMAGES_DIR / name_blur), blur_image(img, severity=random.choice([3, 5, 7])))
        rows.append({"filename": name_blur, "label": "DEGRADED"})

        # Save Underexposed (DEGRADED)
        name_under = f"real_{i}_under.jpg"
        cv2.imwrite(str(IMAGES_DIR / name_under), underexpose_image(img, severity=random.choice([0.2, 0.3, 0.4])))
        rows.append({"filename": name_under, "label": "DEGRADED"})

        # Save Overexposed (DEGRADED)
        name_over = f"real_{i}_over.jpg"
        cv2.imwrite(str(IMAGES_DIR / name_over), overexpose_image(img, severity=random.choice([100, 120, 140])))
        rows.append({"filename": name_over, "label": "DEGRADED"})

        # Save Noisy (DEGRADED)
        name_noise = f"real_{i}_noise.jpg"
        cv2.imwrite(str(IMAGES_DIR / name_noise), noise_image(img, severity=random.choice([25, 35, 45])))
        rows.append({"filename": name_noise, "label": "DEGRADED"})

        # Save JPEG Compressed (DEFECTIVE)
        name_jpeg = f"real_{i}_jpeg.jpg"
        cv2.imwrite(str(IMAGES_DIR / name_jpeg), jpeg_compress_image(img, quality=random.choice([3, 5, 8])))
        rows.append({"filename": name_jpeg, "label": "DEFECTIVE"})

        # Save Combined severe degradation (DEFECTIVE)
        name_comb = f"real_{i}_combined.jpg"
        comb_img = blur_image(img, severity=4)
        comb_img = noise_image(comb_img, severity=20)
        comb_img = underexpose_image(comb_img, severity=0.3)
        cv2.imwrite(str(IMAGES_DIR / name_comb), comb_img)
        rows.append({"filename": name_comb, "label": "DEFECTIVE"})

    # Write labels.csv
    with LABELS_FILE.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "label"])
        writer.writeheader()
        writer.writerows(rows)

    # Clean up temp folder
    for path in temp_dir.glob("*"):
        path.unlink()
    temp_dir.rmdir()

    logger.info(f"Dataset generated with {len(rows)} real-world base images at {LABELS_FILE}")


if __name__ == "__main__":
    main()
