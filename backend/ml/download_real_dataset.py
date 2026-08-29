"""
download_real_dataset.py
========================
Downloads a real-world image quality dataset from public-domain Wikimedia Commons images.
Organizes them into a `labels.csv` and an `images/` directory.
"""

import urllib.request
import csv
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DATASET_DIR = Path("data/real_dataset")
IMAGES_DIR  = DATASET_DIR / "images"
LABELS_FILE = DATASET_DIR / "labels.csv"

# Public domain images from Wikimedia Commons representing various quality issues
URLS = {
    "GOOD": [
        # Landscapes / sharp objects / portraits
        ("good_landscape1.jpg", "https://upload.wikimedia.org/wikipedia/commons/c/c8/Altja_j%C3%B5gi_Lahemaal.jpg"),
        ("good_landscape2.jpg", "https://upload.wikimedia.org/wikipedia/commons/b/b8/Fisheye_view_of_the_interior_of_the_large_dome_of_the_Yerevan_Opera_Theater.jpg"),
        ("good_flower.jpg",      "https://upload.wikimedia.org/wikipedia/commons/a/a5/Flower_poster_2.jpg"),
        ("good_city.jpg",        "https://upload.wikimedia.org/wikipedia/commons/8/82/Toronto_-_ON_-_Toronto_Harbour_and_Skyline.jpg"),
        ("good_portrait1.jpg",   "https://upload.wikimedia.org/wikipedia/commons/f/f9/A_portrait_of_a_woman.jpg"),
        ("good_architecture.jpg","https://upload.wikimedia.org/wikipedia/commons/0/02/Versailles_Palace_Facade.jpg"),
        ("good_nature.jpg",      "https://upload.wikimedia.org/wikipedia/commons/e/f8/Oak_tree_in_summer.jpg"),
        ("good_birds.jpg",       "https://upload.wikimedia.org/wikipedia/commons/2/23/Cyanocitta_cristata_blue_jay.jpg"),
        ("good_statue.jpg",      "https://upload.wikimedia.org/wikipedia/commons/8/8f/Lincoln_Memorial_statue_close_up.jpg"),
        ("good_street.jpg",      "https://upload.wikimedia.org/wikipedia/commons/6/64/NYC_Street_2011.jpg"),
    ],
    "DEGRADED": [
        # Blurry
        ("degraded_blur1.jpg",    "https://upload.wikimedia.org/wikipedia/commons/3/30/Blurry_Car.jpg"),
        ("degraded_blur2.jpg",    "https://upload.wikimedia.org/wikipedia/commons/b/bd/Blurry_forest_trail.jpg"),
        ("degraded_blur3.jpg",    "https://upload.wikimedia.org/wikipedia/commons/d/da/Motion_blur_running_dog.jpg"),
        ("degraded_blur4.jpg",    "https://upload.wikimedia.org/wikipedia/commons/8/82/Blurry_lights_and_bokeh_%2849318991462%29.jpg"),
        # Noisy
        ("degraded_noise1.jpg",   "https://upload.wikimedia.org/wikipedia/commons/d/d4/Noise_example.jpg"),
        ("degraded_noise2.jpg",   "https://upload.wikimedia.org/wikipedia/commons/a/ae/High_ISO_noise_sample.jpg"),
        ("degraded_noise3.jpg",   "https://upload.wikimedia.org/wikipedia/commons/5/52/High_ISO_noise_at_3200_ISO.jpg"),
        ("degraded_noise4.jpg",   "https://upload.wikimedia.org/wikipedia/commons/c/ce/Luminance_noise_sky.jpg"),
        # Exposure
        ("degraded_dark1.jpg",    "https://upload.wikimedia.org/wikipedia/commons/d/db/Under-exposed_photo_of_house.jpg"),
        ("degraded_dark2.jpg",    "https://upload.wikimedia.org/wikipedia/commons/0/0b/Underexposed_sunset.jpg"),
        ("degraded_dark3.jpg",    "https://upload.wikimedia.org/wikipedia/commons/5/56/Underexposed_interior.jpg"),
        ("degraded_bright1.jpg",  "https://upload.wikimedia.org/wikipedia/commons/5/57/Overexposed_portrait.jpg"),
        ("degraded_bright2.jpg",  "https://upload.wikimedia.org/wikipedia/commons/0/00/Overexposed_cityscape.jpg"),
        ("degraded_bright3.jpg",  "https://upload.wikimedia.org/wikipedia/commons/c/c5/Overexposed_winter_landscape.jpg"),
    ],
    "DEFECTIVE": [
        # JPEG artifacts / Glitch / Severe corruption / Defects
        ("defective_jpeg1.jpg",   "https://upload.wikimedia.org/wikipedia/commons/b/b2/JPEG_artifacts.jpg"),
        ("defective_jpeg2.jpg",   "https://upload.wikimedia.org/wikipedia/commons/7/7b/Blocky_JPEG_compression.jpg"),
        ("defective_glitch1.jpg", "https://upload.wikimedia.org/wikipedia/commons/a/ab/Glitch_art_image.png"),
        ("defective_glitch2.jpg", "https://upload.wikimedia.org/wikipedia/commons/0/03/Digital_glitch_corruption.png"),
        ("defective_scratches.jpg","https://upload.wikimedia.org/wikipedia/commons/4/4c/Severe_photo_scratches_damage.jpg"),
        ("defective_dust.jpg",     "https://upload.wikimedia.org/wikipedia/commons/b/bd/Dust_and_scratches_film_scan.jpg"),
        ("defective_stains.jpg",   "https://upload.wikimedia.org/wikipedia/commons/f/f3/Water_damaged_stained_photograph.jpg"),
        ("defective_damaged.jpg",  "https://upload.wikimedia.org/wikipedia/commons/d/de/Torn_and_creased_paper_photo.jpg"),
        ("defective_lowqual.jpg",  "https://upload.wikimedia.org/wikipedia/commons/9/91/Low_resolution_pixelated_compression.jpg"),
        ("defective_severe.jpg",   "https://upload.wikimedia.org/wikipedia/commons/a/a2/Highly_degraded_overcompressed_image.jpg"),
    ]
}


import time

def download_image(url: str, dest: Path) -> bool:
    """Download an image with a descriptive user-agent and 1.5s delay to avoid 429."""
    # Sleep to prevent rate limit
    time.sleep(1.5)
    
    # Wikimedia Commons requires a descriptive User-Agent with contact info or application name
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "ImageQualityAssessmentTool/1.0 (contact: akshithareddy005@gmail.com; user-agent-policy)"}
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as response:
            dest.write_bytes(response.read())
        return True
    except Exception as e:
        logger.error(f"Failed to download {url}: {e}")
        return False


def main():
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    rows = []

    logger.info("Starting real-world image quality dataset download...")

    for label, items in URLS.items():
        logger.info(f"Downloading {label} category...")
        for filename, url in items:
            dest = IMAGES_DIR / filename
            if dest.exists():
                logger.info(f"  Already exists: {filename}")
                rows.append({"filename": filename, "label": label})
                continue

            logger.info(f"  Downloading: {filename} from {url}")
            if download_image(url, dest):
                rows.append({"filename": filename, "label": label})

    # Save labels.csv
    with LABELS_FILE.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "label"])
        writer.writeheader()
        writer.writerows(rows)

    logger.info(f"Download complete! Labeled {len(rows)} images to {LABELS_FILE}")


if __name__ == "__main__":
    main()
