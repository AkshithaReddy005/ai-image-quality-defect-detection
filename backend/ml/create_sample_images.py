"""
create_sample_images.py
=======================
Creates sample demonstration images in the sample_images/ directory.
Run once to generate images for testing and README demonstration.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np


def save(img, name):
    out = Path("sample_images") / name
    out.parent.mkdir(exist_ok=True)
    cv2.imwrite(str(out), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"Saved: {out}")


def base_image():
    """Create a clean reference image with texture."""
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    # Sky gradient
    for i in range(240):
        v = int(40 + i * 0.8)
        img[i, :] = [v + 30, v + 20, v]
    # Ground
    img[240:, :] = [40, 80, 40]
    # Some geometric shapes for texture
    cv2.rectangle(img, (100, 150), (250, 350), (200, 180, 140), -1)
    cv2.rectangle(img, (350, 200), (550, 400), (180, 140, 200), -1)
    cv2.circle(img, (320, 120), 60, (255, 220, 100), -1)
    # Add sharpening
    kernel = np.array([[-0.5,0,-0.5],[0,4,0],[-0.5,0,-0.5]], dtype=np.float32)
    img = cv2.filter2D(img, -1, kernel)
    return np.clip(img, 0, 255).astype(np.uint8)


if __name__ == "__main__":
    base = base_image()

    # 1. Good clean image
    save(base, "good_sharp.jpg")

    # 2. Blurred
    blurred = cv2.GaussianBlur(base, (31, 31), 8)
    save(blurred, "blurred.jpg")

    # 3. Dark / underexposed
    dark = np.clip(base.astype(np.float32) * 0.2, 0, 255).astype(np.uint8)
    save(dark, "dark.jpg")

    # 4. Bright / overexposed
    bright = np.clip(base.astype(np.float32) + 150, 0, 255).astype(np.uint8)
    save(bright, "bright.jpg")

    # 5. Noisy
    noise = np.random.normal(0, 40, base.shape).astype(np.float32)
    noisy = np.clip(base.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    save(noisy, "noisy.jpg")

    # 6. JPEG artifact
    _, enc = cv2.imencode(".jpg", base, [cv2.IMWRITE_JPEG_QUALITY, 3])
    jpeg_art = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    save(jpeg_art, "jpeg_artifact.jpg")

    # 7. Combined degradation (defective)
    combined = cv2.GaussianBlur(dark, (21, 21), 5)
    noise2 = np.random.normal(0, 25, combined.shape).astype(np.float32)
    combined = np.clip(combined.astype(np.float32) + noise2, 0, 255).astype(np.uint8)
    save(combined, "combined_defective.jpg")

    print("\nSample images created in backend/sample_images/")
