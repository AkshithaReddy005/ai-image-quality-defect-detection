"""
train.py
========
Trains a Random Forest classifier on the synthetic image quality dataset.

Pipeline:
  1. Load dataset (generate if not found)
  2. Extract feature vectors for every image
  3. Split into train / test (80/20, stratified)
  4. Scale features (StandardScaler)
  5. Train Random Forest with cross-validation
  6. Save model, scaler, and label encoder
  7. Print quick evaluation summary

Usage:
  cd backend
  python ml/train.py [--seeds 200] [--force-regen]
"""

import sys
import csv
import logging
import argparse
import numpy as np
from pathlib import Path

# Ensure backend root is on sys.path when running directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from app.services.feature_extractor import extract_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DATASET_DIR = Path("data/synthetic_dataset")
MODEL_DIR   = Path("ml/models")


def load_dataset(dataset_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load images and extract features. Returns (X, y)."""
    images_dir  = dataset_dir / "images"
    labels_file = dataset_dir / "labels.csv"

    if not labels_file.exists():
        raise FileNotFoundError(
            f"Dataset labels not found at {labels_file}. "
            "Please ensure the directory contains 'labels.csv' and an 'images/' folder."
        )

    with labels_file.open() as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    X_list, y_list = [], []
    n_failed = 0

    for i, row in enumerate(rows):
        fpath = images_dir / row["filename"]
        if not fpath.exists():
            n_failed += 1
            continue

        img = cv2.imread(str(fpath))
        if img is None:
            n_failed += 1
            continue

        try:
            fv = extract_features(img)
            X_list.append(fv.to_model_array())
            y_list.append(row["label"])
        except Exception as e:
            logger.warning(f"Feature extraction failed for {fpath}: {e}")
            n_failed += 1

        if (i + 1) % 100 == 0:
            logger.info(f"  Processed {i+1}/{len(rows)} images...")

    logger.info(f"Loaded {len(X_list)} samples ({n_failed} failed)")

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list)
    return X, y


def train(dataset_dir: Path, force_regen: bool = False, n_seeds: int = 200) -> None:
    # If using default synthetic dataset and it doesn't exist, generate it
    is_default = (dataset_dir == DATASET_DIR)
    labels_file = dataset_dir / "labels.csv"

    if is_default and (force_regen or not labels_file.exists()):
        logger.info("Generating default synthetic dataset...")
        from ml.generate_dataset import generate_dataset
        generate_dataset(n_seeds=n_seeds)

    logger.info(f"Loading dataset from {dataset_dir} and extracting features...")
    X, y = load_dataset(dataset_dir)

    # Encode labels
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    logger.info(f"Classes: {list(le.classes_)}")

    # Train / test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.20, random_state=42, stratify=y_enc
    )
    logger.info(f"Train: {len(X_train)} | Test: {len(X_test)}")

    # Scale features
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # Train Random Forest
    logger.info("Training Random Forest classifier...")
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train_s, y_train)

    # Cross-validation
    cv_scores = cross_val_score(rf, X_train_s, y_train, cv=5, scoring="f1_macro")
    logger.info(f"5-fold CV F1-macro: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    # Test evaluation
    y_pred = rf.predict(X_test_s)
    acc = accuracy_score(y_test, y_pred)
    logger.info(f"Test accuracy: {acc:.3f}")
    logger.info("\n" + classification_report(y_test, y_pred, target_names=le.classes_))
    cm = confusion_matrix(y_test, y_pred)
    logger.info(f"Confusion matrix:\n{cm}")

    # Feature importance
    feat_names = [
        "laplacian_var", "sobel_mean", "sobel_std", "tenengrad", "sharpness",
        "brightness_mean", "brightness_std", "underexp_ratio", "overexp_ratio", "hist_entropy",
        "noise_estimate", "snr_db",
        "rms_contrast", "michelson_contrast",
        "sat_mean", "sat_std", "colorfulness",
        "glcm_energy", "glcm_homogeneity", "glcm_correlation",
        "jpeg_artifact_score", "blocking_score",
    ]
    importances = rf.feature_importances_
    ranked = sorted(zip(feat_names, importances), key=lambda x: -x[1])
    logger.info("\nTop-10 feature importances:")
    for name, imp in ranked[:10]:
        logger.info(f"  {name:<25} {imp:.4f}")

    # Save models
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(rf,     MODEL_DIR / "quality_rf_model.pkl")
    joblib.dump(scaler, MODEL_DIR / "feature_scaler.pkl")
    joblib.dump(le,     MODEL_DIR / "label_encoder.pkl")
    logger.info(f"\nModels saved to {MODEL_DIR}/")

    # Save evaluation report
    report_path = MODEL_DIR / "evaluation_report.txt"
    with report_path.open("w") as f:
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"CV F1-macro: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}\n\n")
        f.write(classification_report(y_test, y_pred, target_names=le.classes_))
        f.write(f"\nConfusion Matrix:\n{cm}\n\n")
        f.write("Feature Importances:\n")
        for name, imp in ranked:
            f.write(f"  {name:<25} {imp:.4f}\n")
    logger.info(f"Evaluation report saved to {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train image quality classifier")
    parser.add_argument("--dataset-dir", type=str, default=str(DATASET_DIR), help="Path to custom dataset directory containing 'images/' and 'labels.csv'")
    parser.add_argument("--force-regen", action="store_true", help="Regenerate default synthetic dataset before training")
    parser.add_argument("--seeds", type=int, default=200, help="Number of seed images for generation (only if using synthetic dataset)")
    args = parser.parse_args()
    train(dataset_dir=Path(args.dataset_dir), force_regen=args.force_regen, n_seeds=args.seeds)
