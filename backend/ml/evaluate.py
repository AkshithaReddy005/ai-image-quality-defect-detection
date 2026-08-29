"""
evaluate.py
===========
Standalone evaluation script — runs the trained model against a test set
and produces a comprehensive metrics report including:
  - Accuracy, Precision, Recall, F1 (per class + macro)
  - Confusion matrix
  - ROC-AUC (one-vs-rest)
  - Examples of misclassified images
  - Feature importance chart data

Usage:
  cd backend
  python ml/evaluate.py [--output-dir ml/models]
"""

import sys
import csv
import json
import logging
import argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import joblib
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    accuracy_score,
    f1_score,
)
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split

from app.services.feature_extractor import extract_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DATASET_DIR = Path("data/synthetic_dataset")
IMAGES_DIR  = DATASET_DIR / "images"
LABELS_FILE = DATASET_DIR / "labels.csv"
MODEL_DIR   = Path("ml/models")


def load_models():
    rf     = joblib.load(MODEL_DIR / "quality_rf_model.pkl")
    scaler = joblib.load(MODEL_DIR / "feature_scaler.pkl")
    le     = joblib.load(MODEL_DIR / "label_encoder.pkl")
    return rf, scaler, le


def load_features():
    with LABELS_FILE.open() as f:
        rows = list(csv.DictReader(f))

    X_list, y_list, filenames = [], [], []
    for row in rows:
        fpath = IMAGES_DIR / row["filename"]
        if not fpath.exists():
            continue
        img = cv2.imread(str(fpath))
        if img is None:
            continue
        try:
            fv = extract_features(img)
            X_list.append(fv.to_model_array())
            y_list.append(row["label"])
            filenames.append(row["filename"])
        except Exception:
            pass

    return np.array(X_list, dtype=np.float32), np.array(y_list), filenames


def evaluate(output_dir: str = "ml/models") -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    logger.info("Loading models...")
    rf, scaler, le = load_models()

    logger.info("Loading feature vectors...")
    X, y_str, filenames = load_features()
    y = le.transform(y_str)

    # Use held-out test split (same seed as training)
    _, X_test, _, y_test, _, fn_test = train_test_split(
        X, y, filenames, test_size=0.20, random_state=42, stratify=y
    )

    X_test_s = scaler.transform(X_test)
    y_pred   = rf.predict(X_test_s)
    y_proba  = rf.predict_proba(X_test_s)

    # ── Metrics ───────────────────────────────────────────────────────────────
    acc  = accuracy_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred, average="macro")

    # ROC-AUC (one-vs-rest)
    y_bin = label_binarize(y_test, classes=list(range(len(le.classes_))))
    if y_bin.shape[1] > 1:
        roc_auc = roc_auc_score(y_bin, y_proba, multi_class="ovr", average="macro")
    else:
        roc_auc = roc_auc_score(y_test, y_proba[:, 1])

    report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)
    cm     = confusion_matrix(y_test, y_pred)

    logger.info(f"\n{'='*60}")
    logger.info(f"Accuracy:   {acc:.4f}")
    logger.info(f"F1-macro:   {f1:.4f}")
    logger.info(f"ROC-AUC:    {roc_auc:.4f}")
    logger.info(f"\nClassification Report:")
    logger.info(classification_report(y_test, y_pred, target_names=le.classes_))
    logger.info(f"Confusion Matrix:\n{cm}")

    # ── Failure cases ─────────────────────────────────────────────────────────
    misclassified = [
        {"filename": fn, "true": le.classes_[t], "predicted": le.classes_[p]}
        for fn, t, p in zip(fn_test, y_test, y_pred)
        if t != p
    ]
    logger.info(f"\nMisclassified samples: {len(misclassified)}/{len(y_test)}")
    for mc in misclassified[:5]:
        logger.info(f"  {mc['filename']}: true={mc['true']}, pred={mc['predicted']}")

    # ── Feature importance ────────────────────────────────────────────────────
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
    feat_importance = sorted(
        [{"feature": n, "importance": float(v)} for n, v in zip(feat_names, importances)],
        key=lambda x: -x["importance"],
    )

    # ── Save full report ──────────────────────────────────────────────────────
    full_report = {
        "accuracy":  acc,
        "f1_macro":  f1,
        "roc_auc":   roc_auc,
        "per_class": report,
        "confusion_matrix": cm.tolist(),
        "classes": list(le.classes_),
        "feature_importance": feat_importance,
        "misclassified_samples": misclassified,
        "n_test": int(len(y_test)),
        "n_misclassified": len(misclassified),
    }

    report_path = out / "full_evaluation.json"
    with report_path.open("w") as f:
        json.dump(full_report, f, indent=2)

    txt_path = out / "evaluation_report.txt"
    with txt_path.open("w") as f:
        f.write(f"Accuracy:  {acc:.4f}\n")
        f.write(f"F1-macro:  {f1:.4f}\n")
        f.write(f"ROC-AUC:   {roc_auc:.4f}\n\n")
        f.write(classification_report(y_test, y_pred, target_names=le.classes_))
        f.write(f"\nConfusion Matrix:\n{cm}\n\n")
        f.write(f"Misclassified: {len(misclassified)}/{len(y_test)}\n\n")
        f.write("Feature Importances:\n")
        for fi in feat_importance:
            f.write(f"  {fi['feature']:<25} {fi['importance']:.4f}\n")

    logger.info(f"\nFull report saved to {report_path}")
    logger.info(f"Text report saved to {txt_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained image quality model")
    parser.add_argument("--output-dir", default="ml/models", help="Directory to save evaluation results")
    args = parser.parse_args()
    evaluate(output_dir=args.output_dir)
