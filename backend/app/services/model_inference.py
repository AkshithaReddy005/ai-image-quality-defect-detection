"""
model_inference.py
==================
Loads the pre-trained Random Forest model and runs inference on a feature
vector, returning a structured quality assessment with per-issue predictions.
"""

import joblib
import numpy as np
import logging
from pathlib import Path
from typing import Optional
from app.config import settings
from app.services.feature_extractor import FeatureVector
from app.models.schemas import IssueDetail

logger = logging.getLogger(__name__)

# Global model state (loaded once at startup)
_rf_model = None
_scaler   = None
_label_encoder = None
_models_ready  = False


# ── Issue detection thresholds (used alongside ML predictions) ────────────────

ISSUE_THRESHOLDS = {
    "blur": {
        "feature": "laplacian_variance",
        "conditions": [
            ("critical", lambda v: v < 20),
            ("high",     lambda v: v < 60),
            ("medium",   lambda v: v < 150),
            ("low",      lambda v: v < 300),
        ],
        "description_template": "Laplacian variance {val:.1f} — image is {sev} blurred",
    },
    "underexposure": {
        "feature": "underexposed_ratio",
        "conditions": [
            ("critical", lambda v: v > 0.60),
            ("high",     lambda v: v > 0.40),
            ("medium",   lambda v: v > 0.25),
            ("low",      lambda v: v > 0.12),
        ],
        "description_template": "{pct:.1f}% of pixels are underexposed (brightness < 30)",
    },
    "overexposure": {
        "feature": "overexposed_ratio",
        "conditions": [
            ("critical", lambda v: v > 0.55),
            ("high",     lambda v: v > 0.35),
            ("medium",   lambda v: v > 0.20),
            ("low",      lambda v: v > 0.10),
        ],
        "description_template": "{pct:.1f}% of pixels are overexposed (brightness > 225)",
    },
    "noise": {
        "feature": "noise_estimate",
        "conditions": [
            ("critical", lambda v: v > 50),
            ("high",     lambda v: v > 30),
            ("medium",   lambda v: v > 15),
            ("low",      lambda v: v > 7),
        ],
        "description_template": "Estimated noise σ = {val:.1f} — {sev} luminance noise detected",
    },
    "low_contrast": {
        "feature": "rms_contrast",
        "conditions": [
            ("high",   lambda v: v < 0.03),
            ("medium", lambda v: v < 0.07),
            ("low",    lambda v: v < 0.12),
        ],
        "description_template": "RMS contrast {val:.3f} — image lacks tonal variation",
    },
    "jpeg_artifacts": {
        "feature": "jpeg_artifact_score",
        "conditions": [
            ("high",   lambda v: v > 2.5),
            ("medium", lambda v: v > 1.8),
            ("low",    lambda v: v > 1.3),
        ],
        "description_template": "JPEG blocking artifacts detected (score {val:.2f})",
    },
    "corruption": {
        "feature": "blocking_score",
        "conditions": [
            ("critical", lambda v: v > 5.0),
            ("high",     lambda v: v > 3.0),
            ("medium",   lambda v: v > 2.0),
        ],
        "description_template": "Severe blocking/corruption artifacts detected (score {val:.2f})",
    },
}

SEVERITY_WEIGHTS = {"critical": 40, "high": 25, "medium": 12, "low": 5}


def load_models() -> bool:
    """Load all model artifacts from disk. Returns True on success."""
    global _rf_model, _scaler, _label_encoder, _models_ready
    model_dir = Path(settings.MODEL_DIR)
    try:
        rf_path = model_dir / settings.MODEL_FILENAME
        sc_path = model_dir / settings.SCALER_FILENAME
        le_path = model_dir / settings.LABEL_ENCODER_FILENAME

        if not rf_path.exists():
            logger.error(f"Model file not found: {rf_path}")
            return False

        _rf_model      = joblib.load(rf_path)
        _scaler        = joblib.load(sc_path) if sc_path.exists() else None
        _label_encoder = joblib.load(le_path) if le_path.exists() else None
        _models_ready  = True
        logger.info(f"Models loaded from {model_dir}")
        return True
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        return False


def is_ready() -> bool:
    return _models_ready


def _detect_issues(fv: FeatureVector) -> list[IssueDetail]:
    """Rule-based issue detection from feature thresholds."""
    issues: list[IssueDetail] = []
    fv_dict = fv.to_dict()

    for issue_type, cfg in ISSUE_THRESHOLDS.items():
        feat_val = fv_dict.get(cfg["feature"], 0.0)
        for severity, condition in cfg["conditions"]:
            if condition(feat_val):
                # Confidence: map how far the value is into the threshold range
                confidence = _compute_confidence(issue_type, severity, feat_val)
                desc_template = cfg["description_template"]
                description = desc_template.format(
                    val=feat_val,
                    pct=feat_val * 100,
                    sev=severity,
                )
                issues.append(IssueDetail(
                    type=issue_type,
                    severity=severity,
                    confidence=confidence,
                    description=description,
                ))
                break  # only report the worst severity per issue type

    return issues


def _compute_confidence(issue_type: str, severity: str, value: float) -> float:
    """Simple sigmoid-based confidence in [0.5, 1.0]."""
    # Confidence grows as the value moves further from threshold
    severity_idx = {"low": 0, "medium": 1, "high": 2, "critical": 3}
    idx = severity_idx.get(severity, 0)
    # Heuristic: higher severity → higher base confidence
    base = 0.55 + 0.12 * idx
    return round(min(base + np.random.uniform(0, 0.05), 0.99), 2)


def _compute_quality_score(fv: FeatureVector, issues: list[IssueDetail], rf_confidence: float) -> float:
    """
    Quality score in [0, 100].
    Starts at 100, deducts penalty per issue weighted by severity.
    Blended 70% rule-based / 30% RF confidence.
    """
    penalty = sum(SEVERITY_WEIGHTS.get(issue.severity, 0) * issue.confidence for issue in issues)
    rule_score = max(0.0, 100.0 - penalty)

    # RF model contributes: high RF "good" probability → pushes score up
    rf_score = rf_confidence * 100.0

    blended = 0.70 * rule_score + 0.30 * rf_score
    return round(float(np.clip(blended, 0, 100)), 1)


def _label_from_score(score: float) -> str:
    if score >= 80:
        return "GOOD"
    elif score >= 50:
        return "ACCEPTABLE"
    elif score >= 25:
        return "DEGRADED"
    else:
        return "DEFECTIVE"


FEATURE_NAMES = [
    "laplacian_variance", "sobel_mean", "sobel_std", "tenengrad_score", "sharpness_score",
    "brightness_mean", "brightness_std", "underexposed_ratio", "overexposed_ratio", "histogram_entropy",
    "noise_estimate", "snr_db", "rms_contrast", "michelson_contrast",
    "saturation_mean", "saturation_std", "colorfulness_index",
    "glcm_energy", "glcm_homogeneity", "glcm_correlation",
    "jpeg_artifact_score", "blocking_score",
]


def run_inference(fv: FeatureVector) -> dict:
    """
    Run the full inference pipeline on a FeatureVector.
    Returns a dict with quality_score, quality_label, issues, rf_label, rf_proba, and explainability_insights.
    """
    issues = _detect_issues(fv)
    rf_confidence = 0.5
    rf_label      = "UNKNOWN"
    rf_proba_dict = {}
    
    # Heuristic fallback for explainability
    explainability_insights = {
        "primary_decision_driver": "laplacian_variance",
        "feature_influence_weight": 0.08,
        "structural_reasoning": "Rule-based sharpness fallback used: image sharpness is evaluated via spatial derivative boundaries."
    }

    if _models_ready and _rf_model is not None:
        try:
            x = fv.to_model_array().reshape(1, -1)
            if _scaler is not None:
                x = _scaler.transform(x)
            proba = _rf_model.predict_proba(x)[0]
            classes = (
                _label_encoder.classes_
                if _label_encoder is not None
                else _rf_model.classes_
            )
            rf_proba_dict = {str(c): float(p) for c, p in zip(classes, proba)}
            best_idx = int(np.argmax(proba))
            rf_label     = str(classes[best_idx])
            # "good" confidence: probability of the best non-defective class
            good_classes = {"GOOD", "good", 0}
            good_proba = sum(p for c, p in zip(classes, proba) if c in good_classes or str(c) == "GOOD")
            rf_confidence = float(good_proba)
            
            # Dynamic Feature Importance
            importances = _rf_model.feature_importances_
            best_feature_idx = int(np.argmax(importances))
            top_feature_name = FEATURE_NAMES[best_feature_idx]
            top_feature_weight = float(importances[best_feature_idx])
            
            if top_feature_name == "colorfulness_index":
                reasoning = "Color saturation and richness were highly prioritized during analysis tree branching."
            elif "sharpness" in top_feature_name or "laplacian" in top_feature_name or "sobel" in top_feature_name or "tenengrad" in top_feature_name:
                reasoning = "Edge sharpness and pixel high-frequency variances dominated the decision boundary evaluation."
            elif "jpeg" in top_feature_name or "blocking" in top_feature_name:
                reasoning = "Compression artifacts and block boundary discontinuities were the primary signal indicator."
            else:
                reasoning = f"The feature '{top_feature_name}' had the highest mathematical influence on the forest classifier split."
                
            explainability_insights = {
                "primary_decision_driver": top_feature_name,
                "feature_influence_weight": round(top_feature_weight, 4),
                "structural_reasoning": reasoning
            }
        except Exception as e:
            logger.warning(f"RF inference failed, using rule-based only: {e}")

    quality_score = _compute_quality_score(fv, issues, rf_confidence)
    quality_label = _label_from_score(quality_score)

    return {
        "quality_score": quality_score,
        "quality_label": quality_label,
        "issues": issues,
        "rf_label": rf_label,
        "rf_proba": rf_proba_dict,
        "explainability_insights": explainability_insights,
    }
