from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class IssueDetail(BaseModel):
    type: str
    severity: str  # "low" | "medium" | "high" | "critical"
    confidence: float = Field(ge=0.0, le=1.0)
    description: str


class ImageFeatures(BaseModel):
    # Sharpness
    laplacian_variance: float
    sobel_mean: float
    sobel_std: float
    tenengrad_score: float
    sharpness_score: float

    # Exposure
    brightness_mean: float
    brightness_std: float
    underexposed_ratio: float
    overexposed_ratio: float
    histogram_entropy: float

    # Noise
    noise_estimate: float
    snr_db: float

    # Contrast
    rms_contrast: float
    michelson_contrast: float

    # Color / Texture
    saturation_mean: float
    saturation_std: float
    colorfulness_index: float
    glcm_energy: float
    glcm_homogeneity: float
    glcm_correlation: float

    # Corruption
    jpeg_artifact_score: float
    blocking_score: float

    # Dimensions
    width: int
    height: int
    channels: int
    aspect_ratio: float


class ExplainabilityInsights(BaseModel):
    primary_decision_driver: str
    feature_influence_weight: float
    structural_reasoning: str


class AnalysisResult(BaseModel):
    id: str
    filename: str
    file_size_bytes: int
    quality_score: float
    quality_label: str
    issues: list[IssueDetail]
    features: ImageFeatures
    heatmap_url: Optional[str] = None
    analyzed_at: datetime
    processing_time_ms: float
    explainability_insights: Optional[ExplainabilityInsights] = None


class AnalysisSummary(BaseModel):
    id: str
    filename: str
    quality_score: float
    quality_label: str
    issue_count: int
    analyzed_at: datetime
    thumbnail_url: Optional[str] = None


class HistoryResponse(BaseModel):
    items: list[AnalysisSummary]
    total: int
    page: int
    page_size: int
    pages: int


class BatchAnalysisResult(BaseModel):
    results: list[AnalysisResult]
    failed: list[dict]
    total: int
    success_count: int
    failure_count: int


class HealthResponse(BaseModel):
    status: str
    version: str
    model_loaded: bool
    db_connected: bool
    uptime_seconds: float
