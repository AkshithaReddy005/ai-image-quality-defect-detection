from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    # Application
    APP_NAME: str = "Image Quality Detector"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False

    # Database
    DATABASE_URL: str = "sqlite+aiosqlite:///./data/quality.db"

    # Storage
    UPLOAD_DIR: str = "./data/uploads"
    MODEL_DIR: str = "./ml/models"
    HEATMAP_DIR: str = "./data/heatmaps"

    # CORS
    CORS_ORIGINS: list[str] = ["http://localhost:3000", "http://frontend:3000"]

    # Analysis
    MAX_FILE_SIZE_MB: int = 20
    ALLOWED_EXTENSIONS: list[str] = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"]

    # Model
    MODEL_FILENAME: str = "quality_rf_model.pkl"
    SCALER_FILENAME: str = "feature_scaler.pkl"
    LABEL_ENCODER_FILENAME: str = "label_encoder.pkl"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
