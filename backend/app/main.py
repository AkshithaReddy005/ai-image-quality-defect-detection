"""
main.py
=======
FastAPI application entry point.
"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.db.database import init_db
from app.services import model_inference
from app.routers import health, analysis, history

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ──────────────────────────────────────────────────────────────
    logger.info("Starting up Image Quality Detector API...")

    # Ensure data directories exist
    for d in [settings.UPLOAD_DIR, settings.HEATMAP_DIR, "data"]:
        Path(d).mkdir(parents=True, exist_ok=True)

    # Initialize database
    await init_db()
    logger.info("Database initialized")

    # Load ML models
    loaded = model_inference.load_models()
    if loaded:
        logger.info("ML models loaded successfully")
    else:
        logger.warning(
            "ML models not found — running in rule-based-only mode. "
            "Run `python ml/train.py` to train and save the model."
        )

    yield

    # ── Shutdown ─────────────────────────────────────────────────────────────
    logger.info("Shutting down...")


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=(
        "AI-powered image quality and defect detection API. "
        "Analyzes images for blur, exposure issues, noise, and other quality problems."
    ),
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
)

# ── CORS ─────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(health.router, prefix="/api")
app.include_router(analysis.router, prefix="/api")
app.include_router(history.router, prefix="/api")

# ── Static files (uploaded images served for frontend preview) ────────────────
Path(settings.UPLOAD_DIR).mkdir(parents=True, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=settings.UPLOAD_DIR), name="uploads")
