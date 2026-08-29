import time
from fastapi import APIRouter
from app.models.schemas import HealthResponse
from app.config import settings
from app.services import model_inference

router = APIRouter()
_start_time = time.time()


@router.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Health check endpoint — returns app status and model readiness."""
    return HealthResponse(
        status="ok",
        version=settings.APP_VERSION,
        model_loaded=model_inference.is_ready(),
        db_connected=True,
        uptime_seconds=round(time.time() - _start_time, 1),
    )
