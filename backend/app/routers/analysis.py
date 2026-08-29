"""
analysis.py
===========
POST /api/analyze        — single image analysis
POST /api/analyze/batch  — batch analysis (up to 10 images)
GET  /api/analysis/{id}  — retrieve single result
GET  /api/analysis/{id}/heatmap — serve heatmap image
DELETE /api/analysis/{id} — delete a result
"""

import uuid
import time
import logging
from pathlib import Path

import cv2
import numpy as np
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, BackgroundTasks, status
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.db.database import get_db
from app.db.crud import create_analysis, get_analysis, delete_analysis
from app.db.models import AnalysisRecord
from app.models.schemas import AnalysisResult, BatchAnalysisResult
from app.services.feature_extractor import features_from_bytes, extract_features, FeatureVector
from app.services.model_inference import run_inference
from app.services.heatmap import generate_heatmap

logger = logging.getLogger(__name__)
router = APIRouter()

MAX_FILE_BYTES = settings.MAX_FILE_SIZE_MB * 1024 * 1024
ALLOWED_EXT    = set(settings.ALLOWED_EXTENSIONS)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _validate_file(file: UploadFile) -> None:
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_EXT:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"File type '{ext}' not supported. Allowed: {', '.join(ALLOWED_EXT)}",
        )


async def _read_file(file: UploadFile) -> bytes:
    # Read the start of the stream to extract magic bytes (up to 2048 bytes)
    header_bytes = await file.read(2048)
    
    # Reset file pointer so we can read it completely from the beginning
    await file.seek(0)
    
    # Explicit Magic Byte Checking
    is_jpeg = header_bytes.startswith(b'\xff\xd8\xff')
    is_png = header_bytes.startswith(b'\x89\x50\x4e\x47\x0d\x0a\x1a\x0a')
    
    if not (is_jpeg or is_png):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File structure tampered or corrupt. Core binary payload does not match JPEG/PNG magic signatures.",
        )
        
    data = await file.read()
    if len(data) > MAX_FILE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds maximum size of {settings.MAX_FILE_SIZE_MB} MB",
        )
    return data


def _save_upload(data: bytes, analysis_id: str, filename: str) -> str:
    upload_dir = Path(settings.UPLOAD_DIR)
    upload_dir.mkdir(parents=True, exist_ok=True)
    ext = Path(filename).suffix.lower() or ".jpg"
    out_path = upload_dir / f"{analysis_id}{ext}"
    out_path.write_bytes(data)
    return str(out_path)


def _decode_image(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Could not decode image. File may be corrupt or not a valid image.",
        )
    return img


def _build_record(
    analysis_id: str,
    filename: str,
    file_size: int,
    img: np.ndarray,
    fv: FeatureVector,
    result: dict,
    heatmap_path: str | None,
    elapsed_ms: float,
) -> AnalysisRecord:
    h, w = img.shape[:2]
    return AnalysisRecord(
        id=analysis_id,
        filename=filename,
        file_size_bytes=file_size,
        image_width=w,
        image_height=h,
        quality_score=result["quality_score"],
        quality_label=result["quality_label"],
        issues=[i.model_dump() for i in result["issues"]],
        features=fv.to_dict(),
        heatmap_path=heatmap_path,
        processing_time_ms=elapsed_ms,
        explainability_insights=result.get("explainability_insights"),
    )


def _record_to_result(record: AnalysisRecord) -> AnalysisResult:
    from app.models.schemas import IssueDetail, ImageFeatures, ExplainabilityInsights
    issues = [IssueDetail(**i) for i in (record.issues or [])]
    feats  = ImageFeatures(**record.features)
    heatmap_url = f"/api/analysis/{record.id}/heatmap" if record.heatmap_path else None
    
    exp_insights = None
    if record.explainability_insights:
        exp_insights = ExplainabilityInsights(**record.explainability_insights)
        
    return AnalysisResult(
        id=record.id,
        filename=record.filename,
        file_size_bytes=record.file_size_bytes,
        quality_score=record.quality_score,
        quality_label=record.quality_label,
        issues=issues,
        features=feats,
        heatmap_url=heatmap_url,
        analyzed_at=record.analyzed_at,
        processing_time_ms=record.processing_time_ms or 0.0,
        explainability_insights=exp_insights,
    )


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post(
    "/analyze",
    response_model=AnalysisResult,
    status_code=status.HTTP_201_CREATED,
    tags=["Analysis"],
    summary="Analyze a single image for quality issues",
)
async def analyze_image(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
):
    _validate_file(file)
    data = await _read_file(file)

    t0 = time.perf_counter()
    img = _decode_image(data)
    fv  = extract_features(img)
    result = run_inference(fv)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    analysis_id = str(uuid.uuid4())
    _save_upload(data, analysis_id, file.filename or "upload.jpg")

    # Generate heatmap in background
    heatmap_path = generate_heatmap(img, analysis_id)

    record = _build_record(
        analysis_id, file.filename or "upload.jpg", len(data),
        img, fv, result, heatmap_path, elapsed_ms,
    )
    await create_analysis(db, record)
    return _record_to_result(record)


@router.post(
    "/analyze/batch",
    response_model=BatchAnalysisResult,
    status_code=status.HTTP_200_OK,
    tags=["Analysis"],
    summary="Analyze multiple images (up to 10) in one request",
)
async def analyze_batch(
    files: list[UploadFile] = File(...),
    db: AsyncSession = Depends(get_db),
):
    if len(files) > 10:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Maximum 10 files per batch request",
        )

    results: list[AnalysisResult] = []
    failed: list[dict] = []

    for file in files:
        try:
            _validate_file(file)
            data = await _read_file(file)

            t0 = time.perf_counter()
            img = _decode_image(data)
            fv  = extract_features(img)
            result = run_inference(fv)
            elapsed_ms = (time.perf_counter() - t0) * 1000

            analysis_id = str(uuid.uuid4())
            _save_upload(data, analysis_id, file.filename or "upload.jpg")
            heatmap_path = generate_heatmap(img, analysis_id)

            record = _build_record(
                analysis_id, file.filename or "upload.jpg", len(data),
                img, fv, result, heatmap_path, elapsed_ms,
            )
            await create_analysis(db, record)
            results.append(_record_to_result(record))

        except HTTPException as e:
            failed.append({"filename": file.filename, "error": e.detail})
        except Exception as e:
            failed.append({"filename": file.filename, "error": str(e)})

    return BatchAnalysisResult(
        results=results,
        failed=failed,
        total=len(files),
        success_count=len(results),
        failure_count=len(failed),
    )


@router.get(
    "/analysis/{analysis_id}",
    response_model=AnalysisResult,
    tags=["Analysis"],
    summary="Retrieve a single analysis result by ID",
)
async def get_analysis_result(analysis_id: str, db: AsyncSession = Depends(get_db)):
    record = await get_analysis(db, analysis_id)
    if not record:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Analysis not found")
    return _record_to_result(record)


@router.get(
    "/analysis/{analysis_id}/heatmap",
    tags=["Analysis"],
    summary="Serve the saliency heatmap image for a given analysis",
)
async def get_heatmap(analysis_id: str, db: AsyncSession = Depends(get_db)):
    record = await get_analysis(db, analysis_id)
    if not record:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Analysis not found")
    if not record.heatmap_path or not Path(record.heatmap_path).exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Heatmap not available")
    return FileResponse(record.heatmap_path, media_type="image/png")


@router.delete(
    "/analysis/{analysis_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    tags=["Analysis"],
    summary="Delete an analysis record and associated files",
)
async def remove_analysis(analysis_id: str, db: AsyncSession = Depends(get_db)):
    record = await get_analysis(db, analysis_id)
    if not record:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Analysis not found")

    # Clean up files
    for path_attr in ("heatmap_path",):
        p = getattr(record, path_attr, None)
        if p and Path(p).exists():
            Path(p).unlink(missing_ok=True)

    # Clean up upload
    upload_dir = Path(settings.UPLOAD_DIR)
    for ext in settings.ALLOWED_EXTENSIONS:
        f = upload_dir / f"{analysis_id}{ext}"
        f.unlink(missing_ok=True)

    deleted = await delete_analysis(db, analysis_id)
    if not deleted:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Deletion failed")
