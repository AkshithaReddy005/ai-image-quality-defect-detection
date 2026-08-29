from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession
from app.db.database import get_db
from app.db.crud import list_analyses
from app.models.schemas import HistoryResponse, AnalysisSummary
import math

router = APIRouter()


@router.get(
    "/history",
    response_model=HistoryResponse,
    tags=["History"],
    summary="Paginated list of previous analysis results",
)
async def get_history(
    page: int = Query(1, ge=1, description="Page number (1-indexed)"),
    page_size: int = Query(20, ge=1, le=100, description="Results per page"),
    db: AsyncSession = Depends(get_db),
):
    skip = (page - 1) * page_size
    records, total = await list_analyses(db, skip=skip, limit=page_size)
    pages = max(1, math.ceil(total / page_size))

    items = [
        AnalysisSummary(
            id=r.id,
            filename=r.filename,
            quality_score=r.quality_score,
            quality_label=r.quality_label,
            issue_count=len(r.issues or []),
            analyzed_at=r.analyzed_at,
        )
        for r in records
    ]

    return HistoryResponse(
        items=items,
        total=total,
        page=page,
        page_size=page_size,
        pages=pages,
    )
