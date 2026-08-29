from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete, func
from app.db.models import AnalysisRecord
from typing import Optional


async def create_analysis(db: AsyncSession, record: AnalysisRecord) -> AnalysisRecord:
    db.add(record)
    await db.commit()
    await db.refresh(record)
    return record


async def get_analysis(db: AsyncSession, analysis_id: str) -> Optional[AnalysisRecord]:
    result = await db.execute(
        select(AnalysisRecord).where(AnalysisRecord.id == analysis_id)
    )
    return result.scalar_one_or_none()


async def list_analyses(
    db: AsyncSession, skip: int = 0, limit: int = 20
) -> tuple[list[AnalysisRecord], int]:
    count_result = await db.execute(select(func.count()).select_from(AnalysisRecord))
    total = count_result.scalar_one()
    result = await db.execute(
        select(AnalysisRecord)
        .order_by(AnalysisRecord.analyzed_at.desc())
        .offset(skip)
        .limit(limit)
    )
    return result.scalars().all(), total


async def delete_analysis(db: AsyncSession, analysis_id: str) -> bool:
    result = await db.execute(
        delete(AnalysisRecord).where(AnalysisRecord.id == analysis_id)
    )
    await db.commit()
    return result.rowcount > 0
