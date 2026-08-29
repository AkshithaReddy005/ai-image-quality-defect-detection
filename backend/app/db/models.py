import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, String, Float, Integer, DateTime, JSON, Text
from sqlalchemy.dialects.sqlite import TEXT
from app.db.database import Base


class AnalysisRecord(Base):
    __tablename__ = "analysis_records"

    id = Column(TEXT, primary_key=True, default=lambda: str(uuid.uuid4()))
    filename = Column(String(255), nullable=False)
    file_size_bytes = Column(Integer, nullable=False)
    image_width = Column(Integer, nullable=True)
    image_height = Column(Integer, nullable=True)
    quality_score = Column(Float, nullable=False)
    quality_label = Column(String(20), nullable=False)
    issues = Column(JSON, nullable=False, default=list)
    features = Column(JSON, nullable=False, default=dict)
    heatmap_path = Column(String(512), nullable=True)
    analyzed_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    processing_time_ms = Column(Float, nullable=True)
