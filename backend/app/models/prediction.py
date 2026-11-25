import uuid
import datetime
from sqlalchemy import Column, String, Float, ForeignKey, DateTime, Enum, Text, Integer, JSON
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship
import enum
from app.core.database import Base

class PredictionStatus(str, enum.Enum):
    UPLOADED = "UPLOADED"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"

class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(UUID(as_uuid=True).with_variant(String(36), "sqlite"), primary_key=True, default=uuid.uuid4, index=True)
    batch_id = Column(UUID(as_uuid=True).with_variant(String(36), "sqlite"), nullable=True, index=True)
    image_key = Column(String, nullable=False)
    predicted_breed_id = Column(Integer, ForeignKey("breeds.id"), nullable=True)
    score = Column(Float, nullable=True)
    top_k_results = Column(JSONB().with_variant(JSON, "sqlite"), nullable=True)
    model_version = Column(String, nullable=True)
    method = Column(String, nullable=True)  # ModelMethod value: resnet18 | seresnet18 | densenet121 | efficientnet-b0 | dual-stream-efficientnet
    processing_time = Column(Float, nullable=True) # in seconds
    status = Column(Enum(PredictionStatus), default=PredictionStatus.UPLOADED, nullable=False)
    note = Column(String, nullable=True)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    predicted_breed = relationship("Breed", back_populates="predictions")

    @property
    def is_unknown_breed(self) -> bool:
        if not self.top_k_results:
            return False
        for item in self.top_k_results:
            if isinstance(item, dict) and "_unknown_assessment" in item:
                return item["_unknown_assessment"].get("is_unknown", False)
        return False

    @property
    def unknown_score(self) -> float:
        if not self.top_k_results:
            return 0.0
        for item in self.top_k_results:
            if isinstance(item, dict) and "_unknown_assessment" in item:
                return float(item["_unknown_assessment"].get("unknown_score", 0.0))
        return 0.0

    @property
    def breed_status(self) -> str:
        return "unknown" if self.is_unknown_breed else "known"

    @property
    def unknown_assessment(self) -> dict | None:
        if not self.top_k_results:
            return None
        for item in self.top_k_results:
            if isinstance(item, dict) and "_unknown_assessment" in item:
                return item["_unknown_assessment"]
        return None

