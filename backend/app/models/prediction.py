import uuid
import datetime
from sqlalchemy import Column, String, Float, ForeignKey, DateTime, Enum, Text, Integer
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

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    batch_id = Column(UUID(as_uuid=True), nullable=True, index=True)
    image_key = Column(String, nullable=False)
    predicted_breed_id = Column(Integer, ForeignKey("breeds.id"), nullable=True)
    score = Column(Float, nullable=True)
    top_k_results = Column(JSONB, nullable=True)
    model_version = Column(String, nullable=True)
    method = Column(String, nullable=True)  # ModelMethod value: resnet18 | seresnet18 | densenet121 | efficientnet-b0 | dual-stream-efficientnet
    processing_time = Column(Float, nullable=True) # in seconds
    status = Column(Enum(PredictionStatus), default=PredictionStatus.UPLOADED, nullable=False)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    predicted_breed = relationship("Breed", back_populates="predictions")
