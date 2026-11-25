import uuid
from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel
from app.models.prediction import PredictionStatus
from app.schemas.breed import BreedResponse

class PredictionCreateResponse(BaseModel):
    id: uuid.UUID
    batch_id: Optional[uuid.UUID] = None
    status: PredictionStatus
    method: Optional[str] = None

class PredictionResponse(BaseModel):
    model_config = {"protected_namespaces": (), "from_attributes": True}

    id: uuid.UUID
    batch_id: Optional[uuid.UUID] = None
    image_key: str
    predicted_breed: Optional[BreedResponse] = None
    score: Optional[float] = None
    top_k_results: Optional[List[Dict[str, Any]]] = None
    model_version: Optional[str] = None
    method: Optional[str] = None
    processing_time: Optional[float] = None
    status: PredictionStatus
    error_message: Optional[str] = None
    created_at: datetime


class PredictionHistoryGroup(BaseModel):
    group_id: str
    batch_id: Optional[uuid.UUID] = None
    image_key: str
    created_at: datetime
    predictions: List[PredictionResponse]


class PredictionHistoryResponse(BaseModel):
    items: List[PredictionHistoryGroup]
    total: int
    page: int
    limit: int
