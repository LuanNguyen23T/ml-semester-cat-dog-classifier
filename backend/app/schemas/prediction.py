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
    note: Optional[str] = None

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
    note: Optional[str] = None
    error_message: Optional[str] = None
    created_at: datetime
    is_unknown_breed: Optional[bool] = False
    unknown_score: Optional[float] = None
    breed_status: Optional[str] = "known"
    unknown_assessment: Optional[Dict[str, Any]] = None


class PredictionHistoryGroup(BaseModel):
    group_id: str
    batch_id: Optional[uuid.UUID] = None
    image_key: str
    note: Optional[str] = None
    created_at: datetime
    predictions: List[PredictionResponse]


class PredictionHistoryResponse(BaseModel):
    items: List[PredictionHistoryGroup]
    total: int
    page: int
    limit: int
