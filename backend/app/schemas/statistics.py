from datetime import date
from typing import List, Optional

from pydantic import BaseModel

from app.schemas.prediction import PredictionResponse


class StatisticsOverview(BaseModel):
    total_predictions: int
    completed_predictions: int
    failed_predictions: int
    pending_predictions: int
    success_rate: float
    average_confidence: Optional[float] = None
    average_processing_time: Optional[float] = None


class DailyStatistics(BaseModel):
    date: date
    total: int
    completed: int
    failed: int


class MethodStatistics(BaseModel):
    method: str
    total: int
    completed: int
    failed: int
    pending: int
    success_rate: float
    average_confidence: Optional[float] = None
    average_processing_time: Optional[float] = None


class CategoryStatistics(BaseModel):
    name: str
    count: int


class BreedSpeciesStatistics(CategoryStatistics):
    species: str


class StatisticsResponse(BaseModel):
    overview: StatisticsOverview
    daily: List[DailyStatistics]
    by_method: List[MethodStatistics]
    by_species: List[CategoryStatistics]
    by_breed_species: List[BreedSpeciesStatistics]
    top_breeds: List[CategoryStatistics]
    recent_predictions: List[PredictionResponse]
    recent_total: int
    recent_page: int
    recent_limit: int
