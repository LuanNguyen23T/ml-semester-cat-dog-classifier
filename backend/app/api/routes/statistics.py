from datetime import date, datetime, time, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import case, func
from sqlalchemy.orm import Session, joinedload

from app.core.database import get_db
from app.ai.registry import ModelMethod
from app.models.breed import Breed
from app.models.prediction import Prediction, PredictionStatus
from app.models.species import Species
from app.schemas.statistics import StatisticsResponse


router = APIRouter()


def _round_optional(value: Optional[float], digits: int = 4) -> Optional[float]:
    return round(float(value), digits) if value is not None else None


@router.get("/statistics", response_model=StatisticsResponse)
def get_statistics(
    date_from: Optional[date] = Query(default=None),
    date_to: Optional[date] = Query(default=None),
    method: Optional[ModelMethod] = Query(default=None),
    recent_page: int = Query(default=1, ge=1),
    recent_limit: int = Query(default=10, ge=1, le=100),
    db: Session = Depends(get_db),
):
    """Return aggregated prediction metrics for the statistics dashboard."""
    end_date = date_to or date.today()
    start_date = date_from or (end_date - timedelta(days=29))

    if start_date > end_date:
        raise HTTPException(status_code=422, detail="date_from must be on or before date_to")
    if (end_date - start_date).days > 365:
        raise HTTPException(status_code=422, detail="Date range cannot exceed 366 days")

    start_at = datetime.combine(start_date, time.min)
    end_at = datetime.combine(end_date + timedelta(days=1), time.min)
    filters = [Prediction.created_at >= start_at, Prediction.created_at < end_at]
    if method:
        filters.append(Prediction.method == method.value)

    total = db.query(func.count(Prediction.id)).filter(*filters).scalar() or 0
    completed = (
        db.query(func.count(Prediction.id))
        .filter(*filters, Prediction.status == PredictionStatus.COMPLETED)
        .scalar()
        or 0
    )
    failed = (
        db.query(func.count(Prediction.id))
        .filter(*filters, Prediction.status == PredictionStatus.FAILED)
        .scalar()
        or 0
    )
    pending = total - completed - failed
    average_confidence, average_processing_time = (
        db.query(func.avg(Prediction.score), func.avg(Prediction.processing_time))
        .filter(*filters, Prediction.status == PredictionStatus.COMPLETED)
        .one()
    )

    daily_rows = (
        db.query(
            func.date(Prediction.created_at).label("day"),
            func.count(Prediction.id).label("total"),
            func.sum(case((Prediction.status == PredictionStatus.COMPLETED, 1), else_=0)).label("completed"),
            func.sum(case((Prediction.status == PredictionStatus.FAILED, 1), else_=0)).label("failed"),
        )
        .filter(*filters)
        .group_by(func.date(Prediction.created_at))
        .order_by(func.date(Prediction.created_at))
        .all()
    )
    daily_lookup = {
        row.day: {"total": row.total, "completed": row.completed, "failed": row.failed}
        for row in daily_rows
    }
    daily = []
    current_date = start_date
    while current_date <= end_date:
        values = daily_lookup.get(current_date, {"total": 0, "completed": 0, "failed": 0})
        daily.append({"date": current_date, **values})
        current_date += timedelta(days=1)

    method_rows = (
        db.query(
            Prediction.method,
            func.count(Prediction.id).label("total"),
            func.sum(case((Prediction.status == PredictionStatus.COMPLETED, 1), else_=0)).label("completed"),
            func.sum(case((Prediction.status == PredictionStatus.FAILED, 1), else_=0)).label("failed"),
            func.avg(case((Prediction.status == PredictionStatus.COMPLETED, Prediction.score))).label("confidence"),
            func.avg(case((Prediction.status == PredictionStatus.COMPLETED, Prediction.processing_time))).label("processing_time"),
        )
        # Legacy rows without a method still contribute to overall totals,
        # while method breakdowns include every named classification method.
        .filter(*filters, Prediction.method.isnot(None))
        .group_by(Prediction.method)
        .order_by(func.count(Prediction.id).desc())
        .all()
    )
    by_method = [
        {
            "method": row.method,
            "total": row.total,
            "completed": row.completed,
            "failed": row.failed,
            "pending": row.total - row.completed - row.failed,
            "success_rate": (
                round((row.completed / (row.completed + row.failed)) * 100, 2)
                if row.completed + row.failed
                else 0
            ),
            "average_confidence": _round_optional(row.confidence),
            "average_processing_time": _round_optional(row.processing_time),
        }
        for row in method_rows
    ]

    completed_filters = [*filters, Prediction.status == PredictionStatus.COMPLETED]
    species_rows = (
        db.query(Species.name, func.count(Prediction.id).label("count"))
        .join(Breed, Breed.species_id == Species.id)
        .join(Prediction, Prediction.predicted_breed_id == Breed.id)
        .filter(*completed_filters)
        .group_by(Species.name)
        .order_by(func.count(Prediction.id).desc())
        .all()
    )
    breed_rows = (
        db.query(Breed.name, func.count(Prediction.id).label("count"))
        .join(Prediction, Prediction.predicted_breed_id == Breed.id)
        .filter(*completed_filters)
        .group_by(Breed.name)
        .order_by(func.count(Prediction.id).desc())
        .limit(8)
        .all()
    )
    breed_species_rows = (
        db.query(
            Breed.name,
            Species.name.label("species"),
            func.count(Prediction.id).label("count"),
        )
        .join(Species, Breed.species_id == Species.id)
        .join(Prediction, Prediction.predicted_breed_id == Breed.id)
        .filter(*completed_filters)
        .group_by(Breed.name, Species.name)
        .order_by(func.count(Prediction.id).desc())
        .all()
    )

    recent_predictions = (
        db.query(Prediction)
        .options(joinedload(Prediction.predicted_breed).joinedload(Breed.species))
        .filter(*filters)
        .order_by(Prediction.created_at.desc())
        .offset((recent_page - 1) * recent_limit)
        .limit(recent_limit)
        .all()
    )

    return {
        "overview": {
            "total_predictions": total,
            "completed_predictions": completed,
            "failed_predictions": failed,
            "pending_predictions": pending,
            "success_rate": round((completed / (completed + failed)) * 100, 2) if completed + failed else 0,
            "average_confidence": _round_optional(average_confidence),
            "average_processing_time": _round_optional(average_processing_time),
        },
        "daily": daily,
        "by_method": by_method,
        "by_species": [{"name": row.name, "count": row.count} for row in species_rows],
        "by_breed_species": [
            {"name": row.name, "species": row.species, "count": row.count}
            for row in breed_species_rows
        ],
        "top_breeds": [{"name": row.name, "count": row.count} for row in breed_rows],
        "recent_predictions": recent_predictions,
        "recent_total": total,
        "recent_page": recent_page,
        "recent_limit": recent_limit,
    }
