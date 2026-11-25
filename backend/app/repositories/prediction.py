from typing import List, Optional, Any
from uuid import UUID
from sqlalchemy.orm import Session
from app.models.prediction import Prediction, PredictionStatus

def create_prediction(
    db: Session,
    image_key: str,
    method: Optional[str] = None,
    batch_id: Optional[UUID] = None,
    note: Optional[str] = None,
) -> Prediction:
    db_obj = Prediction(
        image_key=image_key,
        method=method,
        batch_id=batch_id,
        note=note,
        status=PredictionStatus.UPLOADED
    )
    db.add(db_obj)
    db.commit()
    db.refresh(db_obj)
    return db_obj


def get_prediction(db: Session, prediction_id: UUID) -> Optional[Prediction]:
    return db.query(Prediction).filter(Prediction.id == prediction_id).first()

def get_predictions(
    db: Session, 
    status: Optional[PredictionStatus] = None, 
    breed_id: Optional[int] = None,
    method: Optional[str] = None,
    skip: int = 0, 
    limit: int = 100
) -> List[Prediction]:
    query = db.query(Prediction)
    if status:
        query = query.filter(Prediction.status == status)
    if breed_id:
        query = query.filter(Prediction.predicted_breed_id == breed_id)
    if method:
        query = query.filter(Prediction.method == method)
    
    return query.order_by(Prediction.created_at.desc()).offset(skip).limit(limit).all()


def update_prediction_status(
    db: Session, 
    db_obj: Prediction, 
    status: PredictionStatus, 
    predicted_breed_id: Optional[int] = None,
    score: Optional[float] = None,
    top_k_results: Optional[Any] = None,
    model_version: Optional[str] = None,
    method: Optional[str] = None,
    error_message: Optional[str] = None,
    processing_time: Optional[float] = None
) -> Prediction:
    db_obj.status = status
    if predicted_breed_id is not None:
        db_obj.predicted_breed_id = predicted_breed_id
    if score is not None:
        db_obj.score = score
    if top_k_results is not None:
        db_obj.top_k_results = top_k_results
    if model_version is not None:
        db_obj.model_version = model_version
    if method is not None:
        db_obj.method = method
    if error_message is not None:
        db_obj.error_message = error_message
    if processing_time is not None:
        db_obj.processing_time = processing_time
        
    db.commit()
    db.refresh(db_obj)
    return db_obj


def delete_prediction(db: Session, prediction_id: UUID, delete_batch: bool = True) -> int:
    pred = db.query(Prediction).filter(Prediction.id == prediction_id).first()
    if not pred:
        return 0
    if pred.batch_id and delete_batch:
        deleted = db.query(Prediction).filter(Prediction.batch_id == pred.batch_id).delete()
    else:
        db.delete(pred)
        deleted = 1
    db.commit()
    return deleted
