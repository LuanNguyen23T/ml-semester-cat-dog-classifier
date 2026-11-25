import io
from typing import List, Optional
from uuid import UUID, uuid4

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile
from PIL import Image as PILImage
from PIL import ImageOps, UnidentifiedImageError
from sqlalchemy import func
from sqlalchemy.orm import Session, joinedload

from app.ai.registry import ClassifierRegistry, ModelMethod
from app.core.database import get_db
from app.models.breed import Breed
from app.models.prediction import Prediction, PredictionStatus
from app.repositories import prediction as prediction_repo
from app.schemas.prediction import (
    PredictionCreateResponse,
    PredictionHistoryResponse,
    PredictionResponse,
)
from app.services.prediction import run_batch_inference_task, run_inference_task
from app.storage.cloudinary_storage import upload_image_to_cloudinary

router = APIRouter()

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/jpg", "image/png", "image/webp"}
ALLOWED_PIL_FORMATS = {"JPEG", "PNG", "WEBP"}
MAX_IMAGE_BYTES = 10 * 1024 * 1024  # 10 MB
MAX_IMAGE_DIMENSION = 8000  # px


async def _read_and_validate_image(file: UploadFile) -> bytes:
    """Validate file MIME type, size (<=10MB), Pillow decode, and max dimension (<=8000px)."""
    content_type = (file.content_type or "").lower()
    if content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Only JPG, PNG, and WebP images are allowed.",
        )

    file_content = await file.read()
    if not file_content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(file_content) > MAX_IMAGE_BYTES:
        raise HTTPException(
            status_code=413,
            detail="Image size exceeds the 10 MB limit.",
        )

    try:
        with PILImage.open(io.BytesIO(file_content)) as opened:
            if opened.format not in ALLOWED_PIL_FORMATS:
                raise HTTPException(
                    status_code=400,
                    detail="Unsupported image format. Only JPEG, PNG, and WebP are allowed.",
                )
            transposed = ImageOps.exif_transpose(opened)
            transposed.load()
            width, height = transposed.size
    except HTTPException:
        raise
    except (UnidentifiedImageError, OSError, SyntaxError) as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid or corrupted image file: {exc}",
        ) from exc

    if width <= 0 or height <= 0 or max(width, height) > MAX_IMAGE_DIMENSION:
        raise HTTPException(
            status_code=400,
            detail=f"Image dimensions must not exceed {MAX_IMAGE_DIMENSION} px.",
        )

    return file_content


@router.post("/predictions", response_model=PredictionCreateResponse)
async def create_prediction(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    method: ModelMethod = ModelMethod.DUAL_STREAM_EFFICIENTNET,
    tta: bool = False,
    note: Optional[str] = Form(None),
    db: Session = Depends(get_db),
):
    """Upload an image and start background classification."""
    if not ClassifierRegistry.is_available(method):
        raise HTTPException(
            status_code=503,
            detail=f"Model '{method.value}' is unavailable because its weights file is missing.",
        )

    file_content = await _read_and_validate_image(file)
    image_url = upload_image_to_cloudinary(file_content, file.filename or "upload.jpg")

    prediction = prediction_repo.create_prediction(
        db, image_key=image_url, method=method.value, note=note
    )
    background_tasks.add_task(background_task_runner, prediction.id, method, tta)
    return prediction


@router.post("/predictions/batch", response_model=List[PredictionCreateResponse])
async def create_prediction_batch(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    tta: bool = False,
    note: Optional[str] = Form(None),
    db: Session = Depends(get_db),
):
    """Upload an image and start background classification for all 5 models (shared segmentation)."""
    file_content = await _read_and_validate_image(file)
    image_url = upload_image_to_cloudinary(file_content, file.filename or "upload.jpg")

    predictions = []
    batch_items = []
    batch_id = uuid4()
    for method in ModelMethod:
        prediction = prediction_repo.create_prediction(
            db,
            image_key=image_url,
            method=method.value,
            batch_id=batch_id,
            note=note,
        )
        predictions.append(prediction)
        batch_items.append((prediction.id, method))

    background_tasks.add_task(run_batch_inference_task, batch_items, image_url, tta)
    return predictions


def background_task_runner(prediction_id: UUID, method: ModelMethod, use_tta: bool = False) -> None:
    """Create a fresh DB session for the single-model background task."""
    from app.core.database import SessionLocal

    db = SessionLocal()
    try:
        classifier = ClassifierRegistry.get(method)
        run_inference_task(db, prediction_id, classifier, use_tta=use_tta)
    finally:
        db.close()


@router.get("/predictions/{prediction_id}", response_model=PredictionResponse)
def get_prediction(prediction_id: UUID, db: Session = Depends(get_db)):
    """Get status and result of a prediction."""
    prediction = prediction_repo.get_prediction(db, prediction_id)
    if not prediction:
        raise HTTPException(status_code=404, detail="Prediction not found")
    return prediction


@router.delete("/predictions/{prediction_id}")
def delete_prediction(
    prediction_id: UUID,
    delete_batch: bool = True,
    db: Session = Depends(get_db),
):
    """Delete a prediction or its entire batch."""
    deleted = prediction_repo.delete_prediction(db, prediction_id, delete_batch=delete_batch)
    if not deleted:
        raise HTTPException(status_code=404, detail="Prediction not found")
    return {"message": "Deleted successfully", "count": deleted}


@router.get("/predictions", response_model=List[PredictionResponse])
def get_predictions(
    status: Optional[PredictionStatus] = None,
    breed_id: Optional[int] = None,
    method: Optional[ModelMethod] = None,
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
):
    """Get history of predictions."""
    return prediction_repo.get_predictions(
        db,
        status=status,
        breed_id=breed_id,
        method=method.value if method else None,
        skip=skip,
        limit=limit,
    )


@router.get("/prediction-history", response_model=PredictionHistoryResponse)
def get_prediction_history(
    status: Optional[PredictionStatus] = None,
    method: Optional[ModelMethod] = None,
    page: int = 1,
    limit: int = 12,
    db: Session = Depends(get_db),
):
    """Return history paginated by classification request, not model run."""
    if page < 1:
        raise HTTPException(status_code=422, detail="page must be at least 1")
    if limit < 1 or limit > 100:
        raise HTTPException(status_code=422, detail="limit must be between 1 and 100")

    group_key = func.coalesce(Prediction.batch_id, Prediction.id)
    matching_filters = []
    if status:
        matching_filters.append(Prediction.status == status)
    if method:
        matching_filters.append(Prediction.method == method.value)

    group_query = (
        db.query(
            group_key.label("group_id"),
            func.max(Prediction.created_at).label("created_at"),
        )
        .filter(*matching_filters)
        .group_by(group_key)
    )
    total = group_query.count()
    group_rows = (
        group_query
        .order_by(func.max(Prediction.created_at).desc())
        .offset((page - 1) * limit)
        .limit(limit)
        .all()
    )
    group_ids = [row.group_id for row in group_rows]

    predictions = []
    if group_ids:
        predictions = (
            db.query(Prediction)
            .options(joinedload(Prediction.predicted_breed).joinedload(Breed.species))
            .filter(group_key.in_(group_ids))
            .order_by(Prediction.created_at.desc())
            .all()
        )

    grouped = {
        str(row.group_id): {
            "group_id": str(row.group_id),
            "batch_id": None,
            "image_key": "",
            "note": None,
            "created_at": row.created_at,
            "predictions": [],
        }
        for row in group_rows
    }
    for prediction in predictions:
        group = grouped[str(prediction.batch_id or prediction.id)]
        group["batch_id"] = prediction.batch_id
        group["image_key"] = prediction.image_key
        if getattr(prediction, "note", None) and not group["note"]:
            group["note"] = prediction.note
        group["predictions"].append(prediction)

    groups = list(grouped.values())
    return {
        "items": groups,
        "total": total,
        "page": page,
        "limit": limit,
    }
