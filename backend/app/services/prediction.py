"""Background inference services for single and batch prediction requests."""

from __future__ import annotations

import time
import traceback
import urllib.request
from pathlib import Path
from typing import List, Tuple
from uuid import UUID

from sqlalchemy.orm import Session

from app.ai.base import Classifier, ClassifierResult
from app.ai.common import load_rgb_image
from app.ai.preprocessing import prepare_batch_inputs
from app.ai.registry import ClassifierRegistry, ModelMethod
from app.models.prediction import Prediction, PredictionStatus
from app.repositories import prediction as prediction_repo


def _download_image_bytes(image_url: str) -> bytes:
    """Fetch image bytes from URL or local static upload path."""
    if "/static/uploads/" in image_url:
        filename = image_url.split("/static/uploads/")[-1]
        local_path = Path(__file__).resolve().parent.parent.parent / "static" / "uploads" / filename
        if local_path.exists():
            return local_path.read_bytes()

    req = urllib.request.Request(image_url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as response:
        return response.read()


def _save_completed_prediction(
    db: Session,
    db_obj: Prediction,
    result: ClassifierResult,
    processing_time: float,
) -> None:
    top_k_dicts = [item.model_dump() for item in result.top_k_results]
    if result.unknown_assessment:
        top_k_dicts.append({"_unknown_assessment": result.unknown_assessment})

    prediction_repo.update_prediction_status(
        db,
        db_obj,
        status=PredictionStatus.COMPLETED,
        predicted_breed_id=result.predicted_breed_id,
        score=result.score,
        top_k_results=top_k_dicts,
        model_version=result.model_version,
        processing_time=processing_time,
    )


def _save_failed_prediction(db: Session, db_obj: Prediction, exc: Exception) -> None:
    error_msg = traceback.format_exc()
    prediction_repo.update_prediction_status(
        db,
        db_obj,
        status=PredictionStatus.FAILED,
        error_message=f"{exc}\n{error_msg}",
    )


def run_inference_task(
    db: Session,
    prediction_id: UUID,
    classifier: Classifier,
    use_tta: bool = False,
) -> None:
    """Background task to run single-model inference."""
    db_obj = prediction_repo.get_prediction(db, prediction_id)
    if not db_obj:
        return

    try:
        prediction_repo.update_prediction_status(db, db_obj, PredictionStatus.PROCESSING)
        image_bytes = _download_image_bytes(db_obj.image_key)

        t0 = time.perf_counter()
        pil_img = load_rgb_image(image_bytes)
        result = classifier.predict(pil_img, use_tta=use_tta)
        processing_time = time.perf_counter() - t0

        _save_completed_prediction(db, db_obj, result, processing_time)
    except Exception as exc:
        _save_failed_prediction(db, db_obj, exc)


def run_batch_inference_task(
    batch_items: List[Tuple[UUID, ModelMethod]],
    image_url: str,
    use_tta: bool = False,
) -> None:
    """Run all models on one image, executing DeepLab segmentation only once."""
    from app.core.database import SessionLocal

    db = SessionLocal()
    try:
        try:
            image_bytes = _download_image_bytes(image_url)
            pil_img = load_rgb_image(image_bytes)
            t_seg_start = time.perf_counter()
            prepared = prepare_batch_inputs(pil_img)
            seg_time = time.perf_counter() - t_seg_start
        except Exception as exc:
            for prediction_id, _ in batch_items:
                db_obj = prediction_repo.get_prediction(db, prediction_id)
                if db_obj:
                    _save_failed_prediction(db, db_obj, exc)
            return

        for idx, (prediction_id, method) in enumerate(batch_items):
            db_obj = prediction_repo.get_prediction(db, prediction_id)
            if not db_obj:
                continue
            try:
                prediction_repo.update_prediction_status(db, db_obj, PredictionStatus.PROCESSING)
                if not ClassifierRegistry.is_available(method):
                    raise FileNotFoundError(
                        f"Model '{method.value}' is unavailable (missing weights file)."
                    )
                classifier = ClassifierRegistry.get(method)

                t0 = time.perf_counter()
                result = classifier.predict_from_preprocessed(prepared, use_tta=use_tta)
                clf_time = time.perf_counter() - t0
                # Attribute the one-time segmentation cost to the first model or total cleanly.
                total_time = (seg_time + clf_time) if idx == 0 else clf_time

                _save_completed_prediction(db, db_obj, result, total_time)
            except Exception as exc:
                _save_failed_prediction(db, db_obj, exc)
    finally:
        db.close()
