from typing import List, Dict, Any
from fastapi import APIRouter
from app.ai.registry import ClassifierRegistry

router = APIRouter()

@router.get("/models", response_model=List[Dict[str, Any]])
def get_models():
    """Get list of available classifier models and their metadata."""
    return ClassifierRegistry.list_models()
