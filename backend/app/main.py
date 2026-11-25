from pathlib import Path
import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.core.config import settings
from app.models import species, breed, prediction  # noqa: F401 - ensure models registered
from app.api.routes import breeds, predictions, models, statistics

app = FastAPI(title=settings.PROJECT_NAME)

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(breeds.router, prefix=settings.API_V1_STR, tags=["breeds"])
app.include_router(predictions.router, prefix=settings.API_V1_STR, tags=["predictions"])
app.include_router(models.router, prefix=settings.API_V1_STR, tags=["models"])
app.include_router(statistics.router, prefix=settings.API_V1_STR, tags=["statistics"])

@app.get("/")
def read_root():
    return {"message": "Welcome to Pet Breed Classifier API"}
