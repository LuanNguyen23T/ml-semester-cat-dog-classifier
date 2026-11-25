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

@app.on_event("startup")
def startup_db_init():
    """Auto-create tables and seed 37 breeds on startup if empty."""
    try:
        from app.core.database import engine, Base, SessionLocal
        Base.metadata.create_all(bind=engine)
        from app.models.breed import Breed
        db = SessionLocal()
        if db.query(Breed).count() == 0:
            from app.models.species import Species
            from app.ai.common import BREED_CATALOGUE
            db.add(Species(id=1, name="Cat"))
            db.add(Species(id=2, name="Dog"))
            db.flush()
            for b_id, b_name, s_name, _ in BREED_CATALOGUE:
                db.add(Breed(id=b_id, name=b_name, species_id=1 if s_name == "Cat" else 2))
            db.commit()
            print("[Startup] Auto-seeded 37 breeds successfully.")
        db.close()
    except Exception as e:
        print("[Startup DB Notice]", e)

@app.get("/")
def read_root():
    return {"message": "Welcome to Pet Breed Classifier API"}
