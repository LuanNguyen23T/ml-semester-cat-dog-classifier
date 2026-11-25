from typing import List, Optional
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.schemas.breed import BreedResponse, SpeciesBase
from app.repositories import breed as breed_repo

router = APIRouter()

@router.get("/species", response_model=List[SpeciesBase])
def read_species(db: Session = Depends(get_db)):
    """Get all species (e.g. Cat, Dog)"""
    return breed_repo.get_species_list(db)

@router.get("/breeds", response_model=List[BreedResponse])
def read_breeds(species_id: Optional[int] = None, skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    """Get all breeds, optionally filtered by species_id"""
    return breed_repo.get_breeds(db, species_id=species_id, skip=skip, limit=limit)
