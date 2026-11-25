from typing import List, Optional
from sqlalchemy.orm import Session
from app.models.breed import Breed
from app.models.species import Species

def get_species_list(db: Session) -> List[Species]:
    return db.query(Species).all()

def get_breed(db: Session, breed_id: int) -> Optional[Breed]:
    return db.query(Breed).filter(Breed.id == breed_id).first()

def get_breeds(db: Session, species_id: Optional[int] = None, skip: int = 0, limit: int = 100) -> List[Breed]:
    query = db.query(Breed)
    if species_id is not None:
        query = query.filter(Breed.species_id == species_id)
    return query.offset(skip).limit(limit).all()
