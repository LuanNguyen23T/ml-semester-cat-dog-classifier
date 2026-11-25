"""Seed the species and 37 Oxford-IIIT Pet breeds idempotently from BREED_CATALOGUE."""

from __future__ import annotations

import os
import sys

# Add backend directory to sys.path so app imports resolve from repo root.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../backend"))

from sqlalchemy import text  # noqa: E402

from app.ai.common import BREED_CATALOGUE  # noqa: E402
from app.core.database import Base, SessionLocal, engine  # noqa: E402
from app.models.breed import Breed  # noqa: E402
from app.models.prediction import Prediction  # noqa: E402, F401
from app.models.species import Species  # noqa: E402


def _upsert_species(db, species_id: int, name: str) -> None:
    existing = db.query(Species).filter(Species.id == species_id).first()
    if existing:
        if existing.name != name:
            existing.name = name
        return
    db.add(Species(id=species_id, name=name))


def seed_data() -> None:
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        _upsert_species(db, 1, "Cat")
        _upsert_species(db, 2, "Dog")
        db.flush()

        species_id_by_name = {"Cat": 1, "Dog": 2}
        inserted = 0
        updated = 0

        for breed_id, breed_name, species_name, _ in BREED_CATALOGUE:
            species_id = species_id_by_name[species_name]
            existing = db.query(Breed).filter(Breed.id == breed_id).first()
            if existing:
                if existing.name != breed_name or existing.species_id != species_id:
                    existing.name = breed_name
                    existing.species_id = species_id
                    updated += 1
            else:
                db.add(Breed(id=breed_id, name=breed_name, species_id=species_id))
                inserted += 1

        db.commit()

        # Sync PostgreSQL sequences when running against Postgres.
        if db.bind and db.bind.dialect.name == "postgresql":
            db.execute(text("SELECT setval(pg_get_serial_sequence('species', 'id'), COALESCE(MAX(id), 1)) FROM species"))
            db.execute(text("SELECT setval(pg_get_serial_sequence('breeds', 'id'), COALESCE(MAX(id), 1)) FROM breeds"))
            db.commit()

        print(f"Database seeding completed: inserted={inserted}, updated={updated}, total={len(BREED_CATALOGUE)}.")
    finally:
        db.close()


if __name__ == "__main__":
    seed_data()
