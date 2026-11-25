"""Seed the species and 37 Oxford-IIIT Pet breeds idempotently from BREED_CATALOGUE."""

from __future__ import annotations

import argparse
import os
import sys

# Add backend directory to sys.path so app imports resolve from repo root.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../backend"))

from sqlalchemy import create_engine, text  # noqa: E402
from sqlalchemy.exc import OperationalError  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402

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


def seed_data(custom_engine=None) -> None:
    target_engine = custom_engine or engine
    target_session_factory = sessionmaker(autocommit=False, autoflush=False, bind=target_engine) if custom_engine else SessionLocal

    try:
        Base.metadata.create_all(bind=target_engine)
    except OperationalError as e:
        print("\n[LỖI KẾT NỐI DATABASE]")
        print(f"Chi tiết: {e}")
        print("\n-> Gợi ý: Hãy đảm bảo Docker Desktop đang chạy và đã khởi động container PostgreSQL:")
        print("   docker compose up -d")
        print("-> Hoặc nếu muốn dùng SQLite để kiểm tra nhanh:")
        print("   python3 scripts/seed_breeds.py --db-url sqlite:///./pet_classifier.db\n")
        sys.exit(1)

    db = target_session_factory()
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
        if target_engine.dialect.name == "postgresql":
            try:
                db.execute(text("SELECT setval(pg_get_serial_sequence('species', 'id'), (SELECT COALESCE(MAX(id), 1) FROM species))"))
                db.execute(text("SELECT setval(pg_get_serial_sequence('breeds', 'id'), (SELECT COALESCE(MAX(id), 1) FROM breeds))"))
                db.commit()
            except Exception as seq_err:
                print(f"[CẢNH BÁO] Không thể reset sequence (không ảnh hưởng dữ liệu): {seq_err}")

        print(f"Database seeding completed: inserted={inserted}, updated={updated}, total={len(BREED_CATALOGUE)}.")
    except OperationalError as e:
        print("\n[LỖI KẾT NỐI DATABASE TRONG KHI GHI DỮ LIỆU]")
        print(f"Chi tiết: {e}")
        sys.exit(1)
    finally:
        db.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Seed species and 37 breeds into database")
    parser.add_argument("--db-url", type=str, default="", help="Custom Database URL (e.g. sqlite:///./pet_classifier.db)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.db_url:
        connect_args = {"check_same_thread": False} if args.db_url.startswith("sqlite") else {}
        custom_eng = create_engine(args.db_url, connect_args=connect_args)
        seed_data(custom_engine=custom_eng)
    else:
        seed_data()

