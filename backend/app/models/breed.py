from sqlalchemy import Column, Integer, String, ForeignKey
from sqlalchemy.orm import relationship
from app.core.database import Base

class Breed(Base):
    __tablename__ = "breeds"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    species_id = Column(Integer, ForeignKey("species.id"))

    species = relationship("Species", back_populates="breeds")
    predictions = relationship("Prediction", back_populates="predicted_breed")
