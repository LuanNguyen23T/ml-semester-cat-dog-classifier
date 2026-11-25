from pydantic import BaseModel

class SpeciesBase(BaseModel):
    id: int
    name: str

    class Config:
        from_attributes = True

class BreedBase(BaseModel):
    id: int
    name: str
    species_id: int

class BreedResponse(BreedBase):
    species: SpeciesBase

    class Config:
        from_attributes = True
