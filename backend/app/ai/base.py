from abc import ABC, abstractmethod
from typing import Union, List, Any
from pydantic import BaseModel
from PIL.Image import Image as PILImage

class PredictionResultItem(BaseModel):
    breed_id: int
    breed_name: str
    species: str
    score: float

class ClassifierResult(BaseModel):
    model_config = {"protected_namespaces": ()}

    predicted_breed_id: int
    score: float
    top_k_results: List[PredictionResultItem]
    model_version: str

class Classifier(ABC):
    @abstractmethod
    def predict(self, image: Union[str, bytes, PILImage]) -> ClassifierResult:
        """
        Run classification on the provided image.
        Image can be a file path (str), raw bytes (bytes), or a PIL Image.
        """
        pass
