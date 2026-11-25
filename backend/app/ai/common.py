"""Shared constants, device selection, and PIL/result helpers without CLIP dependencies."""

from __future__ import annotations

import io
from pathlib import Path
from typing import List, Union

import torch
import torch.nn.functional as F
from PIL import Image as PILImage
from PIL import ImageOps

from app.ai.base import ClassifierResult, PredictionResultItem
from app.core.config import settings

WEIGHTS_DIR = Path(__file__).parent.parent.parent / "weights"

# 37 breeds — order matches Oxford-IIIT Pet class_id 1..37 and DB breeds.id 1..37.
BREED_CATALOGUE = [
    (1,  "Abyssinian",                 "Cat", "Abyssinian cat"),
    (2,  "american_bulldog",           "Dog", "American Bulldog"),
    (3,  "american_pit_bull_terrier",  "Dog", "American Pit Bull Terrier"),
    (4,  "basset_hound",               "Dog", "Basset Hound"),
    (5,  "beagle",                     "Dog", "Beagle"),
    (6,  "Bengal",                     "Cat", "Bengal cat"),
    (7,  "Birman",                     "Cat", "Birman cat"),
    (8,  "Bombay",                     "Cat", "Bombay cat"),
    (9,  "boxer",                      "Dog", "Boxer dog"),
    (10, "British_Shorthair",          "Cat", "British Shorthair cat"),
    (11, "chihuahua",                  "Dog", "Chihuahua"),
    (12, "Egyptian_Mau",               "Cat", "Egyptian Mau cat"),
    (13, "english_cocker_spaniel",     "Dog", "English Cocker Spaniel"),
    (14, "english_setter",             "Dog", "English Setter"),
    (15, "german_shorthaired",         "Dog", "German Shorthaired Pointer"),
    (16, "great_pyrenees",             "Dog", "Great Pyrenees"),
    (17, "havanese",                   "Dog", "Havanese"),
    (18, "japanese_chin",              "Dog", "Japanese Chin"),
    (19, "keeshond",                   "Dog", "Keeshond"),
    (20, "leonberger",                 "Dog", "Leonberger"),
    (21, "Maine_Coon",                 "Cat", "Maine Coon cat"),
    (22, "miniature_pinscher",         "Dog", "Miniature Pinscher"),
    (23, "newfoundland",               "Dog", "Newfoundland dog"),
    (24, "Persian",                    "Cat", "Persian cat"),
    (25, "pomeranian",                 "Dog", "Pomeranian"),
    (26, "pug",                        "Dog", "Pug"),
    (27, "Ragdoll",                    "Cat", "Ragdoll cat"),
    (28, "Russian_Blue",               "Cat", "Russian Blue cat"),
    (29, "saint_bernard",              "Dog", "Saint Bernard"),
    (30, "samoyed",                    "Dog", "Samoyed"),
    (31, "scottish_terrier",           "Dog", "Scottish Terrier"),
    (32, "shiba_inu",                  "Dog", "Shiba Inu"),
    (33, "Siamese",                    "Cat", "Siamese cat"),
    (34, "Sphynx",                     "Cat", "Sphynx cat"),
    (35, "staffordshire_bull_terrier", "Dog", "Staffordshire Bull Terrier"),
    (36, "wheaten_terrier",            "Dog", "Wheaten Terrier"),
    (37, "yorkshire_terrier",          "Dog", "Yorkshire Terrier"),
]


def get_device() -> torch.device:
    """Resolve the configured inference device, preferring MPS on macOS."""
    requested = settings.AI_DEVICE.strip().lower()
    if requested not in {"auto", "cpu", "mps", "cuda"}:
        raise ValueError("AI_DEVICE must be one of: auto, cpu, mps, cuda")

    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("AI_DEVICE=mps was requested, but MPS is not available")
        return torch.device("mps")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("AI_DEVICE=cuda was requested, but CUDA is not available")
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")

    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_rgb_image(image: Union[str, bytes, PILImage.Image]) -> PILImage.Image:
    """Open an image from path, bytes, or PIL, apply EXIF orientation, and convert to RGB."""
    if isinstance(image, str):
        with PILImage.open(image) as opened:
            pil = ImageOps.exif_transpose(opened).convert("RGB")
    elif isinstance(image, bytes):
        with PILImage.open(io.BytesIO(image)) as opened:
            pil = ImageOps.exif_transpose(opened).convert("RGB")
    elif isinstance(image, PILImage.Image):
        pil = ImageOps.exif_transpose(image).convert("RGB")
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")
    return pil


def build_classifier_result(
    probs: torch.Tensor,
    model_version: str,
    top_k: int = 5,
) -> ClassifierResult:
    """Convert a 1D probability tensor [37] into a Top-K ClassifierResult."""
    k = min(top_k, len(BREED_CATALOGUE))
    top_probs, top_idx = torch.topk(probs, k=k)

    top_k_items: List[PredictionResultItem] = []
    for prob, idx in zip(top_probs.detach().cpu().tolist(), top_idx.detach().cpu().tolist()):
        breed_id, breed_name, species, _ = BREED_CATALOGUE[idx]
        top_k_items.append(
            PredictionResultItem(
                breed_id=breed_id,
                breed_name=breed_name,
                species=species,
                score=round(prob, 6),
            )
        )

    best = top_k_items[0]
    return ClassifierResult(
        predicted_breed_id=best.breed_id,
        score=round(best.score, 6),
        top_k_results=top_k_items,
        model_version=model_version,
    )


def load_model_state_dict(weights_path: Path, device: torch.device) -> dict:
    """Load a weight file supporting either raw state_dict or full training checkpoint."""
    if not weights_path.exists():
        raise FileNotFoundError(
            f"Missing model weights: {weights_path}. "
            "Run scripts/extract_state_dicts.py first."
        )
    payload = torch.load(weights_path, map_location=device, weights_only=True)
    if isinstance(payload, dict) and "model_state_dict" in payload:
        return payload["model_state_dict"]
    return payload
