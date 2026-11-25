"""Modular inference preprocessing, lazy pet segmentation, Dual-Stream head cropping, and 4-View TTA."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import threading
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.models.segmentation as seg_models
import torchvision.transforms as T
from PIL import Image

from app.ai.common import get_device

logger = logging.getLogger(__name__)

IMG_SIZE = 288
RESIZE_SIZE = int(IMG_SIZE * 1.14)  # 328
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
TTA_WEIGHTS = (0.40, 0.30, 0.15, 0.15)

COCO_CAT_CLASS_ID = 8
COCO_DOG_CLASS_ID = 12
BG_COLOR = (128, 128, 128)

_INFERENCE_TRANSFORM = T.Compose([
    T.Resize((RESIZE_SIZE, RESIZE_SIZE)),
    T.CenterCrop(IMG_SIZE),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


@dataclass(frozen=True)
class PreprocessedInputs:
    """Segmented & cropped PIL views shared between single and batch inference."""

    body_img: Image.Image
    head_img: Image.Image


# ─── Lazy-loaded singleton segmentation model ─────────────────────────────────
_seg_lock = threading.Lock()
_seg_model: Optional[torch.nn.Module] = None
_seg_transform = None
_seg_device: Optional[torch.device] = None


def _get_segmentation_model() -> Tuple[torch.nn.Module, object, torch.device]:
    """Load DeepLabV3-MobileNetV3-Large once per process on first use."""
    global _seg_model, _seg_transform, _seg_device

    if _seg_model is None:
        with _seg_lock:
            if _seg_model is None:
                device = get_device()
                weights = seg_models.DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT
                model = seg_models.deeplabv3_mobilenet_v3_large(weights=weights).to(device)
                model.eval()
                _seg_transform = weights.transforms()
                _seg_device = device
                _seg_model = model
                logger.info("[Preprocessing] DeepLabV3-MobileNetV3-Large ready on %s.", device)

    return _seg_model, _seg_transform, _seg_device


def _predict_pet_mask(pil_img: Image.Image) -> Optional[np.ndarray]:
    """Return boolean [H, W] mask for cat/dog pixels, or None on failure/empty mask."""
    try:
        seg_model, seg_transform, device = _get_segmentation_model()
        orig_w, orig_h = pil_img.size
        inp = seg_transform(pil_img).unsqueeze(0).to(device)

        seg_model.eval()
        with torch.no_grad():
            logits = seg_model(inp)["out"]
            logits = F.interpolate(
                logits, size=(orig_h, orig_w), mode="bilinear", align_corners=False
            )
            pred = logits.argmax(dim=1).squeeze(0)
            mask = ((pred == COCO_CAT_CLASS_ID) | (pred == COCO_DOG_CLASS_ID)).cpu().numpy()

        return mask if mask.any() else None
    except Exception as exc:
        logger.warning("[Preprocessing] Segmentation failed; falling back to raw image: %s", exc)
        return None


def _extract_foreground_views(pil_img: Image.Image) -> Tuple[Image.Image, Image.Image]:
    """Build (body_img_gray_bg, raw_pet_crop) with automatic raw-image fallback."""
    mask = _predict_pet_mask(pil_img)
    if mask is None:
        raw_copy = pil_img.copy()
        return raw_copy, raw_copy

    img_arr = np.array(pil_img)
    h, w = mask.shape
    ys, xs = np.where(mask)
    pad_y = int(0.04 * h)
    pad_x = int(0.04 * w)
    ymin = max(0, int(ys.min()) - pad_y)
    ymax = min(h, int(ys.max()) + pad_y + 1)
    xmin = max(0, int(xs.min()) - pad_x)
    xmax = min(w, int(xs.max()) + pad_x + 1)

    crop_raw = img_arr[ymin:ymax, xmin:xmax]
    crop_mask = mask[ymin:ymax, xmin:xmax]
    crop_gray = crop_raw.copy()
    crop_gray[~crop_mask] = BG_COLOR

    return Image.fromarray(crop_gray), Image.fromarray(crop_raw)


def crop_head_heuristic(pil_img: Image.Image) -> Image.Image:
    """Crop upper-center region (xmin 10%, xmax 90%, ymin 0, ymax 65%)."""
    w, h = pil_img.size
    xmin = int(w * 0.10)
    xmax = int(w * 0.90)
    ymin = 0
    ymax = int(h * 0.65)
    if xmax <= xmin or ymax <= ymin:
        return pil_img.copy()
    return pil_img.crop((xmin, ymin, xmax, ymax))


def prepare_batch_inputs(pil_img: Image.Image) -> PreprocessedInputs:
    """Run segmentation once and return PIL body/head views for reuse across models."""
    body_img, raw_pet_crop = _extract_foreground_views(pil_img)
    head_img = crop_head_heuristic(raw_pet_crop)
    return PreprocessedInputs(body_img=body_img, head_img=head_img)


def image_to_tensor(pil_img: Image.Image) -> torch.Tensor:
    """Apply canonical 288x288 inference transform to a prepared PIL image -> [1, 3, 288, 288]."""
    return _INFERENCE_TRANSFORM(pil_img).unsqueeze(0)


def preprocess_single_image(pil_img: Image.Image) -> torch.Tensor:
    """Segment pet foreground (fallback raw) and apply 288x288 inference transform."""
    prepared = prepare_batch_inputs(pil_img)
    return image_to_tensor(prepared.body_img)


def preprocess_dual_stream(pil_img: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build (body_tensor, head_tensor): gray-bg body crop + natural-bg 65% head crop."""
    prepared = prepare_batch_inputs(pil_img)
    return image_to_tensor(prepared.body_img), image_to_tensor(prepared.head_img)


def _zoom_tensor(tensor: torch.Tensor, zoom_ratio: float = 0.9) -> torch.Tensor:
    """Center-crop tensor by zoom_ratio and bilinear-interpolate back to original HxW."""
    _, _, h, w = tensor.shape
    h_c, w_c = int(h * zoom_ratio), int(w * zoom_ratio)
    top = (h - h_c) // 2
    left = (w - w_c) // 2
    crop = tensor[:, :, top : top + h_c, left : left + w_c]
    return F.interpolate(crop, size=(h, w), mode="bilinear", align_corners=False)


def make_4view_tta_batch(tensor: torch.Tensor) -> torch.Tensor:
    """Expand [1, C, H, W] into 4-view batch [4, C, H, W]: raw, flip, zoom, flip+zoom."""
    v1 = tensor
    v2 = torch.flip(tensor, dims=[-1])
    v3 = _zoom_tensor(tensor, 0.9)
    v4 = torch.flip(v3, dims=[-1])
    return torch.cat([v1, v2, v3, v4], dim=0)


def aggregate_tta_probs(logits: torch.Tensor) -> torch.Tensor:
    """Compute weighted softmax probability vector [C] from 4-view logits [4, C]."""
    probs = F.softmax(logits, dim=-1)
    weights = probs.new_tensor(TTA_WEIGHTS).unsqueeze(1)  # [4, 1]
    return (probs * weights).sum(dim=0)
