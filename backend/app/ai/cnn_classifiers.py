"""Classifiers for the 5 custom-trained CNN architectures."""

from __future__ import annotations

import logging
from typing import Callable, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL.Image import Image as PILImage

from app.ai.architectures import (
    CustomSEResNet,
    DenseNet121,
    DualStreamEfficientNet,
    EfficientNetB0,
    ResNet18,
)
from app.ai.base import Classifier, ClassifierResult
from app.ai.common import (
    WEIGHTS_DIR,
    build_classifier_result,
    get_device,
    load_model_state_dict,
    load_rgb_image,
)
from app.ai.preprocessing import (
    PreprocessedInputs,
    aggregate_tta_probs,
    image_to_tensor,
    make_4view_tta_batch,
    prepare_batch_inputs,
)

logger = logging.getLogger(__name__)


def _format_model_version(base_version: str, use_tta: bool) -> str:
    return f"{base_version}+tta" if use_tta else base_version


class SingleStreamCNNClassifier(Classifier):
    """Shared classifier wrapper for single-stream 288x288 CNN models."""

    def __init__(
        self,
        model_builder: Callable[[], nn.Module],
        weights_filename: str,
        model_version: str,
        top_k: int = 5,
    ):
        self.top_k = top_k
        self.model_version = model_version
        self._device = get_device()

        self._model = model_builder().to(self._device)
        state_dict = load_model_state_dict(WEIGHTS_DIR / weights_filename, self._device)
        self._model.load_state_dict(state_dict, strict=True)
        self._model.eval()
        logger.info("[%s] Ready on %s (%s).", model_version, self._device, weights_filename)

    def predict_from_preprocessed(
        self,
        prepared: PreprocessedInputs,
        use_tta: bool = False,
    ) -> ClassifierResult:
        """Classify using pre-segmented PIL views (shared in batch requests)."""
        tensor = image_to_tensor(prepared.body_img).to(self._device)

        self._model.eval()
        with torch.no_grad():
            if use_tta:
                logits = self._model(make_4view_tta_batch(tensor))
                probs = aggregate_tta_probs(logits)
            else:
                logits = self._model(tensor).squeeze(0)
                probs = F.softmax(logits, dim=-1)

        return build_classifier_result(
            probs,
            model_version=_format_model_version(self.model_version, use_tta),
            top_k=self.top_k,
        )

    def predict(
        self,
        image: Union[str, bytes, PILImage],
        use_tta: bool = False,
    ) -> ClassifierResult:
        pil_img = load_rgb_image(image)
        prepared = prepare_batch_inputs(pil_img)
        return self.predict_from_preprocessed(prepared, use_tta=use_tta)


class ResNet18Classifier(SingleStreamCNNClassifier):
    def __init__(self, top_k: int = 5):
        super().__init__(
            model_builder=lambda: ResNet18(num_classes=37),
            weights_filename="resnet18_state_dict.pth",
            model_version="resnet18-288-v1",
            top_k=top_k,
        )


class SEResNet18Classifier(SingleStreamCNNClassifier):
    def __init__(self, top_k: int = 5):
        super().__init__(
            model_builder=lambda: CustomSEResNet(num_classes=37),
            weights_filename="seresnet18_state_dict.pth",
            model_version="seresnet18-288-v1",
            top_k=top_k,
        )


class DenseNet121Classifier(SingleStreamCNNClassifier):
    def __init__(self, top_k: int = 5):
        super().__init__(
            model_builder=lambda: DenseNet121(num_classes=37),
            weights_filename="densenet121_state_dict.pth",
            model_version="densenet121-288-v1",
            top_k=top_k,
        )


class EfficientNetB0Classifier(SingleStreamCNNClassifier):
    def __init__(self, top_k: int = 5):
        super().__init__(
            model_builder=lambda: EfficientNetB0(num_classes=37),
            weights_filename="efficientnet_b0_state_dict.pth",
            model_version="efficientnet-b0-288-v1",
            top_k=top_k,
        )


class DualStreamEfficientNetClassifier(Classifier):
    """Classifier for Dual-Stream EfficientNet-B0 (Body + Head Zoom with Gated Fusion)."""

    WEIGHTS_FILENAME = "dual_stream_efficientnet_state_dict.pth"
    MODEL_VERSION = "dual-stream-efficientnet-b0-288-v1"

    def __init__(self, top_k: int = 5):
        self.top_k = top_k
        self._device = get_device()

        self._model = DualStreamEfficientNet(num_classes=37).to(self._device)
        state_dict = load_model_state_dict(WEIGHTS_DIR / self.WEIGHTS_FILENAME, self._device)
        self._model.load_state_dict(state_dict, strict=True)
        self._model.eval()
        logger.info("[%s] Ready on %s (%s).", self.MODEL_VERSION, self._device, self.WEIGHTS_FILENAME)

    def predict_from_preprocessed(
        self,
        prepared: PreprocessedInputs,
        use_tta: bool = False,
    ) -> ClassifierResult:
        """Classify using pre-segmented PIL body & head views (shared in batch requests)."""
        body_tensor = image_to_tensor(prepared.body_img).to(self._device)
        head_tensor = image_to_tensor(prepared.head_img).to(self._device)

        self._model.eval()
        with torch.no_grad():
            if use_tta:
                body_batch = make_4view_tta_batch(body_tensor)
                head_batch = make_4view_tta_batch(head_tensor)
                logits = self._model(body_batch, head_batch)
                probs = aggregate_tta_probs(logits)
            else:
                logits = self._model(body_tensor, head_tensor).squeeze(0)
                probs = F.softmax(logits, dim=-1)

        return build_classifier_result(
            probs,
            model_version=_format_model_version(self.MODEL_VERSION, use_tta),
            top_k=self.top_k,
        )

    def predict(
        self,
        image: Union[str, bytes, PILImage],
        use_tta: bool = False,
    ) -> ClassifierResult:
        pil_img = load_rgb_image(image)
        prepared = prepare_batch_inputs(pil_img)
        return self.predict_from_preprocessed(prepared, use_tta=use_tta)
