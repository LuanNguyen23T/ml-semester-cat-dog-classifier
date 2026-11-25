"""Thread-safe lazy-loading registry for the 5 custom CNN pet breed classifiers."""

from __future__ import annotations

import logging
import threading
from enum import Enum
from typing import Dict

from app.ai.base import Classifier
from app.ai.common import WEIGHTS_DIR

logger = logging.getLogger(__name__)


class ModelMethod(str, Enum):
    RESNET18 = "resnet18"
    SERESNET18 = "seresnet18"
    DENSENET121 = "densenet121"
    EFFICIENTNET_B0 = "efficientnet-b0"
    DUAL_STREAM_EFFICIENTNET = "dual-stream-efficientnet"


MODEL_DISPLAY_INFO: Dict[str, dict] = {
    ModelMethod.DUAL_STREAM_EFFICIENTNET: {
        "name": "Dual-Stream EfficientNet-B0",
        "description": "Mô hình 2 luồng (Thân + Mặt) với cơ chế cổng Gated Fusion, đạt độ chính xác cao nhất.",
        "accuracy": 85.72,
        "accuracy_single": 85.72,
        "accuracy_tta": 86.54,
        "params": 5042081,
        "recommended": True,
        "is_default": True,
        "tag": "Recommended",
        "weights_file": "dual_stream_efficientnet_state_dict.pth",
    },
    ModelMethod.EFFICIENTNET_B0: {
        "name": "EfficientNet-B0",
        "description": "Mạng MBConv + Squeeze-and-Excitation gọn nhẹ, cân bằng tốt giữa tốc độ và độ chính xác.",
        "accuracy": 83.32,
        "accuracy_single": 83.32,
        "accuracy_tta": 83.78,
        "params": 4045605,
        "recommended": False,
        "is_default": False,
        "tag": "Fast & Accurate",
        "weights_file": "efficientnet_b0_state_dict.pth",
    },
    ModelMethod.DENSENET121: {
        "name": "DenseNet-121",
        "description": "Mạng kết nối dày đặc tái sử dụng đặc trưng đa tầng hiệu quả.",
        "accuracy": 74.54,
        "accuracy_single": 74.54,
        "accuracy_tta": 75.33,
        "params": 6991781,
        "recommended": False,
        "is_default": False,
        "tag": "Dense Features",
        "weights_file": "densenet121_state_dict.pth",
    },
    ModelMethod.SERESNET18: {
        "name": "SEResNet-18",
        "description": "ResNet-18 tích hợp khối chú ý kênh Squeeze-and-Excitation.",
        "accuracy": 70.62,
        "accuracy_single": 70.62,
        "accuracy_tta": 71.55,
        "params": 11282533,
        "recommended": False,
        "is_default": False,
        "tag": "Channel Attention",
        "weights_file": "seresnet18_state_dict.pth",
    },
    ModelMethod.RESNET18: {
        "name": "ResNet-18",
        "description": "Mạng tích chập phần dư cổ điển làm mốc đối chiếu cơ sở.",
        "accuracy": 67.84,
        "accuracy_single": 67.84,
        "accuracy_tta": 68.47,
        "params": 11195493,
        "recommended": False,
        "is_default": False,
        "tag": "Baseline",
        "weights_file": "resnet18_state_dict.pth",
    },
}


class ClassifierRegistry:
    """Thread-safe lazy-loading singleton registry for all 5 classifiers."""

    _cache: Dict[ModelMethod, Classifier] = {}
    _lock = threading.Lock()

    @classmethod
    def is_available(cls, method: ModelMethod) -> bool:
        info = MODEL_DISPLAY_INFO.get(method)
        if not info:
            return False
        return (WEIGHTS_DIR / info["weights_file"]).is_file()

    @classmethod
    def get(cls, method: ModelMethod) -> Classifier:
        """Return the cached classifier for method, loading once under a thread lock."""
        if method not in cls._cache:
            with cls._lock:
                if method not in cls._cache:
                    logger.info("[Registry] Loading classifier: %s", method.value)
                    cls._cache[method] = cls._load(method)
                    logger.info("[Registry] Classifier ready: %s", method.value)
        return cls._cache[method]

    @classmethod
    def _load(cls, method: ModelMethod) -> Classifier:
        if method == ModelMethod.RESNET18:
            from app.ai.cnn_classifiers import ResNet18Classifier
            return ResNet18Classifier()
        if method == ModelMethod.SERESNET18:
            from app.ai.cnn_classifiers import SEResNet18Classifier
            return SEResNet18Classifier()
        if method == ModelMethod.DENSENET121:
            from app.ai.cnn_classifiers import DenseNet121Classifier
            return DenseNet121Classifier()
        if method == ModelMethod.EFFICIENTNET_B0:
            from app.ai.cnn_classifiers import EfficientNetB0Classifier
            return EfficientNetB0Classifier()
        if method == ModelMethod.DUAL_STREAM_EFFICIENTNET:
            from app.ai.cnn_classifiers import DualStreamEfficientNetClassifier
            return DualStreamEfficientNetClassifier()
        raise ValueError(f"Unknown model method: {method}")

    @classmethod
    def list_models(cls) -> list:
        """Return metadata list for all 5 models (never crashes if a weights file is missing)."""
        items = []
        for method, info in MODEL_DISPLAY_INFO.items():
            available = cls.is_available(method)
            items.append({
                "id": method.value,
                **info,
                "available": available,
            })
        return items
