"""Module nhận diện các giống chưa tồn tại / ngoài tập dữ liệu huấn luyện (Out-of-Distribution / Open-Set Recognition).

Cung cấp khả năng đánh giá độ bất định (uncertainty), tính điểm năng lượng (Free Energy Score),
biên xác suất (Margin Score), và hợp nhất đa tiêu chí (Composite Uncertainty) để phát hiện
các giống chó/mèo chưa từng xuất hiện trong tập huấn luyện (Novel / Unseen / Unknown Breeds).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F


@dataclass
class UnknownBreedAssessment:
    """Kết quả phân tích độ bất định và nhận diện giống chưa tồn tại."""

    is_unknown: bool
    breed_status: str  # "known" | "unknown"
    status_label_vi: str  # Nhãn tiếng Việt hiển thị
    unknown_score: float  # Điểm bất định trong đoạn [0.0, 1.0] (càng cao càng có khả năng là giống lạ)
    known_confidence: float  # Độ tin cậy thuộc về giống đã biết trong đoạn [0.0, 1.0]
    decision_metric: str  # Tên độ đo quyết định ("composite", "energy", "msp", "margin", "entropy")
    metric_value: float  # Giá trị tính toán thực tế của độ đo
    threshold: float  # Ngưỡng phân định được áp dụng
    summary: str  # Giải thích trực quan
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_unknown": self.is_unknown,
            "breed_status": self.breed_status,
            "status_label_vi": self.status_label_vi,
            "unknown_score": round(self.unknown_score, 4),
            "known_confidence": round(self.known_confidence, 4),
            "decision_metric": self.decision_metric,
            "metric_value": round(self.metric_value, 4),
            "threshold": round(self.threshold, 4),
            "summary": self.summary,
            "details": self.details,
        }


class UnknownBreedDetector:
    """Bộ nhận diện và phát hiện các giống chưa có trong tập dữ liệu huấn luyện.

    Hỗ trợ 5 cơ chế đánh giá độ bất định khoa học:
    1. Composite (Khuyên dùng): Kết hợp có trọng số giữa MSP, Margin Top-1/Top-2 và Normalized Entropy.
    2. MSP (Maximum Softmax Probability - Hendrycks & Gimpel, 2017).
    3. Margin: Hiệu số xác suất giữa lớp cao nhất và lớp thứ nhì (Top-1 - Top-2).
    4. Energy: Free Energy Score (Liu et al., NeurIPS 2020) dựa trên LogSumExp của logits.
    5. Entropy: Normalized Shannon Entropy phản ánh mức độ phân tán xác suất.
    """

    # Ngưỡng mặc định đã được hiệu chuẩn (calibrated) cho từng mô hình ở chế độ composite
    # Ngưỡng composite phản ánh điểm bất định U(x): nếu U(x) > THRESHOLD -> Giống chưa xác định
    DEFAULT_COMPOSITE_THRESHOLDS: Dict[str, float] = {
        "resnet18": 0.38,
        "seresnet18": 0.38,
        "densenet121": 0.38,
        "efficientnet-b0": 0.35,
        "dual-stream-efficientnet": 0.35,
    }

    # Ngưỡng mặc định cho chế độ Maximum Softmax Probability (MSP): nếu MSP < THRESHOLD -> Giống chưa xác định
    DEFAULT_MSP_THRESHOLDS: Dict[str, float] = {
        "resnet18": 0.65,
        "seresnet18": 0.65,
        "densenet121": 0.65,
        "efficientnet-b0": 0.65,
        "dual-stream-efficientnet": 0.65,
    }

    # Ngưỡng mặc định cho chế độ Margin (Top-1 - Top-2): nếu Margin < THRESHOLD -> Giống chưa xác định
    DEFAULT_MARGIN_THRESHOLDS: Dict[str, float] = {
        "resnet18": 0.20,
        "seresnet18": 0.25,
        "densenet121": 0.30,
        "efficientnet-b0": 0.35,
        "dual-stream-efficientnet": 0.40,
    }

    def __init__(
        self,
        default_metric: str = "composite",
        custom_thresholds: Optional[Dict[str, float]] = None,
        temperature: float = 1.0,
    ):
        self.default_metric = default_metric
        self.temperature = max(1e-4, temperature)
        self.custom_thresholds = custom_thresholds or {}

    def get_threshold(self, model_key: str, metric: str) -> float:
        """Lấy ngưỡng quyết định phù hợp cho từng mô hình và từng độ đo."""
        norm_key = model_key.lower().replace("_", "-")
        # Tìm khóa phù hợp nhất
        matched_key = "dual-stream-efficientnet"
        for key in self.DEFAULT_COMPOSITE_THRESHOLDS:
            if key in norm_key:
                matched_key = key
                break

        if matched_key in self.custom_thresholds:
            return self.custom_thresholds[matched_key]

        if metric == "msp":
            return self.DEFAULT_MSP_THRESHOLDS.get(matched_key, 0.50)
        if metric == "margin":
            return self.DEFAULT_MARGIN_THRESHOLDS.get(matched_key, 0.30)
        # Mặc định composite
        return self.DEFAULT_COMPOSITE_THRESHOLDS.get(matched_key, 0.40)

    def evaluate(
        self,
        probs: torch.Tensor,
        logits: Optional[torch.Tensor] = None,
        model_name: str = "dual-stream-efficientnet",
        metric: Optional[str] = None,
        threshold: Optional[float] = None,
    ) -> UnknownBreedAssessment:
        """Đánh giá xác suất và logits để xác định xem ảnh có thuộc giống chưa từng thấy hay không.

        Args:
            probs: Tensor xác suất [num_classes] (ví dụ 37 lớp).
            logits: Tensor logits thô [num_classes] trước softmax (tùy chọn, dùng cho Energy).
            model_name: Tên mô hình đang thực hiện suy luận.
            metric: Độ đo quyết định ('composite' | 'msp' | 'margin' | 'energy' | 'entropy').
            threshold: Ngưỡng tùy chỉnh (nếu None sẽ lấy ngưỡng chuẩn của mô hình).
        """
        chosen_metric = metric or self.default_metric
        applied_threshold = threshold if threshold is not None else self.get_threshold(model_name, chosen_metric)

        p = probs.detach().cpu()
        num_classes = p.shape[-1]
        top_probs, _ = torch.topk(p, k=min(2, num_classes))
        p_top1 = float(top_probs[0].item())
        p_top2 = float(top_probs[1].item()) if num_classes > 1 else 0.0

        # 1. Maximum Softmax Probability (MSP)
        msp_val = p_top1

        # 2. Margin giữa Top-1 và Top-2
        margin_val = max(0.0, p_top1 - p_top2)

        # 3. Normalized Shannon Entropy [0.0, 1.0]
        # H(p) = - sum(p * log(p)) / log(num_classes)
        log_k = math.log(max(2, num_classes))
        entropy_val = float((-(p * torch.log(p + 1e-12)).sum()).item() / log_k)
        entropy_val = max(0.0, min(1.0, entropy_val))

        # 4. Free Energy Score (nếu có logits)
        energy_val = 0.0
        if logits is not None:
            l = logits.detach().cpu()
            # Free Energy = T * logsumexp(z / T)
            scaled_logits = l / self.temperature
            energy_val = float((self.temperature * torch.logsumexp(scaled_logits, dim=-1)).item())

        # 5. Composite Uncertainty Score U(x) in [0.0, 1.0]
        # Điểm tin cậy giống đã biết: C = 0.40 * MSP + 0.35 * Margin + 0.25 * (1 - NormalizedEntropy)
        confidence_known = 0.40 * msp_val + 0.35 * margin_val + 0.25 * (1.0 - entropy_val)
        confidence_known = max(0.0, min(1.0, confidence_known))
        composite_uncertainty = 1.0 - confidence_known

        # Đưa ra quyết định theo độ đo được lựa chọn
        if chosen_metric == "msp":
            metric_val = msp_val
            is_unknown = metric_val < applied_threshold
            unknown_score = 1.0 - msp_val
        elif chosen_metric == "margin":
            metric_val = margin_val
            is_unknown = metric_val < applied_threshold
            unknown_score = 1.0 - margin_val
        elif chosen_metric == "entropy":
            metric_val = entropy_val
            is_unknown = metric_val > applied_threshold
            unknown_score = entropy_val
        elif chosen_metric == "energy":
            metric_val = energy_val
            is_unknown = metric_val < applied_threshold
            unknown_score = 1.0 / (1.0 + math.exp(max(-10.0, min(10.0, energy_val))))
        else:  # "composite"
            metric_val = composite_uncertainty
            # Giống chưa xác định nếu điểm bất định cao HOẶC độ tin cậy Top-1 (MSP) < 65%
            is_unknown = (composite_uncertainty > applied_threshold) or (msp_val < 0.65)
            unknown_score = max(composite_uncertainty, 1.0 - msp_val)

        breed_status = "unknown" if is_unknown else "known"
        if is_unknown:
            status_label_vi = "Giống chưa xác định (Ngoài tập huấn luyện)"
            summary = (
                f"Phát hiện độ bất định cao ({unknown_score:.1%} > ngưỡng {applied_threshold:.1%}). "
                f"Thú cưng có nhiều nét khác biệt với 37 giống đã học, có khả năng cao là giống mới chưa có trong hệ thống."
            )
        else:
            status_label_vi = "Giống đã xác định"
            summary = (
                f"Độ tin cậy nhận diện cao ({confidence_known:.1%}). "
                f"Các đặc trưng giải phẫu khớp rõ ràng với giống loài trong danh mục."
            )

        details = {
            "msp": round(msp_val, 4),
            "margin": round(margin_val, 4),
            "normalized_entropy": round(entropy_val, 4),
            "energy_score": round(energy_val, 4) if logits is not None else None,
            "composite_uncertainty": round(composite_uncertainty, 4),
            "confidence_known": round(confidence_known, 4),
        }

        return UnknownBreedAssessment(
            is_unknown=is_unknown,
            breed_status=breed_status,
            status_label_vi=status_label_vi,
            unknown_score=unknown_score,
            known_confidence=confidence_known,
            decision_metric=chosen_metric,
            metric_value=metric_val,
            threshold=applied_threshold,
            summary=summary,
            details=details,
        )


# Singleton detector toàn cục dùng chung cho các service và API
unknown_breed_detector = UnknownBreedDetector()
