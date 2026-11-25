"""Thực nghiệm kiểm thử Module Nhận Diện Giống Chưa Tồn Tại (Novel / Unknown Breed Recognition).

Đánh giá khả năng phát hiện giống ngoài tập huấn luyện (Out-of-Distribution / Open-Set Recognition)
trên cả 5 mô hình CNN sử dụng tập dữ liệu Stanford Dogs:
- Tập In-Distribution (Giống đã biết): 21 giống chó giao nhau với Oxford-IIIT Pet.
- Tập Out-of-Distribution (Giống chưa từng thấy / lạ): Các giống chó không tồn tại trong Oxford-IIIT Pet
  (ví dụ: Golden Retriever, German Shepherd, Rottweiler, Siberian Husky, Doberman, French Bulldog,...).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "backend"))

from app.ai.common import BREED_CATALOGUE, get_device  # type: ignore # noqa: E402
from app.ai.preprocessing import prepare_batch_inputs  # type: ignore # noqa: E402
from app.ai.registry import ClassifierRegistry, ModelMethod  # type: ignore # noqa: E402
from app.ai.unknown_detector import UnknownBreedDetector, unknown_breed_detector  # type: ignore # noqa: E402

OVERLAPPING_ST_IDS = {
    10, 11, 91, 0, 68, 61, 59, 105, 1, 109, 103, 86, 104, 107, 102, 96, 106, 48, 28, 51, 36
}

MODEL_NAMES = {
    ModelMethod.RESNET18: "ResNet-18",
    ModelMethod.SERESNET18: "SEResNet-18",
    ModelMethod.DENSENET121: "DenseNet-121",
    ModelMethod.EFFICIENTNET_B0: "EfficientNet-B0",
    ModelMethod.DUAL_STREAM_EFFICIENTNET: "Dual-Stream EfficientNet-B0",
}


def calculate_auroc(pos_scores: List[float], neg_scores: List[float]) -> float:
    """Tính Area Under ROC Curve (AUROC) theo định lý Mann-Whitney U.

    pos_scores: Điểm bất định (unknown_score) của mẫu OOD (Unknown) - nhãn dương (1).
    neg_scores: Điểm bất định (unknown_score) của mẫu ID (Known) - nhãn âm (0).
    """
    if not pos_scores or not neg_scores:
        return 0.5
    pos = np.array(pos_scores)
    neg = np.array(neg_scores)
    all_scores = np.concatenate([pos, neg])
    all_labels = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    order = np.argsort(all_scores)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(len(all_scores))
    pos_ranks = ranks[all_labels == 1]
    n_pos = len(pos)
    n_neg = len(neg)
    u_stat = np.sum(pos_ranks) - (n_pos * (n_pos - 1)) / 2.0
    auroc = u_stat / (n_pos * n_neg)
    return float(np.clip(auroc, 0.0, 1.0))


def calculate_fpr95(pos_scores: List[float], neg_scores: List[float]) -> float:
    """Tính False Positive Rate khi True Positive Rate đạt 95% (FPR@95TPR)."""
    if not pos_scores or not neg_scores:
        return 1.0
    # Ngưỡng tại 95% TPR của mẫu OOD (pos_scores)
    threshold = float(np.percentile(pos_scores, 5.0))
    # Tỉ lệ mẫu Known (neg_scores) vượt ngưỡng này (bị nhầm là Unknown)
    fpr = float(np.mean(np.array(neg_scores) >= threshold))
    return fpr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test Unknown Breed Detector")
    parser.add_argument("--num-known", type=int, default=50,
                        help="Number of known breed images to evaluate (default: 50)")
    parser.add_argument("--num-unknown", type=int, default=50,
                        help="Number of unknown breed images to evaluate (default: 50)")
    parser.add_argument("--metric", choices=["composite", "msp", "margin", "entropy"], default="composite",
                        help="Uncertainty metric to test (default: composite)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output-json", type=Path,
                        default=REPO_ROOT / "report" / "unknown_breed_detection_benchmark.json",
                        help="Output JSON summary path")
    return parser.parse_args()


def load_evaluation_data(num_known: int, num_unknown: int, seed: int) -> Tuple[List[dict], List[dict]]:
    print("Loading Stanford Dogs dataset for Unknown Breed evaluation...")
    ds = load_dataset("maurice-fp/stanford-dogs")
    test_ds = ds["test"]
    names = test_ds.features["label"].names

    known_pool: Dict[int, List[object]] = defaultdict(list)
    unknown_pool: Dict[int, List[object]] = defaultdict(list)

    for row in test_ds:
        lbl = row["label"]
        img = row["image"]
        if img.mode != "RGB":
            img = img.convert("RGB")

        if lbl in OVERLAPPING_ST_IDS:
            known_pool[lbl].append(img)
        else:
            unknown_pool[lbl].append(img)

    rng = random.Random(seed)

    # Lấy mẫu phân bổ đa dạng các giống
    known_items = []
    known_keys = list(known_pool.keys())
    rng.shuffle(known_keys)
    while len(known_items) < num_known:
        for k in known_keys:
            if known_pool[k] and len(known_items) < num_known:
                img = known_pool[k].pop(rng.randint(0, len(known_pool[k]) - 1))
                clean_name = names[k].split("-")[-1].replace("_", " ").title()
                known_items.append({"image": img, "label": k, "breed_name": clean_name, "is_known_ground_truth": True})

    unknown_items = []
    unknown_keys = list(unknown_pool.keys())
    rng.shuffle(unknown_keys)
    while len(unknown_items) < num_unknown:
        for k in unknown_keys:
            if unknown_pool[k] and len(unknown_items) < num_unknown:
                img = unknown_pool[k].pop(rng.randint(0, len(unknown_pool[k]) - 1))
                clean_name = names[k].split("-")[-1].replace("_", " ").title()
                unknown_items.append({"image": img, "label": k, "breed_name": clean_name, "is_known_ground_truth": False})

    print(f"Loaded {len(known_items)} Known breed images & {len(unknown_items)} Unknown breed images.")
    return known_items, unknown_items


def main() -> None:
    args = parse_args()
    device = get_device()
    print(f"Inference device: {device}")

    known_items, unknown_items = load_evaluation_data(args.num_known, args.num_unknown, args.seed)

    models = {}
    for m in ModelMethod:
        if ClassifierRegistry.is_available(m):
            models[m] = ClassifierRegistry.get(m)

    print(f"Loaded {len(models)} models for evaluation.\n")

    benchmark_results = {
        "evaluation_name": "Novel / Unknown Breed Recognition Benchmark (Open-Set OOD)",
        "num_known_samples": len(known_items),
        "num_unknown_samples": len(unknown_items),
        "metric_tested": args.metric,
        "models": {},
        "qualitative_cases": [],
    }

    # Đánh giá từng mô hình
    for m, clf in models.items():
        m_name = MODEL_NAMES[m]
        print(f"\n=======================================================")
        print(f"Evaluating Model: {m_name}")
        print(f"=======================================================")

        # Tiền xử lý & dự đoán
        known_unknown_scores = []
        unknown_unknown_scores = []

        known_correct_decisions = 0  # Đoán đúng là Known
        unknown_correct_decisions = 0  # Đoán đúng là Unknown

        # 1. Đánh giá tập Known
        for item in tqdm(known_items, desc=f"Known [{m_name}]", leave=False):
            prep = prepare_batch_inputs(item["image"])
            res = clf.predict_from_preprocessed(prep, unknown_metric=args.metric)
            u_score = res.unknown_score
            known_unknown_scores.append(u_score)
            if not res.is_unknown_breed:
                known_correct_decisions += 1

        # 2. Đánh giá tập Unknown
        for item in tqdm(unknown_items, desc=f"Unknown [{m_name}]", leave=False):
            prep = prepare_batch_inputs(item["image"])
            res = clf.predict_from_preprocessed(prep, unknown_metric=args.metric)
            u_score = res.unknown_score
            unknown_unknown_scores.append(u_score)
            if res.is_unknown_breed:
                unknown_correct_decisions += 1

        # Tính toán các chỉ số
        tpr_known = (known_correct_decisions / len(known_items)) * 100.0
        tnr_unknown = (unknown_correct_decisions / len(unknown_items)) * 100.0
        total_acc = ((known_correct_decisions + unknown_correct_decisions) / (len(known_items) + len(unknown_items))) * 100.0

        auroc = calculate_auroc(pos_scores=unknown_unknown_scores, neg_scores=known_unknown_scores) * 100.0
        fpr95 = calculate_fpr95(pos_scores=unknown_unknown_scores, neg_scores=known_unknown_scores) * 100.0

        mean_u_known = float(np.mean(known_unknown_scores))
        mean_u_unknown = float(np.mean(unknown_unknown_scores))
        threshold_used = clf.unknown_detector.get_threshold(m.value, args.metric)

        print(f"  - Ngưỡng phân định áp dụng: {threshold_used:.3f}")
        print(f"  - Điểm bất định trung bình: Known = {mean_u_known:.3f} | Unknown = {mean_u_unknown:.3f}")
        print(f"  - Độ chính xác phát hiện giống đã biết (Known Accuracy): {tpr_known:.2f}%")
        print(f"  - Độ chính xác phát hiện giống chưa biết (Unknown Detection): {tnr_unknown:.2f}%")
        print(f"  - Độ chính xác tổng hợp (Overall Detection Accuracy): {total_acc:.2f}%")
        print(f"  - AUROC (Area Under ROC Curve): {auroc:.2f}%")
        print(f"  - FPR@95%TPR: {fpr95:.2f}%")

        benchmark_results["models"][m.value] = {
            "model_name": m_name,
            "threshold_used": threshold_used,
            "mean_uncertainty_known": round(mean_u_known, 4),
            "mean_uncertainty_unknown": round(mean_u_unknown, 4),
            "known_accuracy_pct": round(tpr_known, 2),
            "unknown_detection_pct": round(tnr_unknown, 2),
            "overall_accuracy_pct": round(total_acc, 2),
            "auroc_pct": round(auroc, 2),
            "fpr95_pct": round(fpr95, 2),
        }

    # Thu thập một số ca định tính cụ thể với Dual-Stream EfficientNet-B0
    dual_clf = models.get(ModelMethod.DUAL_STREAM_EFFICIENTNET)
    if dual_clf:
        print("\n\n--- CÁC CA ĐỊNH TÍNH MINH HỌA (DUAL-STREAM EFFICIENTNET-B0) ---")
        demo_unknown_breeds = ["Golden Retriever", "German Shepherd", "Rottweiler", "Siberian Husky", "French Bulldog"]
        demo_known_breeds = ["Beagle", "Saint Bernard", "Yorkshire Terrier", "Chihuahua"]

        for item in unknown_items:
            if item["breed_name"] in demo_unknown_breeds:
                prep = prepare_batch_inputs(item["image"])
                res = dual_clf.predict_from_preprocessed(prep)
                closest = BREED_CATALOGUE[res.predicted_breed_id - 1][1]
                case_info = {
                    "type": "UNKNOWN_BREED",
                    "actual_breed": item["breed_name"],
                    "is_unknown_detected": res.is_unknown_breed,
                    "status_label_vi": res.unknown_assessment.get("status_label_vi", "") if res.unknown_assessment else "",
                    "unknown_score": res.unknown_score,
                    "closest_resembling_known_breed": closest,
                    "resembling_score": res.score,
                    "explanation": res.unknown_assessment.get("summary", "") if res.unknown_assessment else "",
                }
                benchmark_results["qualitative_cases"].append(case_info)
                print(f"[UNKNOWN] Thực tế: {item['breed_name']:<18} | Phát hiện: {'Giống Chưa Biết' if res.is_unknown_breed else 'Giống Đã Biết':<16} | Điểm bất định: {res.unknown_score:.2%} | Giống tương đồng gần nhất: {closest} ({res.score:.2%})")
                demo_unknown_breeds.remove(item["breed_name"])
            if not demo_unknown_breeds:
                break

        for item in known_items:
            if item["breed_name"] in demo_known_breeds:
                prep = prepare_batch_inputs(item["image"])
                res = dual_clf.predict_from_preprocessed(prep)
                predicted = BREED_CATALOGUE[res.predicted_breed_id - 1][1]
                case_info = {
                    "type": "KNOWN_BREED",
                    "actual_breed": item["breed_name"],
                    "is_unknown_detected": res.is_unknown_breed,
                    "status_label_vi": res.unknown_assessment.get("status_label_vi", "") if res.unknown_assessment else "",
                    "unknown_score": res.unknown_score,
                    "predicted_breed": predicted,
                    "prediction_score": res.score,
                    "explanation": res.unknown_assessment.get("summary", "") if res.unknown_assessment else "",
                }
                benchmark_results["qualitative_cases"].append(case_info)
                print(f"[KNOWN]   Thực tế: {item['breed_name']:<18} | Phát hiện: {'Giống Đã Biết' if not res.is_unknown_breed else 'Giống Chưa Biết':<16} | Điểm bất định: {res.unknown_score:.2%} | Dự đoán: {predicted} ({res.score:.2%})")
                demo_known_breeds.remove(item["breed_name"])
            if not demo_known_breeds:
                break

    # Lưu kết quả
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(benchmark_results, f, indent=2, ensure_ascii=False)
    print(f"\n[OK] Đã lưu kết quả thực nghiệm nhận diện giống lạ vào: {args.output_json}")


if __name__ == "__main__":
    main()
