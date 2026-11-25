"""Kiểm thử tự động bộ ảnh demo trên mô hình Dual-Stream PetNet và EfficientNet-B0.

Đánh giá:
1. Nhóm 37 Giống Thú Cưng trong danh mục (10 ảnh/giống = 370 ảnh): Tỉ lệ dự đoán chính xác và độ tin cậy.
2. Nhóm Ảnh Không Liên Quan Chó Mèo (Động vật hoang dã, Đồ vật, Phương tiện): Kiểm tra phát hiện giống lạ.
3. Nhóm Giống Chó Lạ Ngoài Danh Mục (Siberian Husky, Golden Retriever,...): Kiểm tra phát hiện giống lạ.
"""

from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import torch
from PIL import Image
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "backend"))

from app.ai.common import BREED_CATALOGUE, get_device  # type: ignore # noqa: E402
from app.ai.preprocessing import prepare_batch_inputs  # type: ignore # noqa: E402
from app.ai.registry import ClassifierRegistry, ModelMethod  # type: ignore # noqa: E402

DEMO_DIR = REPO_ROOT / "demo_samples"
REPORT_PATH = REPO_ROOT / "report" / "demo_samples_evaluation.json"

NAME_TO_OXFORD_ID = {name.lower(): (idx + 1) for idx, (_, name, _, _) in enumerate(BREED_CATALOGUE)}


def evaluate_known_breeds(clf) -> dict:
    known_dir = DEMO_DIR / "known_breeds"
    breed_folders = sorted([f for f in known_dir.iterdir() if f.is_dir()])

    total_images = 0
    correct_top1 = 0
    correct_top5 = 0
    known_flagged_count = 0  # Bị nhận diện là Unknown
    per_breed_stats = {}

    print(f"\n[1/3] Đang kiểm thử 37 giống loài đã biết ({len(breed_folders)} thư mục)...")
    for folder in tqdm(breed_folders, desc="Đánh giá 37 giống"):
        breed_key = folder.name.lower()
        target_id = NAME_TO_OXFORD_ID.get(breed_key)
        images = sorted(list(folder.glob("*.jpg")) + list(folder.glob("*.png")))

        b_total = len(images)
        b_correct1 = 0
        b_correct5 = 0
        b_confidences = []

        for img_path in images:
            try:
                with Image.open(img_path) as opened:
                    pil_img = opened.convert("RGB")
                prep = prepare_batch_inputs(pil_img)
                res = clf.predict_from_preprocessed(prep)

                total_images += 1
                b_confidences.append(res.score)

                if res.predicted_breed_id == target_id:
                    correct_top1 += 1
                    b_correct1 += 1

                top5_ids = [item.breed_id for item in res.top_k_results[:5]]
                if target_id in top5_ids:
                    correct_top5 += 1
                    b_correct5 += 1

                if res.is_unknown_breed:
                    known_flagged_count += 1
            except Exception as exc:
                print(f"Lỗi đọc {img_path}: {exc}")

        acc1 = (b_correct1 / b_total * 100.0) if b_total > 0 else 0.0
        acc5 = (b_correct5 / b_total * 100.0) if b_total > 0 else 0.0
        mean_conf = (sum(b_confidences) / len(b_confidences)) if b_confidences else 0.0
        per_breed_stats[folder.name] = {
            "total_images": b_total,
            "top1_correct": b_correct1,
            "top1_accuracy": round(acc1, 2),
            "top5_accuracy": round(acc5, 2),
            "mean_confidence": round(mean_conf, 4),
        }

    overall_top1 = (correct_top1 / total_images * 100.0) if total_images > 0 else 0.0
    overall_top5 = (correct_top5 / total_images * 100.0) if total_images > 0 else 0.0

    return {
        "total_images": total_images,
        "top1_accuracy": round(overall_top1, 2),
        "top5_accuracy": round(overall_top5, 2),
        "flagged_as_unknown_rate": round(known_flagged_count / max(1, total_images) * 100.0, 2),
        "per_breed": per_breed_stats,
    }


def evaluate_non_pet_objects(clf) -> List[dict]:
    non_pet_dir = DEMO_DIR / "unknown_and_non_pet" / "non_pet_objects"
    if not non_pet_dir.exists():
        return []

    images = sorted(list(non_pet_dir.glob("*.jpg")) + list(non_pet_dir.glob("*.png")))
    results = []
    print(f"\n[2/3] Đang kiểm thử {len(images)} ảnh không phải chó mèo...")

    for img_path in images:
        try:
            with Image.open(img_path) as opened:
                pil_img = opened.convert("RGB")
            prep = prepare_batch_inputs(pil_img)
            res = clf.predict_from_preprocessed(prep)

            closest_breed = BREED_CATALOGUE[res.predicted_breed_id - 1][1]
            results.append({
                "image_name": img_path.name,
                "is_unknown_detected": res.is_unknown_breed,
                "status_label_vi": res.unknown_assessment.get("status_label_vi", "") if res.unknown_assessment else "",
                "unknown_score": res.unknown_score,
                "closest_resembling_breed": closest_breed,
                "resembling_score": res.score,
                "summary": res.unknown_assessment.get("summary", "") if res.unknown_assessment else "",
            })
        except Exception as exc:
            print(f"Lỗi đọc {img_path}: {exc}")

    return results


def evaluate_unseen_dog_breeds(clf) -> List[dict]:
    unseen_dir = DEMO_DIR / "unknown_and_non_pet" / "unseen_dog_breeds"
    if not unseen_dir.exists():
        return []

    images = sorted(list(unseen_dir.glob("*.jpg")) + list(unseen_dir.glob("*.png")))
    results = []
    print(f"\n[3/3] Đang kiểm thử {len(images)} ảnh giống chó lạ ngoài danh mục...")

    for img_path in images:
        try:
            with Image.open(img_path) as opened:
                pil_img = opened.convert("RGB")
            prep = prepare_batch_inputs(pil_img)
            res = clf.predict_from_preprocessed(prep)

            closest_breed = BREED_CATALOGUE[res.predicted_breed_id - 1][1]
            results.append({
                "image_name": img_path.name,
                "is_unknown_detected": res.is_unknown_breed,
                "status_label_vi": res.unknown_assessment.get("status_label_vi", "") if res.unknown_assessment else "",
                "unknown_score": res.unknown_score,
                "closest_resembling_breed": closest_breed,
                "resembling_score": res.score,
                "summary": res.unknown_assessment.get("summary", "") if res.unknown_assessment else "",
            })
        except Exception as exc:
            print(f"Lỗi đọc {img_path}: {exc}")

    return results


def main() -> None:
    clf = ClassifierRegistry.get(ModelMethod.DUAL_STREAM_EFFICIENTNET)
    print("Mô hình kiểm thử chính: Dual-Stream EfficientNet-B0 (PetNet)")

    known_eval = evaluate_known_breeds(clf)
    non_pet_eval = evaluate_non_pet_objects(clf)
    unseen_eval = evaluate_unseen_dog_breeds(clf)

    report = {
        "evaluation_name": "Web Demo Dataset Verification & Stress Test",
        "eval_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_used": "Dual-Stream EfficientNet-B0",
        "known_breeds_summary": known_eval,
        "non_pet_objects_test": non_pet_eval,
        "unseen_dog_breeds_test": unseen_eval,
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print("KẾT QUẢ KIỂM THỬ BỘ ẢNH DEMO THỰC TẾ TRÊN DUAL-STREAM PETNET")
    print("=" * 80)
    print(f"1. Tổng số ảnh giống đã biết (37 giống): {known_eval['total_images']} ảnh")
    print(f"   -> Top-1 Accuracy: {known_eval['top1_accuracy']}%")
    print(f"   -> Top-5 Accuracy: {known_eval['top5_accuracy']}%")
    print(f"2. Ảnh không phải chó mèo (Xe cộ, Đồ vật, Động vật hoang dã):")
    non_pet_unknown_count = sum(1 for item in non_pet_eval if item["is_unknown_detected"])
    print(f"   -> Tỉ lệ phát hiện Giống Chưa Xác Định: {non_pet_unknown_count}/{len(non_pet_eval)} ({non_pet_unknown_count/max(1, len(non_pet_eval))*100:.1f}%)")
    print(f"3. Ảnh giống chó lạ ngoài danh mục (Husky, Golden Retriever, Rottweiler,...):")
    unseen_unknown_count = sum(1 for item in unseen_eval if item["is_unknown_detected"])
    print(f"   -> Tỉ lệ phát hiện Giống Chưa Xác Định: {unseen_unknown_count}/{len(unseen_eval)} ({unseen_unknown_count/max(1, len(unseen_eval))*100:.1f}%)")
    print("=" * 80)
    print(f"[OK] Đã lưu báo cáo chi tiết vào: {REPORT_PATH}")


if __name__ == "__main__":
    main()
