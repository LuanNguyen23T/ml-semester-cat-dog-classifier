"""Cross-Dataset Zero-Shot Benchmark on Stanford Dogs (Overlapping Breeds).

Evaluates the 5 custom-trained CNN architectures on Stanford Dogs images
without retraining (Zero-Shot Cross-Dataset domain transfer).
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

import torch
from datasets import load_dataset
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "backend"))

from app.ai.common import BREED_CATALOGUE, get_device  # type: ignore # noqa: E402
from app.ai.preprocessing import prepare_batch_inputs  # type: ignore # noqa: E402
from app.ai.registry import ClassifierRegistry, ModelMethod  # type: ignore # noqa: E402

# 21 Overlapping dog breeds between Oxford-IIIT Pet (37 classes) and Stanford Dogs (120 classes)
# Format: (oxford_breed_id, oxford_name, stanford_label_id, stanford_label_name, display_name)
OVERLAPPING_BREEDS = [
    (4,  "basset_hound",               10,  "n02088238-basset",                       "Basset Hound"),
    (5,  "beagle",                     11,  "n02088364-beagle",                       "Beagle"),
    (9,  "boxer",                      91,  "n02108089-boxer",                        "Boxer"),
    (11, "chihuahua",                  0,   "n02085620-Chihuahua",                    "Chihuahua"),
    (13, "english_cocker_spaniel",     68,  "n02102318-cocker_spaniel",               "English Cocker Spaniel"),
    (14, "english_setter",             61,  "n02100735-English_setter",               "English Setter"),
    (15, "german_shorthaired",         59,  "n02100236-German_short-haired_pointer",  "German Shorthaired"),
    (16, "great_pyrenees",             105, "n02111500-Great_Pyrenees",               "Great Pyrenees"),
    (18, "japanese_chin",              1,   "n02085782-Japanese_spaniel",             "Japanese Chin"),
    (19, "keeshond",                   109, "n02112350-keeshond",                     "Keeshond"),
    (20, "leonberger",                 103, "n02111129-Leonberg",                     "Leonberger"),
    (22, "miniature_pinscher",         86,  "n02107312-miniature_pinscher",           "Miniature Pinscher"),
    (23, "newfoundland",               104, "n02111277-Newfoundland",                 "Newfoundland"),
    (25, "pomeranian",                 107, "n02112018-Pomeranian",                   "Pomeranian"),
    (26, "pug",                        102, "n02110958-pug",                          "Pug"),
    (29, "saint_bernard",              96,  "n02109525-Saint_Bernard",                "Saint Bernard"),
    (30, "samoyed",                    106, "n02111889-Samoyed",                      "Samoyed"),
    (31, "scottish_terrier",           48,  "n02097298-Scotch_terrier",               "Scottish Terrier"),
    (35, "staffordshire_bull_terrier", 28,  "n02093256-Staffordshire_bullterrier",    "Staffordshire Bull Terrier"),
    (36, "wheaten_terrier",            51,  "n02098105-soft-coated_wheaten_terrier",  "Wheaten Terrier"),
    (37, "yorkshire_terrier",          36,  "n02094433-Yorkshire_terrier",            "Yorkshire Terrier"),
]

# Baseline Top-1 / TTA accuracies on Oxford-IIIT Pet test set (from New Dataset training)
OXFORD_BASELINES = {
    ModelMethod.RESNET18: {"raw": 67.84, "tta": 68.47},
    ModelMethod.SERESNET18: {"raw": 70.62, "tta": 71.55},
    ModelMethod.DENSENET121: {"raw": 74.54, "tta": 75.33},
    ModelMethod.EFFICIENTNET_B0: {"raw": 83.32, "tta": 83.78},
    ModelMethod.DUAL_STREAM_EFFICIENTNET: {"raw": 85.72, "tta": 86.54},
}

MODEL_NAMES = {
    ModelMethod.RESNET18: "ResNet-18",
    ModelMethod.SERESNET18: "SEResNet-18",
    ModelMethod.DENSENET121: "DenseNet-121",
    ModelMethod.EFFICIENTNET_B0: "EfficientNet-B0",
    ModelMethod.DUAL_STREAM_EFFICIENTNET: "Dual-Stream EfficientNet-B0",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cross-Dataset Benchmark on Stanford Dogs")
    parser.add_argument("--split", choices=["test", "train", "all"], default="all",
                        help="Stanford Dogs split to benchmark (test, train, or all; default: all)")
    parser.add_argument("--samples-per-breed", type=int, default=0,
                        help="Number of samples per breed (0 for all available, default: 0)")
    parser.add_argument("--use-tta", action="store_true", default=False,
                        help="Evaluate with 4-View Test-Time Augmentation")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for sampling (default: 42)")
    parser.add_argument("--output-json", type=Path,
                        default=REPO_ROOT / "report" / "stanford_dogs_cross_dataset_benchmark.json",
                        help="Output JSON summary file")
    return parser.parse_args()


def load_stratified_stanford_dogs(split: str, samples_per_breed: int, seed: int) -> List[Tuple[object, int, str]]:
    print(f"Loading Stanford Dogs dataset (split: {split})...")
    ds = load_dataset("maurice-fp/stanford-dogs")

    if split == "all":
        subsets = [ds["train"], ds["test"]]
    else:
        subsets = [ds[split]]

    st_to_oxford = {st_id: (ox_id, display_name) for ox_id, _, st_id, _, display_name in OVERLAPPING_BREEDS}

    breed_buckets: Dict[int, List[object]] = defaultdict(list)
    for subset in subsets:
        for row in subset:
            st_label = row["label"]
            if st_label in st_to_oxford:
                img = row["image"]
                if img.mode != "RGB":
                    img = img.convert("RGB")
                breed_buckets[st_label].append(img)

    rng = random.Random(seed)
    sampled_items = []
    print(f"\nCollecting images across {len(OVERLAPPING_BREEDS)} breeds (split: {split}):")
    for ox_id, _, st_id, _, display_name in OVERLAPPING_BREEDS:
        available_imgs = breed_buckets[st_id]
        if samples_per_breed <= 0 or samples_per_breed >= len(available_imgs):
            chosen = available_imgs
        else:
            chosen = rng.sample(available_imgs, k=samples_per_breed)
        for img in chosen:
            sampled_items.append((img, ox_id, display_name))
        print(f"  - {display_name:<28}: {len(chosen):>4} images (out of {len(available_imgs)})")

    print(f"\nTotal images for evaluation: {len(sampled_items)}")
    return sampled_items


def main() -> None:
    args = parse_args()
    device = get_device()
    print(f"Inference device: {device}")

    samples = load_stratified_stanford_dogs(args.split, args.samples_per_breed, args.seed)
    if not samples:
        print("[ERROR] No samples loaded.")
        sys.exit(1)

    # Initialize all 5 models
    models = {}
    for m in ModelMethod:
        if not ClassifierRegistry.is_available(m):
            print(f"[CẢNH BÁO] Model {m.value} thiếu file weights trong backend/weights/")
            continue
        models[m] = ClassifierRegistry.get(m)

    print(f"\nLoaded {len(models)} models ready for zero-shot cross-dataset evaluation.")

    # Data structures for results
    results_by_model: Dict[ModelMethod, dict] = {
        m: {
            "top1_correct": 0,
            "top5_correct": 0,
            "latencies_ms": [],
            "per_breed_correct": defaultdict(int),
            "per_breed_total": defaultdict(int),
        }
        for m in models
    }

    print("\nRunning Zero-Shot Cross-Dataset Benchmark...")
    pbar = tqdm(samples, desc="Evaluating", unit="img")

    for img, target_oxford_id, breed_name in pbar:
        # Preprocess segmentation & head crop once per image
        t_prep_start = time.perf_counter()
        prep = prepare_batch_inputs(img)
        t_prep = time.perf_counter() - t_prep_start

        for m, clf in models.items():
            t0 = time.perf_counter()
            res = clf.predict_from_preprocessed(prep, use_tta=args.use_tta)
            t_infer = time.perf_counter() - t0

            # Record metrics
            results_by_model[m]["latencies_ms"].append(t_infer * 1000.0)
            results_by_model[m]["per_breed_total"][breed_name] += 1

            if res.predicted_breed_id == target_oxford_id:
                results_by_model[m]["top1_correct"] += 1
                results_by_model[m]["per_breed_correct"][breed_name] += 1

            top5_ids = [item.breed_id for item in res.top_k_results[:5]]
            if target_oxford_id in top5_ids:
                results_by_model[m]["top5_correct"] += 1

    total_samples = len(samples)

    # Compile final metrics
    summary_report = {
        "dataset": f"Stanford Dogs ({args.split.upper()} split, Zero-Shot Cross-Dataset Domain Transfer)",
        "split": args.split,
        "num_overlapping_breeds": len(OVERLAPPING_BREEDS),
        "total_test_samples": total_samples,
        "mode": "4-View TTA" if args.use_tta else "Single-View (Raw)",
        "device": str(device),
        "models": {},
    }

    print("\n" + "=" * 90)
    print(f"{'CROSS-DATASET ZERO-SHOT BENCHMARK RESULTS (STANFORD DOGS)':^90}")
    print("=" * 90)
    print(f"{'Mô Hình':<28} | {'Top-1 (%)':>10} | {'Top-5 (%)':>10} | {'Oxford Top-1':>12} | {'Delta Drop':>11} | {'Latency':>9}")
    print("-" * 90)

    for m, data in results_by_model.items():
        top1_acc = (data["top1_correct"] / total_samples) * 100.0
        top5_acc = (data["top5_correct"] / total_samples) * 100.0
        mean_ms = sum(data["latencies_ms"]) / total_samples

        baseline_acc = OXFORD_BASELINES[m]["tta" if args.use_tta else "raw"]
        delta_drop = baseline_acc - top1_acc

        model_name = MODEL_NAMES[m]
        print(
            f"{model_name:<28} | {top1_acc:>9.2f}% | {top5_acc:>9.2f}% | "
            f"{baseline_acc:>11.2f}% | {delta_drop:>10.2f}% | {mean_ms:>7.1f}ms"
        )

        per_breed_stats = {}
        for _, _, _, _, b_name in OVERLAPPING_BREEDS:
            correct = data["per_breed_correct"][b_name]
            tot = data["per_breed_total"][b_name]
            acc = (correct / tot * 100.0) if tot > 0 else 0.0
            per_breed_stats[b_name] = {"correct": correct, "total": tot, "accuracy": round(acc, 2)}

        summary_report["models"][m.value] = {
            "name": model_name,
            "top1_acc": round(top1_acc, 2),
            "top5_acc": round(top5_acc, 2),
            "oxford_baseline_acc": round(baseline_acc, 2),
            "delta_drop": round(delta_drop, 2),
            "mean_latency_ms": round(mean_ms, 2),
            "per_breed": per_breed_stats,
        }

    print("=" * 90)

    # Save output JSON
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(summary_report, f, indent=2, ensure_ascii=False)
    print(f"\n[OK] Đã lưu kết quả chi tiết vào: {args.output_json}")


if __name__ == "__main__":
    main()
