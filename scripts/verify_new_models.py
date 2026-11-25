"""Verify the 5 custom CNN classifiers on 300 Oxford-IIIT Pet test images."""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import torch
from torchvision.datasets import OxfordIIITPet

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "backend"))

from app.ai.cnn_classifiers import (  # noqa: E402
    DenseNet121Classifier,
    DualStreamEfficientNetClassifier,
    EfficientNetB0Classifier,
    ResNet18Classifier,
    SEResNet18Classifier,
)
from app.ai.common import BREED_CATALOGUE  # noqa: E402

DEFAULT_DATASET_DIR = Path("~/.cache/oxford_iiit_pet_eval").expanduser()


def check_class_alignment(tv_classes: list[str]) -> bool:
    """Compare torchvision 37 classes (lowercase) against BREED_CATALOGUE (lowercase)."""
    if len(tv_classes) != len(BREED_CATALOGUE):
        print(f"[MISMATCH] Class count differs: torchvision={len(tv_classes)} vs BREED_CATALOGUE={len(BREED_CATALOGUE)}")
        return False

    mismatches = []
    for idx, (tv_name, (_, cat_name, _, _)) in enumerate(zip(tv_classes, BREED_CATALOGUE)):
        tv_norm = tv_name.replace("_", " ").strip().lower()
        cat_norm = cat_name.replace("_", " ").strip().lower()
        if tv_norm != cat_norm:
            mismatches.append((idx, tv_name, cat_name))

    if mismatches:
        print("[MISMATCH] Found misaligned classes between torchvision and BREED_CATALOGUE:")
        for idx, tv_name, cat_name in mismatches:
            print(f"  index={idx:2d} | torchvision={tv_name!r} | BREED_CATALOGUE={cat_name!r}")
        return False

    print("[OK] All 37 torchvision classes match BREED_CATALOGUE order (case-insensitive).")
    return True


def sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def evaluate_model(
    name: str,
    classifier,
    samples: list[tuple],
    use_tta: bool,
    warmup_runs: int = 10,
) -> dict:
    """Run inference on samples and compute Top-1, Top-5 accuracy, and mean latency."""
    device = classifier._device

    # Warm-up passes so MPS/CUDA kernel compilation is excluded from timing.
    warmup_img = samples[0][0]
    for _ in range(warmup_runs):
        _ = classifier.predict(warmup_img, use_tta=use_tta)
    sync_device(device)

    top1_correct = 0
    top5_correct = 0
    latencies_ms = []

    for img, target_idx in samples:
        target_breed_id = BREED_CATALOGUE[target_idx][0]
        sync_device(device)
        t0 = time.perf_counter()
        result = classifier.predict(img, use_tta=use_tta)
        sync_device(device)
        t1 = time.perf_counter()

        latencies_ms.append((t1 - t0) * 1000.0)
        if result.predicted_breed_id == target_breed_id:
            top1_correct += 1
        top5_ids = [item.breed_id for item in result.top_k_results[:5]]
        if target_breed_id in top5_ids:
            top5_correct += 1

    total = len(samples)
    return {
        "model": name,
        "mode": "4-View TTA" if use_tta else "Single-View",
        "top1": (top1_correct / total) * 100.0,
        "top5": (top5_correct / total) * 100.0,
        "mean_ms": sum(latencies_ms) / total,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--num-samples", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.dataset_dir.mkdir(parents=True, exist_ok=True)
    print(f"Dataset directory (outside repo): {args.dataset_dir}")

    dataset = OxfordIIITPet(root=str(args.dataset_dir), split="test", download=True)
    aligned = check_class_alignment(list(dataset.classes))
    if not aligned:
        raise SystemExit("Class alignment check failed; inspect mismatches above.")

    rng = random.Random(args.seed)
    indices = rng.sample(range(len(dataset)), k=min(args.num_samples, len(dataset)))
    samples = [dataset[i] for i in indices]
    print(f"Sampled {len(samples)} images from Oxford-IIIT Pet test set (seed={args.seed}).\n")

    specs = [
        ("ResNet18", ResNet18Classifier),
        ("SEResNet18", SEResNet18Classifier),
        ("DenseNet121", DenseNet121Classifier),
        ("EfficientNet-B0", EfficientNetB0Classifier),
        ("Dual-Stream EfficientNet-B0", DualStreamEfficientNetClassifier),
    ]

    rows = []
    for label, cls in specs:
        classifier = cls(top_k=5)
        for use_tta in (False, True):
            metrics = evaluate_model(label, classifier, samples, use_tta=use_tta)
            rows.append(metrics)

    print(f"\n{'Model':<28} | {'Mode':<11} | {'Top-1 (%)':>9} | {'Top-5 (%)':>9} | {'Latency (ms)':>12}")
    print("-" * 80)
    for row in rows:
        print(
            f"{row['model']:<28} | {row['mode']:<11} | "
            f"{row['top1']:>8.2f}% | {row['top5']:>8.2f}% | {row['mean_ms']:>11.2f}"
        )


if __name__ == "__main__":
    main()
