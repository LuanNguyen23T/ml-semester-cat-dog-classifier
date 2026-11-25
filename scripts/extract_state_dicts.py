"""Extract lightweight model_state_dict files from full training checkpoints.

Original *_best.pth files in backend/weights/ are kept untouched.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "backend"))

from app.ai.architectures import (  # noqa: E402
    CustomSEResNet,
    DenseNet121,
    DualStreamEfficientNet,
    EfficientNetB0,
    ResNet18,
)
from app.ai.common import WEIGHTS_DIR  # noqa: E402

EXPORT_SPECS = [
    ("ResNet18", "resnet18_augmented_best.pth", "resnet18_state_dict.pth", lambda: ResNet18(num_classes=37)),
    ("SEResNet18", "seresnet18_augmented_best.pth", "seresnet18_state_dict.pth", lambda: CustomSEResNet(num_classes=37)),
    ("DenseNet121", "densenet121_augmented_best.pth", "densenet121_state_dict.pth", lambda: DenseNet121(num_classes=37)),
    ("EfficientNet-B0", "efficientnet_b0_augmented_best.pth", "efficientnet_b0_state_dict.pth", lambda: EfficientNetB0(num_classes=37)),
    ("Dual-Stream EfficientNet-B0", "dual_stream_efficientnet_best.pth", "dual_stream_efficientnet_state_dict.pth", lambda: DualStreamEfficientNet(num_classes=37)),
]


def main() -> None:
    print(f"Weights directory: {WEIGHTS_DIR}\n")
    print(f"{'Model':<28} | {'Original (MB)':>13} | {'StateDict (MB)':>14} | {'strict=True'}")
    print("-" * 76)

    for label, src_name, dst_name, builder in EXPORT_SPECS:
        src_path = WEIGHTS_DIR / src_name
        dst_path = WEIGHTS_DIR / dst_name
        if not src_path.exists():
            raise FileNotFoundError(f"Missing original checkpoint: {src_path}")

        ckpt = torch.load(src_path, map_location="cpu", weights_only=False)
        if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt:
            raise KeyError(f"Checkpoint {src_name} does not contain 'model_state_dict'")

        state_dict = ckpt["model_state_dict"]
        torch.save(state_dict, dst_path)

        # Verify strict=True loading from the newly extracted file.
        model = builder()
        reloaded = torch.load(dst_path, map_location="cpu", weights_only=True)
        model.load_state_dict(reloaded, strict=True)
        model.eval()

        src_mb = src_path.stat().st_size / (1024 * 1024)
        dst_mb = dst_path.stat().st_size / (1024 * 1024)
        print(f"{label:<28} | {src_mb:>13.2f} | {dst_mb:>14.2f} | PASS")


if __name__ == "__main__":
    main()
