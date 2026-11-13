#!/usr/bin/env python3
"""
Training & Evaluation Pipeline for Oxford-IIIT Pet Classification.
Features:
- Step 1: Libraries & Device setup (Apple Silicon MPS / CUDA / CPU)
- Step 2: Argparse configuration
- Step 3: Dataset loading & metadata printing
- Step 4: Model selection from scratch (10 architectures)
- Step 5: Training with per-epoch checkpointing & resume, with exact output format:
          Epoch 01 | Train Loss: 1.951 | Train Acc: 29.82% | Test Loss: 1.560 | Test Acc: 40.99%
- Step 6: Visualizations & figure exports (Accuracy/Loss curves, Confusion Matrix, Predictions)
"""

import os
import sys
import json
import time
import argparse
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

try:
    from src.dataset import OxfordPetDataset, get_transforms
    from src.models import build_model, count_parameters, SUPPORTED_MODELS
except ImportError:
    from dataset import OxfordPetDataset, get_transforms
    from models import build_model, count_parameters, SUPPORTED_MODELS


def get_device():
    """Detects and returns the fastest available compute device (MPS > CUDA > CPU)."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def parse_args(args_list=None):
    """Parses command line arguments or accepts custom list for notebook usage."""
    parser = argparse.ArgumentParser(
        description="Train Oxford-IIIT Pet Classifier from Scratch with Part-Level Augmentation"
    )
    # Dataset & Task
    parser.add_argument("--data_dir", type=str, default="dataset/oxford-iiit-pet",
                        help="Path to root directory of Oxford-IIIT Pet dataset")
    parser.add_argument("--task", type=str, choices=["breed", "binary"], default="breed",
                        help="Task type: 'breed' (37 classes) or 'binary' (Cat vs Dog: 2 classes)")
    parser.add_argument("--img_size", type=int, default=224,
                        help="Input image resolution (default: 224)")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Mini-batch size (default: 64)")
    parser.add_argument("--num_workers", type=int, default=2,
                        help="Number of DataLoader workers (default: 2)")

    # Data Augmentation (Teacher's requirement + CLIP-Animal-Classifier reuse)
    parser.add_argument("--part_aug", action="store_true", default=True,
                        help="Enable teacher's part-crop augmentation (ear, tail, paw crops)")
    parser.add_argument("--clahe", action="store_true", default=True,
                        help="Enable CLAHE and Bilateral filter enhancement from CLIP-Animal-Classifier")

    # Model & Optimization
    parser.add_argument("--model", type=str, default="resnet18", choices=SUPPORTED_MODELS,
                        help=f"Model architecture to train from scratch. Choices: {SUPPORTED_MODELS}")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Total number of epochs (default: 50)")
    parser.add_argument("--lr", type=float, default=5e-4,
                        help="Initial learning rate (default: 0.0005)")
    parser.add_argument("--weight_decay", type=float, default=1e-2,
                        help="Weight decay for regularization (default: 0.01)")
    parser.add_argument("--label_smoothing", type=float, default=0.1,
                        help="Label smoothing for CrossEntropyLoss (default: 0.1)")

    # Checkpointing & Resume
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints",
                        help="Directory to save and load per-epoch checkpoints")
    parser.add_argument("--no_resume", action="store_true", default=False,
                        help="Do not auto-resume from existing checkpoint; start from epoch 1")
    parser.add_argument("--plots_dir", type=str, default="plots",
                        help="Directory to export metric curves and figures")

    if args_list is not None:
        return parser.parse_args(args_list)
    return parser.parse_args()


# ==============================================================================
# Step 3: Load Data & Print Information
# ==============================================================================
def load_data(args):
    """Loads Oxford-IIIT Pet trainval and test datasets and prints detailed stats."""
    train_transform = get_transforms(
        img_size=args.img_size,
        is_train=True,
        use_part_aug=args.part_aug,
        use_clahe=args.clahe
    )
    test_transform = get_transforms(
        img_size=args.img_size,
        is_train=False
    )

    if getattr(args, 'model', '') == 'clip':
        from src.dataset import OxfordPetCLIPDataset
        from src.models import SimpleTokenizer
        classes_ref = OxfordPetDataset(root_dir=args.data_dir, split="trainval", task=args.task).class_names
        tok = SimpleTokenizer(breeds=classes_ref if args.task == "breed" else ["Cat", "Dog"])
        train_dataset = OxfordPetCLIPDataset(
            tok,
            root_dir=args.data_dir,
            split="trainval",
            task=args.task,
            transform=train_transform,
            is_train=True
        )
    else:
        train_dataset = OxfordPetDataset(
            root_dir=args.data_dir,
            split="trainval",
            task=args.task,
            transform=train_transform
        )
    test_dataset = OxfordPetDataset(
        root_dir=args.data_dir,
        split="test",
        task=args.task,
        transform=test_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )

    num_classes = 37 if args.task == "breed" else 2
    class_names = train_dataset.class_names if args.task == "breed" else ["Cat", "Dog"]

    print("=" * 70)
    print("DATASET INFORMATION SUMMARY")
    print("=" * 70)
    print(f"Dataset root       : {args.data_dir}")
    print(f"Classification task: {args.task.upper()} ({num_classes} classes)")
    print(f"Train samples      : {len(train_dataset):,}")
    print(f"Test samples       : {len(test_dataset):,}")
    print(f"Total samples      : {len(train_dataset) + len(test_dataset):,}")
    print(f"Batch size         : {args.batch_size} (Train batches: {len(train_loader)}, Test batches: {len(test_loader)})")
    print(f"Image resolution   : {args.img_size}x{args.img_size}")
    print(f"Augmentations      : PartCrop={args.part_aug}, CLAHE/Bilateral={args.clahe}")
    print("=" * 70)

    return train_dataset, test_dataset, train_loader, test_loader, num_classes, class_names


# ==============================================================================
# Step 5: Training Engine with Checkpoint & Resume
# ==============================================================================
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """Runs one training epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch in dataloader:
        if len(batch) >= 3 and torch.is_tensor(batch[1]) and batch[1].dtype == torch.long and batch[1].dim() == 2:
            inputs = batch[0].to(device)
            text_tokens = batch[1].to(device)
            targets = batch[2].to(device)

            optimizer.zero_grad()
            logits_i, logits_t = model(inputs, text_tokens)
            labels = torch.arange(len(inputs), device=device)
            loss = (F.cross_entropy(logits_i, labels) + F.cross_entropy(logits_t, labels)) / 2.0
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            with torch.no_grad():
                preds = model(inputs).max(1)[1]
                correct += preds.eq(targets).sum().item()
            total += targets.size(0)
            continue

        inputs, targets, _ = batch
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(targets).sum().item()
        total += targets.size(0)

    epoch_loss = running_loss / total
    epoch_acc = (correct / total) * 100.0
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    """Evaluates the model on test/val set."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []

    for batch in dataloader:
        if len(batch) >= 3 and torch.is_tensor(batch[1]) and batch[1].dtype == torch.long and batch[1].dim() == 2:
            inputs = batch[0].to(device)
            text_tokens = batch[1].to(device)
            targets = batch[2].to(device)

            logits_i, logits_t = model(inputs, text_tokens)
            labels = torch.arange(len(inputs), device=device)
            loss = (F.cross_entropy(logits_i, labels) + F.cross_entropy(logits_t, labels)) / 2.0

            running_loss += loss.item() * inputs.size(0)
            preds = model(inputs).max(1)[1]
            correct += preds.eq(targets).sum().item()
            total += targets.size(0)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_targets.extend(targets.cpu().numpy().tolist())
            continue

        inputs, targets, _ = batch
        inputs = inputs.to(device)
        targets = targets.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        running_loss += loss.item() * inputs.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(targets).sum().item()
        total += targets.size(0)

        all_preds.extend(preds.cpu().numpy().tolist())
        all_targets.extend(targets.cpu().numpy().tolist())

    epoch_loss = running_loss / total
    epoch_acc = (correct / total) * 100.0
    return epoch_loss, epoch_acc, np.array(all_preds), np.array(all_targets)


def train_model(model, train_loader, test_loader, args, device):
    """
    Trains model for `args.epochs` epochs.
    Saves checkpoint after each epoch.
    Supports resuming from previously saved checkpoint to save time.
    Prints epoch log strictly matching requested format:
    Epoch 01 | Train Loss: 1.951 | Train Acc: 29.82% | Test Loss: 1.560 | Test Acc: 40.99%
    """
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    criterion = nn.CrossEntropyLoss(label_smoothing=getattr(args, 'label_smoothing', 0.1))
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    latest_ckpt_path = os.path.join(args.checkpoint_dir, f"{args.model}_{args.task}_latest.pth")
    history_json_path = os.path.join(args.checkpoint_dir, f"{args.model}_{args.task}_history.json")

    start_epoch = 1
    best_acc = 0.0
    history = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": []
    }

    # Auto-resume check
    if not args.no_resume and os.path.exists(latest_ckpt_path):
        print(f"\n[INFO] Found existing checkpoint: {latest_ckpt_path}")
        checkpoint = torch.load(latest_ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint and checkpoint["scheduler_state_dict"]:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_acc = checkpoint.get("best_acc", 0.0)
        history = checkpoint.get("history", history)
        print(f"[RESUME] Resuming training from Epoch {start_epoch:02d} to {args.epochs:02d} (Best Acc so far: {best_acc:.2f}%)\n")
    else:
        print(f"\n[INFO] Starting fresh training from scratch for {args.epochs} epochs...\n")

    if start_epoch > args.epochs:
        print(f"[INFO] Model already trained for {args.epochs} epochs. Skipping training loop.")
        return history, best_acc

    for epoch in range(start_epoch, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc, _, _ = evaluate(model, test_loader, criterion, device)
        scheduler.step()

        # Update history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)

        is_best = test_acc > best_acc
        if is_best:
            best_acc = test_acc

        # Required print format:
        # Epoch 01 | Train Loss: 1.951 | Train Acc: 29.82% | Test Loss: 1.560 | Test Acc: 40.99%
        print(f"Epoch {epoch:02d} | Train Loss: {train_loss:.3f} | Train Acc: {train_acc:.2f}% | Test Loss: {test_loss:.3f} | Test Acc: {test_acc:.2f}%")

        # Save per-epoch checkpoint
        checkpoint_data = {
            "epoch": epoch,
            "model_name": args.model,
            "task": args.task,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "train_loss": train_loss,
            "train_acc": train_acc,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "best_acc": best_acc,
            "history": history
        }
        # Best model checkpoint (only saves best performing epoch)
        if is_best or epoch == 1:
            best_ckpt_path = os.path.join(args.checkpoint_dir, f"{args.model}_{args.task}_best.pth")
            torch.save(checkpoint_data, best_ckpt_path)

        # Latest checkpoint (used to resume)
        torch.save(checkpoint_data, latest_ckpt_path)

        # Save history as JSON for easy external inspection
        with open(history_json_path, "w") as f:
            json.dump(history, f, indent=2)

    print(f"\n[DONE] Training complete. Best Test Accuracy: {best_acc:.2f}%")
    return history, best_acc


# ==============================================================================
# Step 6: Visualizations & Plotting Figures
# ==============================================================================
def plot_metrics(history, model_name, task, save_dir="plots"):
    """Plots and exports Loss and Accuracy curves."""
    os.makedirs(save_dir, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss Curve
    axes[0].plot(epochs, history["train_loss"], "o-", color="#2b5c8f", label="Train Loss", linewidth=2)
    axes[0].plot(epochs, history["test_loss"], "s-", color="#e76f51", label="Test Loss", linewidth=2)
    axes[0].set_title(f"{model_name.upper()} - Cross-Entropy Loss", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Epoch", fontsize=11)
    axes[0].set_ylabel("Loss", fontsize=11)
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[0].legend(fontsize=11)

    # Accuracy Curve
    axes[1].plot(epochs, history["train_acc"], "o-", color="#2a9d8f", label="Train Acc (%)", linewidth=2)
    axes[1].plot(epochs, history["test_acc"], "s-", color="#e76f51", label="Test Acc (%)", linewidth=2)
    axes[1].set_title(f"{model_name.upper()} - Top-1 Accuracy (%)", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Epoch", fontsize=11)
    axes[1].set_ylabel("Accuracy (%)", fontsize=11)
    axes[1].grid(True, linestyle="--", alpha=0.6)
    axes[1].legend(fontsize=11)

    plt.tight_layout()
    save_path = os.path.join(save_dir, f"{model_name}_{task}_curves.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[FIGURE] Saved training curves figure to: {save_path}")
    return save_path


def plot_confusion_matrix_figure(y_true, y_pred, class_names, model_name, task, save_dir="plots"):
    """Plots and exports publication-ready confusion matrix with all breed labels."""
    os.makedirs(save_dir, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    display_names = [name.replace('_', ' ').title() for name in class_names]

    fig, ax = plt.subplots(figsize=(15, 13) if len(class_names) > 2 else (6, 5))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Số lượng ảnh', fontweight='bold', fontsize=11)

    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=display_names, yticklabels=display_names)
    plt.setp(ax.get_xticklabels(), rotation=90, ha="center", fontsize=8)
    plt.setp(ax.get_yticklabels(), fontsize=8)

    ax.set_title(f"HÌNH: MA TRẬN NHẦM LẪN (CONFUSION MATRIX) - {model_name.upper()} TRÊN {len(class_names)} GIỐNG PETS", fontsize=12, fontweight="bold", pad=15)
    ax.set_ylabel("Nhãn Thực Tế (Ground Truth Breed)", fontsize=11, fontweight="bold", labelpad=10)
    ax.set_xlabel("Nhãn Dự Đoán (Predicted Breed)", fontsize=11, fontweight="bold", labelpad=10)
    plt.tight_layout()

    save_path = os.path.join(save_dir, f"{model_name}_{task}_confusion_matrix.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[FIGURE] Saved confusion matrix to: {save_path}")
    return save_path


def main():
    args = parse_args()
    device = get_device()
    print(f"\n[HARDWARE] Selected device: {device.type.upper()}")

    # Step 3: Load Data
    train_dataset, test_dataset, train_loader, test_loader, num_classes, class_names = load_data(args)

    # Step 4: Build Model from Scratch
    print(f"\n[MODEL] Building '{args.model}' from scratch (pretrained=False)...")
    model = build_model(args.model, num_classes=num_classes).to(device)
    if args.model == "clip":
        model.build_classifier_weights(class_names, device=device)
    total_params, trainable_params = count_parameters(model)
    print(f"Total Parameters     : {total_params:,}")
    print(f"Trainable Parameters : {trainable_params:,}")

    # Step 5: Training Loop
    history, best_acc = train_model(model, train_loader, test_loader, args, device)

    # Step 6: Generate Figures
    if len(history["train_loss"]) > 0:
        plot_metrics(history, args.model, args.task, save_dir=args.plots_dir)

        # Final evaluation for confusion matrix (loads best checkpoint)
        print("\n[EVALUATION] Generating final test predictions for confusion matrix...")
        best_ckpt_path = os.path.join(args.checkpoint_dir, f"{args.model}_{args.task}_best.pth")
        if os.path.exists(best_ckpt_path):
            best_data = torch.load(best_ckpt_path, map_location=device)
            model.load_state_dict(best_data["model_state_dict"])
            print(f"[EVALUATION] Loaded best model weights from Epoch {best_data['epoch']} (Best Acc: {best_data['best_acc']:.2f}%)")
        criterion = nn.CrossEntropyLoss()
        _, _, y_pred, y_true = evaluate(model, test_loader, criterion, device)
        plot_confusion_matrix_figure(y_true, y_pred, class_names, args.model, args.task, save_dir=args.plots_dir)


if __name__ == "__main__":
    main()

