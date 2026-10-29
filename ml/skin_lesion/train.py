import os
import sys
import argparse
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Ensure repository root is on sys.path
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from ml.skin_lesion.config import SkinLesionConfig
from ml.skin_lesion.model import build_skin_lesion_model
from ml.skin_lesion.dataset import (
    validate_dataset_presence,
    split_lesions_zero_leakage,
    get_skin_transforms,
    HAM10000Dataset,
)
from ml.skin_lesion.evaluate import evaluate_model, save_metrics


def train_skin_lesion_model(
    config: Optional[SkinLesionConfig] = None,
    epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    lr: Optional[float] = None,
    pretrained: bool = False,
) -> Dict[str, Any]:
    """
    Executes resource-safe CPU training for Skin Lesion Analysis.
    Preserves existing artifacts and writes checkpoints to config.artifacts_dir.
    """
    cfg = config or SkinLesionConfig()
    epochs = epochs or cfg.epochs
    batch_size = batch_size or cfg.batch_size
    lr = lr or cfg.lr
    device = cfg.device

    print(f"[MediVision-AI] Initializing Skin Lesion training pipeline on {device.upper()}...")
    print(f"[MediVision-AI] Target artifacts directory: {cfg.artifacts_dir}")

    # 1. Dataset verification
    status = validate_dataset_presence(cfg)
    if not status["available"]:
        print(f"[MediVision-AI] Error: {status['message']}")
        return {
            "status": "dataset_missing",
            "message": status["message"],
        }

    # 2. Ingest metadata and perform zero-leakage split
    meta_df = pd.read_csv(cfg.get_metadata_path())
    print(f"[MediVision-AI] Ingested {len(meta_df)} metadata rows across {meta_df['lesion_id'].nunique()} unique lesions.")

    train_df, val_df = split_lesions_zero_leakage(meta_df, val_ratio=0.2, seed=42)
    print(f"[MediVision-AI] Patient-level split: {len(train_df)} train images, {len(val_df)} val images (0% lesion leakage).")

    train_dataset = HAM10000Dataset(train_df, cfg.get_images_dir(), transform=get_skin_transforms(cfg.image_size, is_train=True))
    val_dataset = HAM10000Dataset(val_df, cfg.get_images_dir(), transform=get_skin_transforms(cfg.image_size, is_train=False))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=cfg.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=cfg.num_workers)

    # 3. Model setup
    model = build_skin_lesion_model(num_classes=cfg.num_classes, pretrained=pretrained)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    cfg.artifacts_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")
    best_metrics = None

    print(f"[MediVision-AI] Starting training: {epochs} epochs, batch_size={batch_size}, lr={lr}...")

    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (images, targets) in enumerate(train_loader):
            images = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

        epoch_train_loss = running_loss / max(total, 1)
        epoch_train_acc = correct / max(total, 1)

        # Validation evaluation
        val_eval = evaluate_model(model, val_loader, criterion, device=device)
        val_loss = val_eval["val_loss"]
        val_acc = val_eval["accuracy"]
        val_bal_acc = val_eval["balanced_accuracy"]

        print(
            f"Epoch {epoch:02d}/{epochs:02d} | "
            f"Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} BalAcc: {val_bal_acc:.4f}"
        )

        history.append({
            "epoch": epoch,
            "train_loss": epoch_train_loss,
            "train_acc": epoch_train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_balanced_acc": val_bal_acc,
        })

        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_metrics = val_eval
            best_metrics["epoch"] = epoch
            best_metrics["history"] = history
            best_metrics["model_architecture"] = "EfficientNet-B0"
            best_metrics["checkpoint_path"] = str(cfg.get_checkpoint_path())

            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_acc": val_acc,
                "num_classes": cfg.num_classes,
                "classes": cfg.num_classes,
            }, cfg.get_checkpoint_path())

            save_metrics(best_metrics, cfg.get_metrics_path())
            print(f"  --> Saved new best checkpoint to {cfg.get_checkpoint_path()}")

        # Save last checkpoint
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "val_loss": val_loss,
        }, cfg.artifacts_dir / "last_checkpoint.pth")

    print("[MediVision-AI] Training complete.")
    return {
        "status": "success",
        "best_val_loss": best_val_loss,
        "checkpoint_path": str(cfg.get_checkpoint_path()),
        "metrics_path": str(cfg.get_metrics_path()),
        "metrics": best_metrics,
    }


def main():
    parser = argparse.ArgumentParser(description="MediVision-AI Skin Lesion Training CLI")
    parser.add_argument("--verify-only", action="store_true", help="Only verify dataset availability and structure")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=2, help="Training batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--pretrained", action="store_true", help="Initialize from pretrained ImageNet weights")
    args = parser.parse_args()

    cfg = SkinLesionConfig()
    if args.verify_only:
        res = validate_dataset_presence(cfg)
        print(f"[Dataset Verification] Available: {res['available']}")
        print(f"[Dataset Verification] Message: {res['message']}")
        sys.exit(0 if res["available"] else 1)

    result = train_skin_lesion_model(
        config=cfg,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        pretrained=args.pretrained,
    )
    if result.get("status") != "success":
        sys.exit(1)


if __name__ == "__main__":
    main()
