import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

_project_root = Path(__file__).resolve().parents[3]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from torchvision import models
except ImportError:
    torch = None
    nn = None
    DataLoader = None
    models = None

from ml.chest_xray.cnn.config import CNNConfig, default_config
from ml.chest_xray.cnn.dataset import (
    RSNAPneumoniaDataset,
    get_transforms,
    load_and_split_rsna_metadata,
    verify_dataset_presence,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("medivision.ml.train")


def set_seed(seed: int = 42):
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def build_resnet50_model(pretrained: bool = True, use_dropout: bool = False) -> "torch.nn.Module":
    """
    Constructs a ResNet50 model with transfer learning weights
    and a custom single-logit output head for binary classification.
    """
    if torch is None or models is None:
        raise ImportError("PyTorch and torchvision are required. Run `pip install torch torchvision`.")

    if pretrained:
        try:
            weights = models.ResNet50_Weights.DEFAULT
            model = models.resnet50(weights=weights)
        except Exception as exc:
            logger.warning(f"Could not download pre-trained weights ({exc}); initializing random weights.")
            model = models.resnet50(weights=None)
    else:
        model = models.resnet50(weights=None)

    in_features = model.fc.in_features
    if use_dropout:
        model.fc = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, 1),
        )
    else:
        model.fc = nn.Linear(in_features, 1)
    return model


def train_one_epoch(
    model: "torch.nn.Module",
    dataloader: "DataLoader",
    criterion: "torch.nn.Module",
    optimizer: "torch.optim.Optimizer",
    device: str,
) -> Tuple[float, float]:
    """Runs one training epoch, returning (avg_loss, accuracy)."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, targets, _ in dataloader:
        images = images.to(device)
        targets = targets.to(device).unsqueeze(1)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        probs = torch.sigmoid(outputs)
        preds = (probs >= 0.5).float()
        correct += (preds == targets).sum().item()
        total += targets.size(0)

    epoch_loss = running_loss / max(total, 1)
    epoch_acc = correct / max(total, 1)
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate_epoch(
    model: "torch.nn.Module",
    dataloader: "DataLoader",
    criterion: "torch.nn.Module",
    device: str,
) -> Tuple[float, float, float]:
    """Runs evaluation, returning (avg_loss, accuracy, auroc)."""
    from sklearn.metrics import roc_auc_score

    model.eval()
    running_loss = 0.0
    all_targets = []
    all_probs = []

    for images, targets, _ in dataloader:
        images = images.to(device)
        targets = targets.to(device).unsqueeze(1)

        outputs = model(images)
        loss = criterion(outputs, targets)

        running_loss += loss.item() * images.size(0)
        probs = torch.sigmoid(outputs)

        all_targets.extend(targets.cpu().numpy().flatten())
        all_probs.extend(probs.cpu().numpy().flatten())

    total = len(all_targets)
    epoch_loss = running_loss / max(total, 1)
    preds = (np.array(all_probs) >= 0.5).astype(float)
    epoch_acc = float((preds == np.array(all_targets)).mean())

    try:
        epoch_auroc = float(roc_auc_score(all_targets, all_probs))
    except Exception:
        epoch_auroc = 0.5  # If single class present in batch

    return epoch_loss, epoch_acc, epoch_auroc


def run_training(config: Optional[CNNConfig] = None) -> Optional[Dict]:
    """
    Executes the complete ResNet50 training pipeline.
    If the RSNA dataset is missing, gracefully logs the exact setup instructions
    and exits safely without throwing an unhandled exception.
    """
    cfg = config or default_config
    cfg.ensure_artifact_dir()
    set_seed(cfg.seed)

    # 1. Dataset presence check
    is_present, status_msg = verify_dataset_presence(cfg)
    if not is_present:
        logger.error("=" * 70)
        logger.error("[DATASET BLOCKED] Cannot start model training.")
        logger.error(status_msg)
        logger.error("=" * 70)
        return {
            "status": "dataset_missing",
            "message": status_msg,
            "required_dir": str(cfg.dataset_dir),
        }

    device = cfg.get_device()
    logger.info(f"[MediVision AI] Starting ResNet50 training on device: {device}")

    # 2. Split dataset by patientId
    train_df, val_df, test_df, pos_weight = load_and_split_rsna_metadata(
        cfg.labels_csv_path,
        train_ratio=cfg.train_ratio,
        val_ratio=cfg.val_ratio,
        test_ratio=cfg.test_ratio,
        seed=cfg.seed,
    )

    # 3. Create datasets and dataloaders
    train_dataset = RSNAPneumoniaDataset(
        train_df, cfg.images_dir_path, transform=get_transforms(cfg.img_size, is_train=True)
    )
    val_dataset = RSNAPneumoniaDataset(
        val_df, cfg.images_dir_path, transform=get_transforms(cfg.img_size, is_train=False)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
    )

    # 4. Model, loss, optimizer, scheduler
    model = build_resnet50_model(pretrained=True).to(device)
    pos_weight_tensor = torch.tensor([pos_weight], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    best_auroc = 0.0
    patience_counter = 0

    # 5. Training loop
    for epoch in range(1, cfg.num_epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_auroc = evaluate_epoch(model, val_loader, criterion, device)
        scheduler.step(val_auroc)

        logger.info(
            f"Epoch {epoch:02d}/{cfg.num_epochs:02d} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc*100:.2f}% | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc*100:.2f}% AUROC: {val_auroc:.4f}"
        )

        # Checkpoint best model
        if val_auroc > best_auroc:
            best_auroc = val_auroc
            patience_counter = 0
            checkpoint_payload = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_auroc": val_auroc,
                "val_acc": val_acc,
                "val_loss": val_loss,
                "pos_weight": pos_weight,
                "config": {
                    "architecture": "ResNet50",
                    "img_size": cfg.img_size,
                    "num_epochs": cfg.num_epochs,
                    "learning_rate": cfg.learning_rate,
                },
            }
            torch.save(checkpoint_payload, cfg.checkpoint_path)
            logger.info(f"==> Saved new best model checkpoint to '{cfg.checkpoint_path}' (AUROC: {val_auroc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= cfg.early_stopping_patience:
                logger.info(f"Early stopping triggered after {epoch} epochs.")
                break

    logger.info(f"Training completed. Best Validation AUROC: {best_auroc:.4f}")
    return {
        "status": "completed",
        "best_auroc": best_auroc,
        "checkpoint_path": str(cfg.checkpoint_path),
    }


def main():
    parser = argparse.ArgumentParser(description="MediVision AI - ResNet50 Chest X-ray Classifier Training")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", "--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--device", type=str, default="", help="Device override ('cuda' or 'cpu')")
    parser.add_argument("--data-dir", "--data_dir", type=str, default="datasets/rsna_pneumonia", help="Path to RSNA dataset directory")
    parser.add_argument("--image-size", "--image_size", type=int, default=224, help="Image resolution")
    args = parser.parse_args()

    cfg = CNNConfig(
        dataset_dir=Path(args.data_dir),
        img_size=(args.image_size, args.image_size),
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device_override=args.device,
    )
    run_training(cfg)


if __name__ == "__main__":
    main()
