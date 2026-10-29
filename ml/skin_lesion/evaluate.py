import json
from pathlib import Path
from typing import Dict, Any, List
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from ml.skin_lesion.config import HAM10000_CLASSES, CLASS_NAMES


def evaluate_model(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: str = "cpu",
) -> Dict[str, Any]:
    """
    Evaluates multi-class classifier on validation set and computes genuine metrics.
    """
    model.eval()
    total_loss = 0.0
    all_preds: List[int] = []
    all_targets: List[int] = []

    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * images.size(0)

            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(targets.cpu().numpy())

    num_samples = len(all_targets)
    avg_loss = total_loss / max(num_samples, 1)

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)

    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    rec = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(HAM10000_CLASSES)))).tolist()

    # Per-class breakdown
    per_class = {}
    for idx, cls_code in enumerate(HAM10000_CLASSES):
        cls_mask = y_true == idx
        cls_total = int(np.sum(cls_mask))
        if cls_total > 0:
            cls_acc = float(np.sum((y_pred == idx) & cls_mask) / cls_total)
        else:
            cls_acc = 0.0
        per_class[cls_code] = {
            "name": CLASS_NAMES.get(cls_code, cls_code),
            "samples": cls_total,
            "accuracy": cls_acc,
        }

    return {
        "val_loss": float(avg_loss),
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "precision_macro": prec,
        "recall_macro": rec,
        "f1_macro": f1,
        "total_validation_samples": num_samples,
        "confusion_matrix": cm,
        "per_class": per_class,
        "classes": HAM10000_CLASSES,
    }


def save_metrics(metrics: Dict[str, Any], filepath: Path) -> None:
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
