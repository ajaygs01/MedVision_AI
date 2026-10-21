import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

import numpy as np

try:
    import torch
    from torch.utils.data import DataLoader
except ImportError:
    torch = None
    DataLoader = None

from ml.chest_xray.cnn.config import CNNConfig, default_config
from ml.chest_xray.cnn.dataset import (
    RSNAPneumoniaDataset,
    get_transforms,
    load_and_split_rsna_metadata,
    verify_dataset_presence,
)
from ml.chest_xray.cnn.train import build_resnet50_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("medivision.ml.evaluate")


def compute_metrics(y_true: np.ndarray, y_probs: np.ndarray, threshold: float = 0.5) -> Dict:
    """
    Computes diagnostic classification metrics:
    Accuracy, Precision, Sensitivity (Recall), Specificity, F1-Score, ROC-AUC, and Confusion Matrix.
    """
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    y_pred = (y_probs >= threshold).astype(int)
    y_true = y_true.astype(int)

    acc = float(accuracy_score(y_true, y_pred))
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    sensitivity = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    # Confusion matrix: tn, fp, fn, tp
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    try:
        auroc = float(roc_auc_score(y_true, y_probs))
    except Exception:
        auroc = None

    return {
        "accuracy": round(acc, 4),
        "precision": round(precision, 4),
        "sensitivity": round(sensitivity, 4),
        "specificity": round(specificity, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(auroc, 4) if auroc is not None else None,
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },
        "threshold": threshold,
        "total_evaluated_samples": len(y_true),
    }


def run_evaluation(
    config: Optional[CNNConfig] = None,
    checkpoint_path: Optional[Path] = None,
) -> Dict:
    """
    Runs model evaluation on the held-out test split using the trained checkpoint.
    """
    cfg = config or default_config
    ckpt = checkpoint_path or cfg.checkpoint_path

    # Check 1: Model Checkpoint must exist
    if not ckpt.exists():
        msg = f"Model checkpoint not found at '{ckpt}'. Model must be trained before evaluation."
        logger.error(f"[NOT TRAINED] {msg}")
        return {
            "status": "not_trained",
            "message": msg,
            "checkpoint_path": str(ckpt),
        }

    # Check 2: Dataset must exist
    is_present, status_msg = verify_dataset_presence(cfg)
    if not is_present:
        logger.error(f"[DATASET MISSING] {status_msg}")
        return {
            "status": "dataset_missing",
            "message": status_msg,
        }

    device = cfg.get_device()
    logger.info(f"Loading checkpoint from '{ckpt}' for test evaluation on device: {device}")

    # Load test split
    _, _, test_df, _ = load_and_split_rsna_metadata(
        cfg.labels_csv_path,
        train_ratio=cfg.train_ratio,
        val_ratio=cfg.val_ratio,
        test_ratio=cfg.test_ratio,
        seed=cfg.seed,
    )

    test_dataset = RSNAPneumoniaDataset(
        test_df, cfg.images_dir_path, transform=get_transforms(cfg.img_size, is_train=False)
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
    )

    # Reconstruct model and load weights
    checkpoint_data = torch.load(ckpt, map_location=device)
    if isinstance(checkpoint_data, dict) and "model_state_dict" in checkpoint_data:
        state_dict = checkpoint_data["model_state_dict"]
    else:
        state_dict = checkpoint_data

    has_fc_weight = "fc.weight" in state_dict
    model = build_resnet50_model(pretrained=False, use_dropout=not has_fc_weight)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    all_targets = []
    all_probs = []

    with torch.no_grad():
        for images, targets, _ in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.sigmoid(outputs)
            all_targets.extend(targets.cpu().numpy().flatten())
            all_probs.extend(probs.cpu().numpy().flatten())

    metrics = compute_metrics(np.array(all_targets), np.array(all_probs), threshold=0.5)

    evaluation_report = {
        "status": "success",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_name": "Chest X-ray ResNet50 Classifier",
        "dataset": "RSNA Pneumonia Detection Challenge",
        "test_samples": len(test_df),
        "metrics": metrics,
        "checkpoint_epoch": checkpoint_data.get("epoch") if isinstance(checkpoint_data, dict) else None,
        "disclaimer": "Academic research prototype. Results are not clinically validated.",
    }

    # Save to metrics.json
    cfg.ensure_artifact_dir()
    with open(cfg.metrics_path, "w") as f:
        json.dump(evaluation_report, f, indent=2)

    logger.info("=" * 60)
    logger.info("TEST SET EVALUATION RESULTS")
    logger.info("=" * 60)
    logger.info(f"Accuracy:    {metrics['accuracy']*100:.2f}%")
    logger.info(f"Sensitivity: {metrics['sensitivity']*100:.2f}% (Recall)")
    logger.info(f"Specificity: {metrics['specificity']*100:.2f}%")
    logger.info(f"Precision:   {metrics['precision']*100:.2f}%")
    logger.info(f"F1-Score:    {metrics['f1_score']:.4f}")
    if metrics["roc_auc"]:
        logger.info(f"ROC-AUC:     {metrics['roc_auc']:.4f}")
    logger.info(f"Confusion Matrix: {metrics['confusion_matrix']}")
    logger.info(f"Exported metrics to '{cfg.metrics_path}'.")
    logger.info("=" * 60)

    return evaluation_report


def main():
    parser = argparse.ArgumentParser(description="MediVision AI - ResNet50 Chest X-ray Evaluation")
    parser.add_argument("--checkpoint", type=str, default="", help="Path to checkpoint .pth")
    parser.add_argument("--batch-size", "--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--data-dir", "--data_dir", type=str, default="datasets/rsna_pneumonia", help="Path to RSNA dataset directory")
    args = parser.parse_args()

    cfg = CNNConfig(
        dataset_dir=Path(args.data_dir),
        batch_size=args.batch_size,
    )
    ckpt = Path(args.checkpoint) if args.checkpoint else cfg.checkpoint_path
    run_evaluation(cfg, ckpt)


if __name__ == "__main__":
    main()
