"""
YOLOv8 training and validation pipeline for MediVision-AI RSNA Pneumonia detection.
Trains YOLOv8 on prepared Darknet RSNA dataset, evaluates genuine validation & test metrics,
and saves validated checkpoints to backend/model_artifacts/yolo_detection/.
"""

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Union

# Ensure project root is on sys.path so direct and module executions both resolve ml.*
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from ml.yolo_detection.config import (
    ARTIFACTS_DIR,
    CHECKPOINT_PATH,
    METRICS_PATH,
    MODEL_ARCH,
    PROJECT_ROOT,
    YOLO_DATA_YAML,
    YOLO_DATASET_DIR,
)
from ml.yolo_detection.dataset import verify_yolo_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("medivision.ml.yolo_train")


def train_yolo_model(
    data_yaml: Path = YOLO_DATA_YAML,
    weights_path: Optional[Union[str, Path]] = None,
    epochs: int = 3,
    batch_size: int = 2,
    imgsz: int = 320,
    workers: int = 0,
    device: str = "cpu",
    output_dir: Path = ARTIFACTS_DIR,
    run_name: str = "rsna_cpu_exp1",
    warmup_epochs: float = 1.0,
    mosaic: float = 0.0,
    patience: int = 8,
    cls_gain: float = 1.5,
    cos_lr: bool = True,
    skip_test: bool = True,
    no_promote: bool = False,
) -> Dict[str, Any]:
    """
    Trains YOLOv8 architecture on the prepared RSNA dataset, evaluates metrics on validation/test set,
    and updates the official checkpoint at backend/model_artifacts/yolo_detection/.
    """
    from ultralytics import YOLO

    # 1. Dataset verification
    dataset_status = verify_yolo_dataset(data_yaml.parent)
    if dataset_status.get("status") != "valid":
        raise ValueError(f"Dataset verification failed: {dataset_status.get('message')}")

    # 2. Checkpoint path resolution & validation
    resolved_weights = Path(weights_path) if weights_path else (PROJECT_ROOT / "yolov8n.pt")
    if not resolved_weights.is_absolute():
        resolved_weights = (PROJECT_ROOT / resolved_weights).resolve()

    if not resolved_weights.exists():
        raise FileNotFoundError(
            f"Pretrained weights not found at {resolved_weights}. "
            "Please ensure yolov8n.pt is placed in the project root."
        )

    file_size = resolved_weights.stat().st_size
    if file_size < 5_000_000:
        raise ValueError(
            f"Pretrained weights file at {resolved_weights} appears corrupt or incomplete "
            f"({file_size} bytes, expected ~6.5 MB)."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = output_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Initializing YOLO model ({MODEL_ARCH}) from local pretrained weights: {resolved_weights}...")
    model = YOLO(str(resolved_weights))

    logger.info(
        f"Starting training on {data_yaml} for {epochs} epochs "
        f"(batch={batch_size}, imgsz={imgsz}, workers={workers}, device={device}, exp={run_name}, "
        f"patience={patience}, cls_gain={cls_gain}, cos_lr={cos_lr}, skip_test={skip_test}, no_promote={no_promote})..."
    )

    # 3. Run Training (Uses 'val' split for epoch validation & best checkpoint selection; 'test' split is held out)
    train_results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        batch=batch_size,
        imgsz=imgsz,
        workers=workers,
        device=device,
        project=str(runs_dir),
        name=run_name,
        warmup_epochs=warmup_epochs,
        mosaic=mosaic,
        patience=patience,
        cls=cls_gain,
        cos_lr=cos_lr,
        exist_ok=False,
        save=True,
        verbose=True,
    )

    # 4. Locate run directory and best.pt
    save_dir = Path(getattr(model.trainer, "save_dir", runs_dir / run_name))
    run_best_pt = save_dir / "weights" / "best.pt"
    if not run_best_pt.exists():
        run_best_pt = save_dir / "weights" / "last.pt"

    if not run_best_pt.exists():
        raise FileNotFoundError(
            f"Training completed but no weight file was found under {save_dir / 'weights'}"
        )

    # 5. Evaluate Validation Split (Used to confirm checkpoint quality)
    logger.info("Evaluating model metrics on validation split...")
    eval_model = YOLO(str(run_best_pt))
    val_metrics: Dict[str, Any] = {}
    try:
        val_results = eval_model.val(
            data=str(data_yaml),
            split="val",
            batch=batch_size,
            imgsz=imgsz,
            device=device,
            verbose=False,
        )
        val_box = getattr(val_results, "box", None)
        if val_box is not None and hasattr(val_box, "map50"):
            val_metrics = {
                "precision": round(float(val_box.mp), 4),
                "recall": round(float(val_box.mr), 4),
                "mAP50": round(float(val_box.map50), 4),
                "mAP50_95": round(float(val_box.map), 4),
            }
        else:
            logger.warning("Validation completed but metrics box was unavailable.")
            val_metrics = {"error": "validation_box_metrics_unavailable"}
    except Exception as val_exc:
        logger.error(f"Validation evaluation failed: {val_exc}")
        val_metrics = {"error": str(val_exc)}

    # 6. Evaluate Test Split (Unbiased holdout test set - evaluated only once after experiment finalized)
    test_metrics: Dict[str, Any] = {}
    if not skip_test:
        logger.info("Evaluating model metrics on holdout test split...")
        try:
            test_results = eval_model.val(
                data=str(data_yaml),
                split="test",
                batch=batch_size,
                imgsz=imgsz,
                device=device,
                verbose=False,
            )
            test_box = getattr(test_results, "box", None)
            if test_box is not None and hasattr(test_box, "map50"):
                test_metrics = {
                    "precision": round(float(test_box.mp), 4),
                    "recall": round(float(test_box.mr), 4),
                    "mAP50": round(float(test_box.map50), 4),
                    "mAP50_95": round(float(test_box.map), 4),
                }
            else:
                logger.warning("Test evaluation completed but metrics box was unavailable.")
                test_metrics = {"error": "test_box_metrics_unavailable"}
        except Exception as test_exc:
            logger.error(f"Test evaluation failed: {test_exc}")
            test_metrics = {"error": str(test_exc)}
    else:
        logger.info("Skipping holdout test split evaluation (--skip-test active; holdout set preserved).")
        test_metrics = {"status": "skipped_holdout_test_set_preserved"}

    # 6b. Extract Training Run Metrics
    train_metrics: Dict[str, Any] = {}
    if hasattr(train_results, "results_dict") and isinstance(train_results.results_dict, dict):
        train_metrics = {
            "precision": round(float(train_results.results_dict.get("metrics/precision(B)", 0.0)), 4),
            "recall": round(float(train_results.results_dict.get("metrics/recall(B)", 0.0)), 4),
            "mAP50": round(float(train_results.results_dict.get("metrics/mAP50(B)", 0.0)), 4),
            "mAP50_95": round(float(train_results.results_dict.get("metrics/mAP50-95(B)", 0.0)), 4),
            "box_loss": round(float(train_results.results_dict.get("train/box_loss", 0.0)), 4),
            "cls_loss": round(float(train_results.results_dict.get("train/cls_loss", 0.0)), 4),
            "dfl_loss": round(float(train_results.results_dict.get("train/dfl_loss", 0.0)), 4),
        }

    # 7. Verification before marking model trained
    has_val_error = "error" in val_metrics
    has_test_error = ("error" in test_metrics) if not skip_test else False
    checkpoint_valid = run_best_pt.exists() and run_best_pt.stat().st_size > 1_000_000

    if has_val_error or has_test_error or not checkpoint_valid:
        error_msg = (
            f"Evaluation verification failed (val_error={has_val_error}, "
            f"test_error={has_test_error}, valid_checkpoint={checkpoint_valid}). "
            "Canonical checkpoint was not updated; module remains not_trained."
        )
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    # 8. Save Metrics JSON & Handle Canonical Promotion
    metrics_data = {
        "model_architecture": MODEL_ARCH,
        "base_pretrained_weights": str(resolved_weights),
        "epochs": epochs,
        "batch_size": batch_size,
        "imgsz": imgsz,
        "workers": workers,
        "device": device,
        "warmup_epochs": warmup_epochs,
        "mosaic": mosaic,
        "dataset_yaml": str(data_yaml),
        "checkpoint_path": str(run_best_pt),
        "experiment_run_dir": str(save_dir),
        "no_promote": no_promote,
        "training_metrics": train_metrics,
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
    }

    # Save run-specific metrics
    run_metrics_path = save_dir / "metrics.json"
    with open(run_metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=2)
    logger.info(f"Run-specific metrics recorded in {run_metrics_path}")

    if no_promote:
        logger.info(
            f"--no-promote active: Canonical checkpoint ({CHECKPOINT_PATH}) and canonical metrics "
            f"remain untouched. Checkpoint remains safely isolated under {run_best_pt}."
        )
    else:
        with open(METRICS_PATH, "w", encoding="utf-8") as f:
            json.dump(metrics_data, f, indent=2)
        shutil.copy2(run_best_pt, CHECKPOINT_PATH)
        logger.info(f"Copied validated checkpoint to canonical location: {CHECKPOINT_PATH}")
        logger.info(f"Canonical metrics updated in {METRICS_PATH}")

    logger.info(f"Genuine metrics for {run_name}:")
    logger.info(f"  • Validation Split: {val_metrics}")
    logger.info(f"  • Test Split:       {test_metrics}")

    return metrics_data


def build_parser() -> argparse.ArgumentParser:
    """Builds and returns the CLI argument parser for YOLO model training."""
    parser = argparse.ArgumentParser(
        description="Train MediVision-AI YOLOv8 Lesion Localization Model on RSNA Pneumonia Dataset"
    )
    parser.add_argument(
        "--data",
        type=str,
        default=str(YOLO_DATA_YAML),
        help="Path to dataset YAML configuration (default: datasets/yolo_rsna/data.yaml)",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="yolov8n.pt",
        help="Path to pretrained weights checkpoint (default: yolov8n.pt)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs (default: 3)",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=2,
        help="Batch size per step (default: 2)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=320,
        help="Input image resolution in pixels (default: 320)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Dataloader worker processes, 0 for main process on CPU (default: 0)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to train on, e.g. cpu or 0 (default: cpu)",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="rsna_cpu_exp1",
        help="Experiment run name directory (default: rsna_cpu_exp1)",
    )
    parser.add_argument(
        "--warmup-epochs",
        "--warmup_epochs",
        type=float,
        default=1.0,
        dest="warmup_epochs",
        help="Number of warmup epochs (default: 1.0)",
    )
    parser.add_argument(
        "--mosaic",
        type=float,
        default=0.0,
        help="Mosaic augmentation probability, 0.0 to disable (default: 0.0)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=8,
        help="Early stopping patience in epochs (default: 8)",
    )
    parser.add_argument(
        "--cls",
        type=float,
        default=1.5,
        dest="cls_gain",
        help="Classification loss gain (default: 1.5)",
    )
    parser.add_argument(
        "--cos-lr",
        action="store_true",
        default=True,
        dest="cos_lr",
        help="Use cosine learning rate scheduler (default: True)",
    )
    parser.add_argument(
        "--skip-test",
        action="store_true",
        default=True,
        dest="skip_test",
        help="Skip holdout test set evaluation to preserve holdout isolation (default: True)",
    )
    parser.add_argument(
        "--eval-test",
        action="store_false",
        dest="skip_test",
        help="Evaluate holdout test set (default: False)",
    )
    parser.add_argument(
        "--no-promote",
        "--no_promote",
        action="store_true",
        dest="no_promote",
        help="Do not overwrite canonical yolov8_best.pt or update canonical metrics.json",
    )
    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    train_yolo_model(
        data_yaml=Path(args.data),
        weights_path=Path(args.weights),
        epochs=args.epochs,
        batch_size=args.batch,
        imgsz=args.imgsz,
        workers=args.workers,
        device=args.device,
        run_name=args.name,
        warmup_epochs=args.warmup_epochs,
        mosaic=args.mosaic,
        patience=args.patience,
        cls_gain=args.cls_gain,
        cos_lr=args.cos_lr,
        skip_test=args.skip_test,
        no_promote=args.no_promote,
    )
