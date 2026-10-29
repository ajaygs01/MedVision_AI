"""
Evaluation script for 3D U-Net on held-out BraTS test cohort.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import torch
from torch.utils.data import DataLoader

from ml.brain_mri.config import (
    CHECKPOINT_BEST_PATH,
    DATASET_DIR,
    DEFAULT_PATCH_SIZE,
    IN_CHANNELS,
    METRICS_PATH,
    NUM_CLASSES,
)
from ml.brain_mri.dataset import BraTSDataset, get_patient_splits, scan_brats_patients, verify_brats_dataset
from ml.brain_mri.losses import compute_subregion_dice
from ml.brain_mri.model import UNet3D


def run_evaluation(
    checkpoint_path: Path = CHECKPOINT_BEST_PATH,
    test_patients: Optional[List[Dict[str, Any]]] = None,
    data_dir: Path = DATASET_DIR,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Evaluates trained 3D U-Net checkpoint on held-out test split.
    Computes per-subregion Dice Similarity Coefficients (WT, TC, ET) and Mean Dice.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not checkpoint_path.exists():
        return {
            "status": "not_trained",
            "message": f"No model checkpoint found at {checkpoint_path.resolve()}.",
            "metrics": {},
        }

    if test_patients is None:
        data_info = verify_brats_dataset(data_dir)
        if data_info["status"] == "dataset_missing":
            return {
                "status": "dataset_missing",
                "message": data_info["message"],
                "metrics": {},
            }
        patients = scan_brats_patients(data_dir)
        complete = [p for p in patients if p.get("is_complete")]
        if not complete:
            return {
                "status": "no_test_data",
                "message": "No complete patient studies available in dataset directory.",
                "metrics": {},
            }
        _, _, test_patients = get_patient_splits(complete)

    if not test_patients:
        return {
            "status": "empty_test_split",
            "message": "Held-out test split contains 0 studies.",
            "metrics": {},
        }

    # Load model from checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device)
    base_filters = ckpt.get("config", {}).get("base_filters", 16)
    patch_size = tuple(ckpt.get("config", {}).get("patch_size", DEFAULT_PATCH_SIZE))
    model = UNet3D(in_channels=IN_CHANNELS, num_classes=NUM_CLASSES, base_filters=base_filters).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    test_dataset = BraTSDataset(test_patients, patch_size=patch_size, is_training=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    all_wt, all_tc, all_et = [], [], []

    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)

            logits = model(images)
            preds = torch.argmax(logits, dim=1)

            for i in range(preds.shape[0]):
                scores = compute_subregion_dice(preds[i], masks[i])
                all_wt.append(scores["WT"])
                all_tc.append(scores["TC"])
                all_et.append(scores["ET"])

    mean_wt = float(sum(all_wt) / max(1, len(all_wt)))
    mean_tc = float(sum(all_tc) / max(1, len(all_tc)))
    mean_et = float(sum(all_et) / max(1, len(all_et)))
    mean_dice = float((mean_wt + mean_tc + mean_et) / 3.0)

    results = {
        "status": "success",
        "evaluated_studies": len(test_patients),
        "metrics": {
            "whole_tumor_dice": round(mean_wt, 4),
            "tumor_core_dice": round(mean_tc, 4),
            "enhancing_tumor_dice": round(mean_et, 4),
            "mean_dice": round(mean_dice, 4),
        },
    }
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate 3D U-Net on held-out BraTS test cohort")
    parser.add_argument("--checkpoint", type=str, default=str(CHECKPOINT_BEST_PATH), help="Path to checkpoint .pth")
    parser.add_argument("--data-dir", type=str, default=str(DATASET_DIR), help="Path to BraTS dataset directory")
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    print("=" * 70)
    print("MEDIVISION AI: BRAIN MRI 3D U-NET HELD-OUT EVALUATION")
    print("=" * 70)
    print(f"Checkpoint: {ckpt_path.resolve()}")
    print(f"Data Dir:   {Path(args.data_dir).resolve()}")

    res = run_evaluation(checkpoint_path=ckpt_path, data_dir=Path(args.data_dir))
    status = res.get("status")
    print(f"\nEvaluation Status: {status.upper()}")
    print(f"Message:           {res.get('message', 'Evaluation finished.')}")

    if status == "success":
        m = res.get("metrics", {})
        print(f"\nHeld-Out Test Split Benchmark Metrics (N={res.get('evaluated_studies')}):")
        print(f"  - Whole Tumor (WT) Dice:     {m.get('whole_tumor_dice'):.4f}")
        print(f"  - Tumor Core (TC) Dice:      {m.get('tumor_core_dice'):.4f}")
        print(f"  - Enhancing Tumor (ET) Dice: {m.get('enhancing_tumor_dice'):.4f}")
        print(f"  - Overall Mean Dice:         {m.get('mean_dice'):.4f}")
    else:
        print("\nPrerequisite Check:")
        if status == "not_trained":
            print("  Model checkpoint has not been trained yet. Run ml/brain_mri/train.py once dataset is in place.")
        elif status == "dataset_missing":
            print("  BraTS dataset is missing. See datasets/brats/README.md for setup instructions.")

    print("=" * 70)


if __name__ == "__main__":
    main()
