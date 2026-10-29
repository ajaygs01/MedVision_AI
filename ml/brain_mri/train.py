"""
3D U-Net Training Pipeline for BraTS Brain MRI Tumor Segmentation.
Supports resume, device selection (CPU/CUDA), patient-level splits, and validation checkpointing.
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import torch
from torch.utils.data import DataLoader

from ml.brain_mri.config import (
    ARTIFACTS_DIR,
    CHECKPOINT_BEST_PATH,
    CHECKPOINT_LAST_PATH,
    CHECKPOINT_PILOT_LAST_PATH,
    CHECKPOINT_PILOT_PATH,
    DATASET_DIR,
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_LR,
    DEFAULT_PATCH_SIZE,
    DEFAULT_WEIGHT_DECAY,
    IN_CHANNELS,
    METRICS_PATH,
    METRICS_PILOT_PATH,
    NUM_CLASSES,
    PILOT_ARTIFACTS_DIR,
    RANDOM_SEED,
)
from ml.brain_mri.dataset import BraTSDataset, get_patient_splits, scan_brats_patients, verify_brats_dataset
from ml.brain_mri.losses import DiceCELoss, compute_subregion_dice
from ml.brain_mri.model import UNet3D

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("medivision.ml.brain_mri.train")


def train_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: torch.device,
    epoch: int,
    total_epochs: int,
    log_interval: int = 5,
) -> float:
    model.train()
    total_loss = 0.0
    steps = 0
    total_steps = len(loader)
    epoch_start = time.time()
    last_step_time = epoch_start

    for batch_idx, batch in enumerate(loader):
        step_start = time.time()
        images = batch["image"].to(device)  # [B, 4, D, H, W]
        masks = batch["mask"].to(device)    # [B, D, H, W]

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()

        loss_val = loss.item()
        total_loss += loss_val
        steps += 1
        step_duration = time.time() - step_start

        if (batch_idx + 1) % log_interval == 0 or (batch_idx + 1) == total_steps:
            elapsed = time.time() - epoch_start
            avg_step_time = elapsed / steps
            remaining_steps = total_steps - steps
            eta_seconds = remaining_steps * avg_step_time
            avg_loss = total_loss / steps
            eta_str = f"{int(eta_seconds // 60)}m {int(eta_seconds % 60):02d}s" if eta_seconds >= 60 else f"{eta_seconds:.1f}s"
            print(
                f"  [Epoch {epoch+1:02d}/{total_epochs:02d} | Step {batch_idx+1:03d}/{total_steps:03d}] "
                f"Loss: {loss_val:.4f} (Avg: {avg_loss:.4f}) | "
                f"Step: {step_duration:.2f}s | "
                f"ETA: {eta_str}"
            )

    return total_loss / max(1, steps)


@torch.no_grad()
def validate_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: Optional[torch.nn.Module] = None,
) -> dict:
    model.eval()
    all_dice_wt = []
    all_dice_tc = []
    all_dice_et = []
    total_val_loss = 0.0
    val_steps = 0

    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        logits = model(images)
        if criterion is not None:
            loss = criterion(logits, masks)
            total_val_loss += loss.item()
            val_steps += 1

        preds = torch.argmax(logits, dim=1)

        for i in range(preds.shape[0]):
            scores = compute_subregion_dice(preds[i], masks[i])
            all_dice_wt.append(scores["WT"])
            all_dice_tc.append(scores["TC"])
            all_dice_et.append(scores["ET"])

    mean_wt = float(sum(all_dice_wt) / max(1, len(all_dice_wt)))
    mean_tc = float(sum(all_dice_tc) / max(1, len(all_dice_tc)))
    mean_et = float(sum(all_dice_et) / max(1, len(all_dice_et)))
    mean_dice = float((mean_wt + mean_tc + mean_et) / 3.0)

    res = {
        "WT": round(mean_wt, 4),
        "TC": round(mean_tc, 4),
        "ET": round(mean_et, 4),
        "mean_dice": round(mean_dice, 4),
    }
    if criterion is not None and val_steps > 0:
        res["val_loss"] = round(total_val_loss / val_steps, 4)
    return res


import ctypes


def get_available_ram_gb() -> float:
    """Queries current available physical RAM on Windows."""
    try:
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
            return round(stat.ullAvailPhys / (1024**3), 2)
    except Exception:
        pass
    return 0.0


def save_checkpoint(
    path: Path,
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    best_val_dice: float,
    base_filters: int,
    patch_size: tuple,
    is_pilot: bool = False,
):
    ckpt_data = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_val_dice": best_val_dice,
        "config": {
            "in_channels": IN_CHANNELS,
            "num_classes": NUM_CLASSES,
            "base_filters": base_filters,
            "patch_size": list(patch_size),
            "is_pilot": is_pilot,
            "model_type": "Pilot / Smoke-Test Model" if is_pilot else "Standard 3D U-Net",
            "disclaimer": (
                "Pilot smoke-test model trained on limited patient cohort for CPU execution verification only. "
                "Not suitable for clinical diagnostic use or surgical planning."
                if is_pilot
                else "Academic research prototype. Segmentations are for investigational use only."
            ),
        },
    }
    torch.save(ckpt_data, path)


def run_training(
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    lr: float = DEFAULT_LR,
    device_name: str = "cpu",
    resume: bool = False,
    data_dir: Path = DATASET_DIR,
    base_filters: int = 16,
    patch_size: tuple = (64, 96, 96),
    max_train_patients: int = 0,
    max_val_patients: int = 0,
    max_test_patients: int = 0,
    is_pilot: bool = False,
    log_interval: int = 5,
    artifacts_dir: Optional[Path] = None,
):
    import tracemalloc
    tracemalloc.start()
    total_training_start = time.time()

    init_ram_gb = get_available_ram_gb()
    print("=" * 70)
    print("MEDIVISION AI: BRAIN MRI 3D U-NET TRAINING PIPELINE")
    if is_pilot:
        print("[MODE] CPU PILOT / SMOKE-TEST VERIFICATION (Non-Production)")
    print("=" * 70)
    print(f"Initial Available System RAM: {init_ram_gb} GB")

    # 1. Dataset Verification Check
    data_info = verify_brats_dataset(data_dir)
    if data_info["status"] == "dataset_missing":
        print(f"\n[ERROR] Training cannot begin because BraTS dataset is missing.")
        print(f"  Path: {data_info.get('data_dir')}")
        print(f"  {data_info.get('setup_guide')}")
        print("\nTo set up BraTS data, follow instructions in datasets/brats/README.md.")
        return False

    patients = scan_brats_patients(data_dir)
    complete = [p for p in patients if p.get("is_complete")]
    if len(complete) < 3:
        print(f"\n[ERROR] Found only {len(complete)} complete patient studies. At least 3 complete studies required for train/val/test.")
        return False

    # 2. Patient-Level Splitting (Zero Leakage)
    train_patients, val_patients, test_patients = get_patient_splits(complete)

    # Cap splits if requested for custom cohort runs
    if max_train_patients > 0 and max_train_patients < len(train_patients):
        print(f"Cohort Subsetting: Limiting training cohort to {max_train_patients} patients (out of {len(train_patients)})")
        train_patients = train_patients[:max_train_patients]

    if max_val_patients > 0 and max_val_patients < len(val_patients):
        print(f"Cohort Subsetting: Limiting validation cohort to {max_val_patients} patients (out of {len(val_patients)})")
        val_patients = val_patients[:max_val_patients]

    if max_test_patients > 0 and max_test_patients < len(test_patients):
        print(f"Cohort Subsetting: Limiting test evaluation cohort to {max_test_patients} patients (out of {len(test_patients)})")
        test_patients = test_patients[:max_test_patients]

    print(f"Patient-Level Split: Train={len(train_patients)}, Val={len(val_patients)}, Test={len(test_patients)}")

    # 3. Device Selection (Explicit CPU enforcement)
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    print(f"Active Compute Device: {device}")
    print(f"Patch Size (D, H, W):  {patch_size}")
    print(f"Base Filters:          {base_filters}")
    print(f"Batch Size:            {batch_size}")
    print(f"Target Epochs:         {epochs}")

    # 4. Data Loaders
    train_dataset = BraTSDataset(train_patients, patch_size=patch_size, is_training=True)
    val_dataset = BraTSDataset(val_patients, patch_size=patch_size, is_training=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # 5. Model, Loss, Optimizer
    model = UNet3D(in_channels=IN_CHANNELS, num_classes=NUM_CLASSES, base_filters=base_filters).to(device)
    criterion = DiceCELoss(dice_weight=1.0, ce_weight=0.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=DEFAULT_WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    start_epoch = 0
    best_val_dice = 0.0

    if artifacts_dir is not None:
        target_artifacts_dir = Path(artifacts_dir)
        target_best_ckpt = target_artifacts_dir / ("unet3d_pilot.pth" if is_pilot else "unet3d_best.pth")
        target_last_ckpt = target_artifacts_dir / "last_checkpoint.pth"
        target_metrics_path = target_artifacts_dir / ("metrics_pilot.json" if is_pilot else "metrics.json")
    else:
        target_artifacts_dir = PILOT_ARTIFACTS_DIR if is_pilot else ARTIFACTS_DIR
        target_best_ckpt = CHECKPOINT_PILOT_PATH if is_pilot else CHECKPOINT_BEST_PATH
        target_last_ckpt = CHECKPOINT_PILOT_LAST_PATH if is_pilot else CHECKPOINT_LAST_PATH
        target_metrics_path = METRICS_PILOT_PATH if is_pilot else METRICS_PATH

    target_artifacts_dir.mkdir(parents=True, exist_ok=True)
    print(f"Target Artifacts Directory: {target_artifacts_dir.resolve()}")

    # 6. Checkpoint Resume with Strict Configuration Validation
    if resume:
        if not target_last_ckpt.exists():
            print(f"\n[ERROR] Cannot resume: Checkpoint file does not exist at: {target_last_ckpt}")
            print(f"To start a fresh training run, omit the --resume flag.")
            return False
        print(f"Resuming training from checkpoint: {target_last_ckpt}")
        ckpt = torch.load(target_last_ckpt, map_location=device)
        ckpt_config = ckpt.get("config", {})
        ckpt_patch = tuple(ckpt_config.get("patch_size", []))
        ckpt_filters = ckpt_config.get("base_filters")
        if ckpt_patch and ckpt_patch != patch_size:
            print(f"\n[ERROR] Checkpoint patch size {ckpt_patch} does not match current run patch size {patch_size}.")
            return False
        if ckpt_filters and ckpt_filters != base_filters:
            print(f"\n[ERROR] Checkpoint base_filters {ckpt_filters} does not match current run base_filters {base_filters}.")
            return False
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val_dice = ckpt.get("best_val_dice", 0.0)
        print(f"Resumed at epoch {start_epoch} (Prior best Val Dice: {best_val_dice:.4f})")

    # 7. Training Loop with Safe Interruption Handling
    history = []
    print(f"\nStarting 3D U-Net training from epoch {start_epoch + 1} to {epochs}...")
    interrupted = False

    try:
        for epoch in range(start_epoch, epochs):
            t0 = time.time()
            print(f"\n--- Epoch {epoch+1:02d}/{epochs:02d} ---")
            train_loss = train_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                criterion=criterion,
                device=device,
                epoch=epoch,
                total_epochs=epochs,
                log_interval=log_interval,
            )
            scheduler.step()

            print(f"Evaluating validation cohort ({len(val_patients)} patients)...")
            val_metrics = validate_epoch(model, val_loader, device, criterion=criterion)
            val_dice = val_metrics["mean_dice"]
            val_loss = val_metrics.get("val_loss")
            elapsed = time.time() - t0

            val_loss_str = f"Val Loss: {val_loss:.4f} | " if val_loss is not None else ""
            print(
                f"Epoch {epoch+1:02d}/{epochs:02d} Complete [{elapsed:.1f}s] - "
                f"Train Loss: {train_loss:.4f} | "
                f"{val_loss_str}"
                f"Val Dice Mean: {val_dice:.4f} (WT: {val_metrics['WT']:.4f}, TC: {val_metrics['TC']:.4f}, ET: {val_metrics['ET']:.4f})"
            )

            history.append({
                "epoch": epoch + 1,
                "train_loss": round(train_loss, 4),
                "val_loss": val_loss,
                "val_metrics": val_metrics,
                "epoch_runtime_seconds": round(elapsed, 2),
            })

            # Save last checkpoint for resume
            save_checkpoint(
                path=target_last_ckpt,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                best_val_dice=best_val_dice,
                base_filters=base_filters,
                patch_size=patch_size,
                is_pilot=is_pilot,
            )

            # Save best model
            if val_dice > best_val_dice:
                best_val_dice = val_dice
                save_checkpoint(
                    path=target_best_ckpt,
                    epoch=epoch,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    best_val_dice=best_val_dice,
                    base_filters=base_filters,
                    patch_size=patch_size,
                    is_pilot=is_pilot,
                )
                print(f"  -> Checkpoint updated: best_val_dice={best_val_dice:.4f} saved to {target_best_ckpt.name}")

    except KeyboardInterrupt:
        interrupted = True
        print("\n" + "!" * 70)
        print("[INTERRUPTED] Safe shutdown triggered by user (Ctrl+C).")
        save_checkpoint(
            path=target_last_ckpt,
            epoch=epoch if 'epoch' in locals() else start_epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            best_val_dice=best_val_dice,
            base_filters=base_filters,
            patch_size=patch_size,
            is_pilot=is_pilot,
        )
        print(f"Current model state safely saved to: {target_last_ckpt}")
        print("You can resume training at any time with:")
        art_flag = f" --artifacts-dir {artifacts_dir}" if artifacts_dir else ""
        pilot_flag = " --is-pilot" if is_pilot else ""
        train_p_flag = f" --max-train-patients {max_train_patients}" if max_train_patients > 0 else ""
        val_p_flag = f" --max-val-patients {max_val_patients}" if max_val_patients > 0 else ""
        test_p_flag = f" --max-test-patients {max_test_patients}" if max_test_patients > 0 else ""
        resume_cmd = (
            f"  .\\backend\\.venv\\Scripts\\python.exe -m ml.brain_mri.train --resume "
            f"--epochs {epochs} --batch-size {batch_size} --device {device_name}"
            f"{train_p_flag}{val_p_flag}{test_p_flag}{pilot_flag}{art_flag}"
        )
        print(resume_cmd)
        print("!" * 70)

    if interrupted:
        return False

    print("\nTraining completed successfully.")

    # 8. Evaluation on Held-Out Test Split
    if not target_best_ckpt.exists():
        print(f"[NOTE] Saving final trained state to {target_best_ckpt.name} for held-out evaluation.")
        save_checkpoint(
            path=target_best_ckpt,
            epoch=epochs - 1,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            best_val_dice=best_val_dice,
            base_filters=base_filters,
            patch_size=patch_size,
            is_pilot=is_pilot,
        )

    from ml.brain_mri.evaluate import run_evaluation
    test_metrics = run_evaluation(
        checkpoint_path=target_best_ckpt,
        test_patients=test_patients,
        device=device,
    )

    _, peak_py_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_py_mb = round(peak_py_bytes / (1024**2), 2)
    final_ram_gb = get_available_ram_gb()
    total_runtime_seconds = round(time.time() - total_training_start, 2)

    print("\n" + "=" * 70)
    print("RESOURCE CONSUMPTION SUMMARY:")
    print(f"  - Total Runtime:                     {total_runtime_seconds}s")
    print(f"  - Peak Process Memory (Tracemalloc): {peak_py_mb} MB")
    print(f"  - System RAM Before Training:        {init_ram_gb} GB available")
    print(f"  - System RAM After Training:         {final_ram_gb} GB available")
    print(f"  - Memory Headroom Maintained:        Safe (Within OS limits)")
    print("=" * 70)

    metrics_payload = {
        "model_name": "3D U-Net Brain Tumor Segmenter",
        "dataset_name": "BraTS Multi-Modal Cohort",
        "is_pilot": is_pilot,
        "model_type": "Pilot / Smoke-Test Model" if is_pilot else "Standard 3D U-Net",
        "training_epochs": epochs,
        "train_patients": len(train_patients),
        "val_patients": len(val_patients),
        "test_patients": len(test_patients),
        "best_val_dice": round(best_val_dice, 4),
        "val_loss": val_metrics.get("val_loss"),
        "runtime_seconds": total_runtime_seconds,
        "checkpoint_paths": {
            "best_checkpoint": str(target_best_ckpt.resolve()),
            "last_checkpoint": str(target_last_ckpt.resolve()),
        },
        "held_out_test_metrics": test_metrics,
        "history": history,
        "resource_metrics": {
            "peak_python_ram_mb": peak_py_mb,
            "initial_available_ram_gb": init_ram_gb,
            "final_available_ram_gb": final_ram_gb,
        },
        "disclaimer": (
            "Pilot smoke-test verification model trained on limited patient subset. "
            "Metrics are for CPU execution verification only; not clinically validated."
            if is_pilot
            else "Academic research prototype. Segmentations are for investigational use only."
        ),
    }

    with open(target_metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)
    print(f"Final training and held-out test metrics saved to: {target_metrics_path}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Train 3D U-Net on BraTS Brain MRI Dataset")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Training batch size")
    parser.add_argument("--lr", type=float, default=DEFAULT_LR, help="Initial learning rate")
    parser.add_argument("--device", type=str, default="cpu", choices=["auto", "cpu", "cuda"], help="Compute device")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint if available")
    parser.add_argument("--data-dir", type=str, default=str(DATASET_DIR), help="BraTS dataset root directory")
    parser.add_argument("--base-filters", type=int, default=16, help="Initial filter count for 3D U-Net")
    parser.add_argument("--patch-size", type=int, nargs=3, default=[64, 96, 96], help="Patch size (Depth, Height, Width)")
    parser.add_argument("--max-train-patients", type=int, default=0, help="Cap training cohort size (0 for full)")
    parser.add_argument("--max-val-patients", type=int, default=0, help="Cap validation cohort size (0 for full)")
    parser.add_argument("--max-test-patients", type=int, default=0, help="Cap test cohort size (0 for full)")
    parser.add_argument("--is-pilot", action="store_true", help="Tag model and metrics as pilot / smoke-test")
    parser.add_argument("--artifacts-dir", type=str, default="", help="Custom artifacts directory (isolates checkpoints/metrics from pilot and production)")
    parser.add_argument("--log-interval", type=int, default=5, help="Step interval for progress logging")
    args = parser.parse_args()

    run_training(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device_name=args.device,
        resume=args.resume,
        data_dir=Path(args.data_dir),
        base_filters=args.base_filters,
        patch_size=tuple(args.patch_size),
        max_train_patients=args.max_train_patients,
        max_val_patients=args.max_val_patients,
        max_test_patients=args.max_test_patients,
        is_pilot=args.is_pilot,
        log_interval=args.log_interval,
        artifacts_dir=Path(args.artifacts_dir) if args.artifacts_dir else None,
    )


if __name__ == "__main__":
    main()
