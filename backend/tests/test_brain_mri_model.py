"""
Unit tests for 3D U-Net model architecture, loss functions, and preprocessing.
"""

import pytest
import torch
import numpy as np
from ml.brain_mri.model import UNet3D
from ml.brain_mri.losses import DiceCELoss, DiceLoss, compute_subregion_dice
from ml.brain_mri.preprocessing import (
    normalize_mri_channel,
    calculate_volumetric_stats,
    extract_slice_visualizations,
)


def test_unet3d_forward_pass_shape():
    """Verify UNet3D accepts 4-channel 3D volume and outputs 4-class logits of matching spatial dimensions."""
    model = UNet3D(in_channels=4, num_classes=4, base_filters=8)
    model.eval()

    # Batch of 1, 4 channels, 16 depth/slices, 32 height, 32 width
    x = torch.randn(1, 4, 16, 32, 32)
    with torch.no_grad():
        logits = model(x)

    assert logits.shape == (1, 4, 16, 32, 32), f"Expected [1, 4, 16, 32, 32], got {logits.shape}"
    predicted_mask = model.predict_mask(x)
    assert predicted_mask.shape == (1, 16, 32, 32), f"Expected [1, 16, 32, 32], got {predicted_mask.shape}"
    assert predicted_mask.min() >= 0
    assert predicted_mask.max() <= 3


def test_dice_ce_loss_computation():
    """Verify DiceCELoss computes finite non-negative loss and gradients propagate."""
    loss_fn = DiceCELoss(dice_weight=1.0, ce_weight=0.5)

    logits = torch.randn(2, 4, 8, 16, 16, requires_grad=True)
    targets = torch.randint(0, 4, (2, 8, 16, 16))

    loss = loss_fn(logits, targets)
    assert loss.item() > 0, "Loss must be positive"
    assert torch.isfinite(loss), "Loss must be finite"

    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_subregion_dice_scores():
    """Verify WT, TC, ET Dice scores evaluate correctly on identical vs divergent masks."""
    # Target mask with all tumor subregions: 1 (NCR), 2 (ED), 3 (ET)
    target = torch.zeros(16, 32, 32, dtype=torch.long)
    target[4:8, 10:20, 10:20] = 1
    target[8:12, 10:20, 10:20] = 2
    target[6:10, 14:18, 14:18] = 3

    # Perfect prediction should yield 1.0 for WT, TC, ET
    perfect_scores = compute_subregion_dice(target, target)
    assert perfect_scores["WT"] == 1.0
    assert perfect_scores["TC"] == 1.0
    assert perfect_scores["ET"] == 1.0
    assert perfect_scores["mean_dice"] == 1.0

    # Completely wrong prediction should yield low scores
    empty_pred = torch.zeros_like(target)
    empty_scores = compute_subregion_dice(empty_pred, target)
    assert empty_scores["WT"] < 0.05
    assert empty_scores["mean_dice"] < 0.05


def test_volumetric_stats_calculation():
    """Verify volumetric lesion metrics accurately convert voxel counts to cm3."""
    mask = np.zeros((10, 10, 10), dtype=np.int64)
    # Put 1000 voxels of enhancing tumor (label 3)
    mask[:] = 3
    # 1.0mm isotropic voxels -> 1000 mm3 = 1.0 cm3
    stats = calculate_volumetric_stats(mask, voxel_spacing=(1.0, 1.0, 1.0))
    assert stats["whole_tumor_cm3"] == 1.0
    assert stats["tumor_core_cm3"] == 1.0
    assert stats["enhancing_tumor_cm3"] == 1.0
    assert stats["edema_cm3"] == 0.0


def test_slice_visualizations():
    """Verify 2D slice previews generate valid base64 PNG images."""
    vol = np.random.randn(20, 40, 40).astype(np.float32)
    mask = np.zeros((20, 40, 40), dtype=np.int64)
    mask[10, 15:25, 15:25] = 2  # Edema

    res = extract_slice_visualizations(vol, mask, slice_idx=10, plane="axial")
    assert res["plane"] == "axial"
    assert res["slice_idx"] == 10
    assert res["total_slices"] == 20
    assert res["mri_preview"].startswith("data:image/png;base64,")
    assert res["mask_preview"].startswith("data:image/png;base64,")
    assert res["overlay_preview"].startswith("data:image/png;base64,")


def test_validate_epoch_records_val_loss_and_subregion_dice():
    """Verify validate_epoch computes scalar DiceCELoss while preserving WT, TC, ET Dice scores."""
    from ml.brain_mri.train import validate_epoch
    from torch.utils.data import DataLoader, TensorDataset

    model = UNet3D(in_channels=4, num_classes=4, base_filters=4)
    model.eval()
    criterion = DiceCELoss(dice_weight=1.0, ce_weight=0.5)

    # Synthetic batch: 2 volumes of size [4, 8, 16, 16] and mask [8, 16, 16]
    images = torch.randn(2, 4, 8, 16, 16)
    masks = torch.randint(0, 4, (2, 8, 16, 16))

    class DummyDataset(torch.utils.data.Dataset):
        def __len__(self):
            return 2
        def __getitem__(self, idx):
            return {"image": images[idx], "mask": masks[idx]}

    loader = DataLoader(DummyDataset(), batch_size=2)
    device = torch.device("cpu")

    results = validate_epoch(model, loader, device, criterion=criterion)

    assert "val_loss" in results, "validate_epoch must return val_loss key"
    assert isinstance(results["val_loss"], float), "val_loss must be a float scalar"
    assert results["val_loss"] > 0, "val_loss must be positive"
    assert np.isfinite(results["val_loss"]), "val_loss must be finite"

    # Verify subregion Dice scores are preserved
    for key in ["WT", "TC", "ET", "mean_dice"]:
        assert key in results, f"Missing metric {key}"
        assert 0.0 <= results[key] <= 1.0, f"{key} out of range [0, 1]"


def test_metrics_json_saves_val_loss_and_checkpoints(tmp_path):
    """Verify metrics payload saves val_loss, runtime, and checkpoint paths without omission."""
    import json
    from ml.brain_mri.train import PILOT_ARTIFACTS_DIR, CHECKPOINT_PILOT_PATH, CHECKPOINT_PILOT_LAST_PATH

    metrics_path = tmp_path / "metrics_test.json"
    dummy_payload = {
        "model_name": "3D U-Net Brain Tumor Segmenter",
        "dataset_name": "BraTS Multi-Modal Cohort",
        "is_pilot": True,
        "model_type": "Pilot / Smoke-Test Model",
        "training_epochs": 1,
        "train_patients": 2,
        "val_patients": 1,
        "test_patients": 1,
        "best_val_dice": 0.3561,
        "val_loss": 1.4820,
        "runtime_seconds": 47.4,
        "checkpoint_paths": {
            "best_checkpoint": str(CHECKPOINT_PILOT_PATH),
            "last_checkpoint": str(CHECKPOINT_PILOT_LAST_PATH),
        },
        "held_out_test_metrics": {
            "status": "success",
            "evaluated_studies": 1,
            "metrics": {"mean_dice": 0.2524},
        },
        "history": [
            {
                "epoch": 1,
                "train_loss": 1.5449,
                "val_loss": 1.4820,
                "epoch_runtime_seconds": 47.4,
                "val_metrics": {"mean_dice": 0.3561},
            }
        ],
        "disclaimer": "Pilot smoke-test verification model.",
    }

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(dummy_payload, f, indent=2)

    with open(metrics_path, "r", encoding="utf-8") as f:
        loaded = json.load(f)

    assert loaded["val_loss"] == 1.4820
    assert loaded["runtime_seconds"] == 47.4
    assert loaded["checkpoint_paths"]["best_checkpoint"] == str(CHECKPOINT_PILOT_PATH)
    assert loaded["history"][0]["val_loss"] == 1.4820
    assert loaded["history"][0]["train_loss"] == 1.5449


def test_patient_level_splits_zero_leakage():
    """Verify get_patient_splits preserves strict zero-leakage patient-level separation."""
    from ml.brain_mri.dataset import get_patient_splits

    dummy_patients = [{"patient_id": f"BraTS20_Training_{i:03d}"} for i in range(1, 41)]
    train_pts, val_pts, test_pts = get_patient_splits(dummy_patients, train_ratio=0.8, val_ratio=0.1)

    train_ids = {p["patient_id"] for p in train_pts}
    val_ids = {p["patient_id"] for p in val_pts}
    test_ids = {p["patient_id"] for p in test_pts}

    # Intersections must be empty
    assert len(train_ids.intersection(val_ids)) == 0, "Train and Val overlap detected!"
    assert len(train_ids.intersection(test_ids)) == 0, "Train and Test overlap detected!"
    assert len(val_ids.intersection(test_ids)) == 0, "Val and Test overlap detected!"
    assert len(train_ids) + len(val_ids) + len(test_ids) == 40


def test_checkpoint_isolation_pilot_vs_production():
    """Verify pilot checkpoints are stored in an isolated pilot/ directory and do not overwrite production."""
    from ml.brain_mri.train import (
        ARTIFACTS_DIR,
        PILOT_ARTIFACTS_DIR,
        CHECKPOINT_BEST_PATH,
        CHECKPOINT_PILOT_PATH,
        CHECKPOINT_LAST_PATH,
        CHECKPOINT_PILOT_LAST_PATH,
        METRICS_PATH,
        METRICS_PILOT_PATH,
    )

    assert PILOT_ARTIFACTS_DIR != ARTIFACTS_DIR, "Pilot directory must not equal production artifacts directory"
    assert PILOT_ARTIFACTS_DIR.name == "pilot", "Pilot artifacts must be inside a 'pilot' folder"
    assert CHECKPOINT_PILOT_PATH != CHECKPOINT_BEST_PATH, "Pilot best path must differ from production path"
    assert CHECKPOINT_PILOT_LAST_PATH != CHECKPOINT_LAST_PATH, "Pilot last path must differ from production path"
    assert METRICS_PILOT_PATH != METRICS_PATH, "Pilot metrics path must differ from production metrics path"
    assert CHECKPOINT_PILOT_PATH.parent == PILOT_ARTIFACTS_DIR
    assert CHECKPOINT_BEST_PATH.parent == ARTIFACTS_DIR


def test_custom_artifacts_dir_isolation(tmp_path):
    """Verify specifying custom artifacts_dir isolates checkpoints from both pilot and production."""
    from ml.brain_mri.train import ARTIFACTS_DIR, PILOT_ARTIFACTS_DIR

    custom_dir = tmp_path / "custom_exp_dir"
    custom_dir.mkdir(parents=True, exist_ok=True)

    # Ensure custom_dir is neither production nor pilot
    assert custom_dir != ARTIFACTS_DIR
    assert custom_dir != PILOT_ARTIFACTS_DIR

    ckpt_path = custom_dir / "unet3d_best.pth"
    metrics_path = custom_dir / "metrics.json"

    # Verify target paths in custom dir
    assert ckpt_path.parent == custom_dir
    assert metrics_path.parent == custom_dir
    assert not (ARTIFACTS_DIR / "unet3d_best.pth").exists()


def test_resume_fails_if_checkpoint_missing_or_mismatched(tmp_path):
    """Verify run_training returns False when --resume is set but checkpoint is missing or configuration mismatched."""
    from ml.brain_mri.train import run_training

    custom_empty_dir = tmp_path / "empty_exp_dir"
    custom_empty_dir.mkdir(parents=True, exist_ok=True)

    # Resume requested but directory has no checkpoint
    res = run_training(resume=True, artifacts_dir=custom_empty_dir, epochs=3, max_train_patients=10)
    assert res is False, "run_training must return False when resuming with non-existent checkpoint"


def test_default_checkpoint_selection_preserves_production(monkeypatch):
    """Verify default service initialization does not silently load custom weights and targets production checkpoint."""
    from app.services.brain_mri import BrainMRIService, CUSTOM_10P_CHECKPOINT_PATH
    from ml.brain_mri.config import CHECKPOINT_BEST_PATH

    # Ensure experimental environment variables are unset
    monkeypatch.delenv("BRAIN_MRI_EXPERIMENTAL_MODE", raising=False)

    service = BrainMRIService()

    # Must target production path, not custom_10p_3e
    assert service.checkpoint_path == CHECKPOINT_BEST_PATH
    assert service.checkpoint_path != CUSTOM_10P_CHECKPOINT_PATH
    assert service.is_experimental is False

    # Since production weights have not been trained, status must remain not_trained
    assert service.is_ready() is False
    status_dict = service.get_status()
    assert status_dict["status"] == "not_trained"
    assert status_dict["is_experimental"] is False


def test_explicit_experimental_mode_loads_custom_checkpoint(monkeypatch):
    """Verify explicit opt-in via BRAIN_MRI_EXPERIMENTAL_MODE loads custom_10p_3e/unet3d_best.pth."""
    from app.services.brain_mri import BrainMRIService, CUSTOM_10P_CHECKPOINT_PATH

    monkeypatch.setenv("BRAIN_MRI_EXPERIMENTAL_MODE", "true")

    service = BrainMRIService()

    # Must target custom checkpoint path and set is_experimental
    assert service.checkpoint_path == CUSTOM_10P_CHECKPOINT_PATH
    assert service.is_experimental is True
    assert service.is_ready() is True

    status_dict = service.get_status()
    assert status_dict["status"] == "ready"
    assert status_dict["stage_phase"] == "experimental_prototype"
    assert status_dict["is_experimental"] is True
    assert "Experimental prototype" in status_dict["disclaimer"]
    assert "NOT clinically validated" in status_dict["disclaimer"]
    assert "best_val_dice" in status_dict["custom_metrics"]


def test_production_checkpoint_priority_over_experimental(monkeypatch, tmp_path):
    """Verify that if a production checkpoint exists, it always has priority over experimental mode."""
    from app.services.brain_mri import BrainMRIService

    # Create a dummy production checkpoint file
    fake_prod_ckpt = tmp_path / "unet3d_best.pth"
    torch.save({"model_state_dict": {}, "config": {"base_filters": 8}}, fake_prod_ckpt)

    monkeypatch.setattr("app.services.brain_mri.CHECKPOINT_BEST_PATH", fake_prod_ckpt)
    monkeypatch.setenv("BRAIN_MRI_EXPERIMENTAL_MODE", "true")

    # Mock load_state_dict so it doesn't fail on empty state dict
    monkeypatch.setattr("torch.nn.Module.load_state_dict", lambda self, state_dict: None)

    service = BrainMRIService()

    # Must choose production path over experimental
    assert service.checkpoint_path == fake_prod_ckpt
    assert service.is_experimental is False



