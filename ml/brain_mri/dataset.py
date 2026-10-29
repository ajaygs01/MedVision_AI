"""
BraTS multi-parametric MRI dataset scanner, patient-level splitter, and PyTorch Dataset loader.
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset

from ml.brain_mri.config import (
    DATASET_DIR,
    DEFAULT_PATCH_SIZE,
    MODALITIES,
    RANDOM_SEED,
    TEST_RATIO,
    TRAIN_RATIO,
    VAL_RATIO,
)

_scan_cache: Dict[str, Any] = {"timestamp": 0.0, "path": "", "data": []}


def scan_brats_patients(data_dir: Optional[Path] = None, force_rescan: bool = False) -> List[Dict[str, Any]]:
    """
    Scans the BraTS directory for patient study folders containing
    FLAIR, T1, T1ce, T2, and segmentation ground-truth NIfTI files.
    Supports direct, shallow, or arbitrarily nested folder hierarchies
    (e.g. datasets/brats/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/...)
    without hardcoding a single fixed path.
    Caches scan results in-memory for 60 seconds to provide sub-millisecond API responses.
    """
    if data_dir is None:
        data_dir = DATASET_DIR

    data_path = Path(data_dir)
    if not data_path.exists():
        return []

    path_str = str(data_path.resolve())
    now = time.time()
    if not force_rescan and (now - _scan_cache["timestamp"] < 60.0) and (_scan_cache["path"] == path_str):
        return list(_scan_cache["data"])

    # Recursively find all directories containing NIfTI files (.nii or .nii.gz)
    candidate_dirs: List[Path] = []
    for root, dirs, files in os.walk(data_path):
        if any(f.lower().endswith((".nii", ".nii.gz")) for f in files):
            candidate_dirs.append(Path(root))

    patients_dict: Dict[str, Dict[str, Any]] = {}

    for folder in candidate_dirs:
        # Determine candidate patient ID from folder name
        patient_id = folder.name

        modality_files: Dict[str, Optional[Path]] = {m: None for m in MODALITIES}
        seg_file: Optional[Path] = None

        try:
            folder_files = list(folder.iterdir())
        except Exception:
            continue

        for f in folder_files:
            fname_lower = f.name.lower()
            if not (fname_lower.endswith(".nii") or fname_lower.endswith(".nii.gz")):
                continue

            if "flair" in fname_lower:
                modality_files["flair"] = f
            elif "t1ce" in fname_lower or "t1-ce" in fname_lower or "t1post" in fname_lower:
                modality_files["t1ce"] = f
            elif "t1" in fname_lower and "t1ce" not in fname_lower and "t1-ce" not in fname_lower and "t1post" not in fname_lower:
                modality_files["t1"] = f
            elif "t2" in fname_lower:
                modality_files["t2"] = f
            elif "seg" in fname_lower or "mask" in fname_lower or "label" in fname_lower:
                seg_file = f

        has_all_modalities = all(modality_files[m] is not None for m in MODALITIES)
        is_complete = has_all_modalities and (seg_file is not None)

        if has_all_modalities or seg_file is not None:
            record = {
                "patient_id": patient_id,
                "folder_path": str(folder),
                "flair": str(modality_files["flair"]) if modality_files["flair"] else None,
                "t1": str(modality_files["t1"]) if modality_files["t1"] else None,
                "t1ce": str(modality_files["t1ce"]) if modality_files["t1ce"] else None,
                "t2": str(modality_files["t2"]) if modality_files["t2"] else None,
                "seg": str(seg_file) if seg_file else None,
                "is_complete": is_complete,
            }

            # Deduplication: if patient_id was already encountered in another directory,
            # upgrade to complete record if the current one is complete
            if patient_id in patients_dict:
                if not patients_dict[patient_id]["is_complete"] and is_complete:
                    patients_dict[patient_id] = record
            else:
                patients_dict[patient_id] = record

    # Sort deterministically by patient_id
    patients = sorted(patients_dict.values(), key=lambda p: p["patient_id"])
    _scan_cache["timestamp"] = time.time()
    _scan_cache["path"] = path_str
    _scan_cache["data"] = patients
    return patients


def get_patient_splits(
    patients: List[Dict[str, Any]],
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    test_ratio: float = TEST_RATIO,
    seed: int = RANDOM_SEED,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Partitions patients strictly by patient ID into train, validation,
    and test splits to prevent data leakage across volumetric scans.
    """
    complete_patients = [p for p in patients if p.get("is_complete")]
    if not complete_patients:
        complete_patients = patients

    rng = random.Random(seed)
    shuffled = list(complete_patients)
    rng.shuffle(shuffled)

    n_total = len(shuffled)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_set = shuffled[:n_train]
    val_set = shuffled[n_train : n_train + n_val]
    test_set = shuffled[n_train + n_val :]

    return train_set, val_set, test_set


def verify_brats_dataset(data_dir: Optional[Path] = None) -> Dict[str, Any]:
    """
    Validates availability and integrity of the BraTS dataset on disk.
    Returns status, patient counts, and resolution steps if missing.
    """
    if data_dir is None:
        data_dir = DATASET_DIR

    data_path = Path(data_dir)
    if not data_path.exists():
        return {
            "status": "dataset_missing",
            "message": f"BraTS directory not found at {data_path.resolve()}.",
            "patient_count": 0,
            "complete_patient_count": 0,
            "incomplete_patient_count": 0,
            "data_dir": str(data_path.resolve()),
            "setup_guide": "See datasets/brats/README.md for step-by-step download instructions via Kaggle CLI.",
        }

    patients = scan_brats_patients(data_path)
    complete = [p for p in patients if p.get("is_complete")]
    incomplete = [p for p in patients if not p.get("is_complete")]

    if not patients:
        return {
            "status": "dataset_missing",
            "message": f"BraTS directory exists at {data_path.resolve()}, but contains no patient studies.",
            "patient_count": 0,
            "complete_patient_count": 0,
            "incomplete_patient_count": 0,
            "data_dir": str(data_path.resolve()),
            "setup_guide": "Download BraTS 2020/2021 training dataset and extract into datasets/brats/.",
        }

    train_set, val_set, test_set = get_patient_splits(complete)

    return {
        "status": "ready" if len(complete) >= 3 else "partial",
        "message": f"Found {len(patients)} total patient studies ({len(complete)} complete with all 4 modalities and segmentation mask; {len(incomplete)} incomplete).",
        "patient_count": len(patients),
        "complete_patient_count": len(complete),
        "incomplete_patient_count": len(incomplete),
        "data_dir": str(data_path.resolve()),
        "splits": {
            "train": len(train_set),
            "val": len(val_set),
            "test": len(test_set),
        },
        "sample_patient": complete[0]["patient_id"] if complete else None,
        "sample_patient_path": complete[0]["folder_path"] if complete else None,
    }


class BraTSDataset(Dataset):
    """
    PyTorch Dataset loading 4-channel mpMRI volumes (FLAIR, T1, T1ce, T2)
    and corresponding multi-class ground-truth segmentation masks.
    """

    def __init__(
        self,
        patients: List[Dict[str, Any]],
        patch_size: Tuple[int, int, int] = DEFAULT_PATCH_SIZE,
        is_training: bool = True,
    ):
        self.patients = [p for p in patients if p.get("is_complete")]
        self.patch_size = patch_size
        self.is_training = is_training

    def __len__(self) -> int:
        return len(self.patients)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        patient = self.patients[idx]

        # Load 4 modalities
        mod_arrays = []
        for mod in MODALITIES:
            file_path = patient[mod]
            nii = nib.load(file_path)
            data = nii.get_fdata().astype(np.float32)
            # Channel-wise non-zero intensity z-score normalization
            brain_mask = data > 0
            if brain_mask.sum() > 0:
                mean = data[brain_mask].mean()
                std = data[brain_mask].std() + 1e-8
                data[brain_mask] = (data[brain_mask] - mean) / std
            mod_arrays.append(data)

        # Stack into 4-channel volume: [4, H, W, D]
        volume = np.stack(mod_arrays, axis=0)

        # Load segmentation mask
        seg_nii = nib.load(patient["seg"])
        seg_data = seg_nii.get_fdata().astype(np.int64)

        # Remap BraTS label 4 (enhancing tumor) to label 3
        # Standard: 0=background, 1=NCR/NET, 2=ED, 4=ET -> 3
        seg_data[seg_data == 4] = 3

        # Transpose to [C, D, H, W] for PyTorch 3D Conv (D=Depth/slices, H, W)
        volume = np.transpose(volume, (0, 3, 1, 2))  # [4, D, H, W]
        seg_data = np.transpose(seg_data, (2, 0, 1))  # [D, H, W]

        # Crop / pad to patch size
        target_d, target_h, target_w = self.patch_size
        cur_d, cur_h, cur_w = seg_data.shape

        if self.is_training:
            # Random crop within brain bounds
            d_start = random.randint(0, max(0, cur_d - target_d))
            h_start = random.randint(0, max(0, cur_h - target_h))
            w_start = random.randint(0, max(0, cur_w - target_w))
        else:
            # Centered crop
            d_start = max(0, (cur_d - target_d) // 2)
            h_start = max(0, (cur_h - target_h) // 2)
            w_start = max(0, (cur_w - target_w) // 2)

        vol_patch = volume[
            :,
            d_start : d_start + target_d,
            h_start : h_start + target_h,
            w_start : w_start + target_w,
        ]
        seg_patch = seg_data[
            d_start : d_start + target_d,
            h_start : h_start + target_h,
            w_start : w_start + target_w,
        ]

        # Pad if volume is smaller than patch size
        pad_d = max(0, target_d - vol_patch.shape[1])
        pad_h = max(0, target_h - vol_patch.shape[2])
        pad_w = max(0, target_w - vol_patch.shape[3])
        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            vol_patch = np.pad(vol_patch, ((0, 0), (0, pad_d), (0, pad_h), (0, pad_w)), mode="constant")
            seg_patch = np.pad(seg_patch, ((0, pad_d), (0, pad_h), (0, pad_w)), mode="constant")

        return {
            "image": torch.from_numpy(vol_patch).float(),
            "mask": torch.from_numpy(seg_patch).long(),
            "patient_id": patient["patient_id"],
        }


def main():
    parser = argparse.ArgumentParser(description="BraTS Dataset Verification and Scanner")
    parser.add_argument("--verify", action="store_true", help="Inspect and verify dataset availability")
    parser.add_argument("--data-dir", type=str, default=str(DATASET_DIR), help="Path to BraTS dataset directory")
    args = parser.parse_args()

    info = verify_brats_dataset(Path(args.data_dir))
    print("=" * 70)
    print("MEDIVISION AI: BRATS BRAIN MRI DATASET VERIFICATION")
    print("=" * 70)
    print(f"Status:                 {info.get('status').upper()}")
    print(f"Dataset Directory:      {info.get('data_dir')}")
    print(f"Total Patient Studies:  {info.get('patient_count')}")
    print(f"Complete Studies (4ch): {info.get('complete_patient_count')}")
    print(f"Incomplete Studies:     {info.get('incomplete_patient_count', 0)}")
    print(f"Message:                {info.get('message')}")

    if info.get("status") in ["ready", "partial"]:
        splits = info.get("splits", {})
        print(f"Patient-Level Splits:   Train={splits.get('train')} (80%), Val={splits.get('val')} (10%), Test={splits.get('test')} (10%)")
        print(f"Sample Patient ID:      {info.get('sample_patient')}")
        if info.get("sample_patient_path"):
            print(f"Sample Patient Path:    {info.get('sample_patient_path')}")
    else:
        print("\nSetup Required:")
        print(f"  {info.get('setup_guide')}")

    print("=" * 70)


if __name__ == "__main__":
    main()
