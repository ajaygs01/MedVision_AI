"""
Brain MRI Lesion & Tumor Segmentation Service Layer.
Manages 3D U-Net model lifecycle, NIfTI study ingestion, inference, and volumetric quantification.
"""

import io
import json
import logging
import os
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

_project_root = Path(__file__).resolve().parents[3]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import nibabel as nib
import numpy as np
import torch

from app.core.config import get_settings
from ml.brain_mri.config import (
    ARTIFACTS_DIR,
    CHECKPOINT_BEST_PATH,
    CHECKPOINT_LAST_PATH,
    CHECKPOINT_PILOT_PATH,
    DATASET_DIR,
    IN_CHANNELS,
    METRICS_PATH,
    METRICS_PILOT_PATH,
    MODALITIES,
    NUM_CLASSES,
)
from ml.brain_mri.dataset import verify_brats_dataset
from ml.brain_mri.model import UNet3D
from ml.brain_mri.preprocessing import (
    calculate_volumetric_stats,
    extract_slice_visualizations,
    normalize_mri_channel,
)

CUSTOM_10P_DIR = ARTIFACTS_DIR / "custom_10p_3e"
CUSTOM_10P_CHECKPOINT_PATH = CUSTOM_10P_DIR / "unet3d_best.pth"
CUSTOM_10P_METRICS_PATH = CUSTOM_10P_DIR / "metrics.json"

logger = logging.getLogger("medivision.services.brain_mri")


class BrainMRIService:
    """Manages 3D U-Net loading, verification, and volumetric inference."""

    def __init__(self, checkpoint_path: Optional[Path] = None):
        settings = get_settings()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[UNet3D] = None
        self.is_experimental: bool = False
        self.custom_metrics: Dict[str, Any] = {}
        self.cached_volume: Optional[np.ndarray] = None
        self.cached_mask: Optional[np.ndarray] = None

        if checkpoint_path is not None:
            self.checkpoint_path = checkpoint_path
            if "custom_10p_3e" in str(checkpoint_path) or "experimental" in str(checkpoint_path):
                self.is_experimental = True
        elif CHECKPOINT_BEST_PATH.exists():
            # Production model has absolute priority
            self.checkpoint_path = CHECKPOINT_BEST_PATH
            self.is_experimental = False
        elif (
            settings.brain_mri_experimental_mode
            or os.environ.get("BRAIN_MRI_EXPERIMENTAL_MODE", "").lower() in ["true", "1", "yes"]
        ):
            # Explicit opt-in for experimental 10p/3e testing
            custom_ckpt = (
                Path(settings.brain_mri_custom_checkpoint_path)
                if settings.brain_mri_custom_checkpoint_path
                else CUSTOM_10P_CHECKPOINT_PATH
            )
            if custom_ckpt.exists():
                self.checkpoint_path = custom_ckpt
                self.is_experimental = True
            else:
                self.checkpoint_path = CHECKPOINT_BEST_PATH
                self.is_experimental = False
        else:
            # Default production path (never silently switch)
            self.checkpoint_path = CHECKPOINT_BEST_PATH
            self.is_experimental = False

        self._load_model()

    def _load_model(self):
        """Attempts to load model weights from checkpoint if available."""
        if not self.checkpoint_path.exists():
            logger.info(f"Brain MRI 3D U-Net checkpoint not found at {self.checkpoint_path} (unconfigured/not trained).")
            self.model = None
            return

        try:
            prefix = "EXPERIMENTAL " if self.is_experimental else ""
            logger.info(f"Loading {prefix}Brain MRI 3D U-Net weights from {self.checkpoint_path} on {self.device}...")
            ckpt = torch.load(self.checkpoint_path, map_location=self.device)
            base_filters = ckpt.get("config", {}).get("base_filters", 16)
            model = UNet3D(in_channels=IN_CHANNELS, num_classes=NUM_CLASSES, base_filters=base_filters)
            model.load_state_dict(ckpt["model_state_dict"])
            model.to(self.device)
            model.eval()
            self.model = model
            logger.info(f"Brain MRI 3D U-Net model ({'Experimental' if self.is_experimental else 'Production'}) successfully initialized.")
        except Exception as exc:
            logger.error(f"Failed to load Brain MRI checkpoint: {exc}")
            self.model = None

        # Load custom metrics if available
        if CUSTOM_10P_METRICS_PATH.exists():
            try:
                with open(CUSTOM_10P_METRICS_PATH, "r", encoding="utf-8") as f:
                    self.custom_metrics = json.load(f)
            except Exception:
                pass

    def is_ready(self) -> bool:
        """Returns True if the trained 3D U-Net model is loaded and ready."""
        return self.model is not None and self.checkpoint_path.exists()

    def get_status(self) -> Dict[str, Any]:
        """Returns comprehensive module readiness, dataset status, and evaluation metrics."""
        dataset_info = verify_brats_dataset(DATASET_DIR)

        # Read production metrics if available
        metrics = {}
        if METRICS_PATH.exists():
            try:
                with open(METRICS_PATH, "r", encoding="utf-8") as f:
                    metrics = json.load(f)
            except Exception:
                pass

        # Read pilot metrics if available
        pilot_metrics = {}
        if METRICS_PILOT_PATH.exists():
            try:
                with open(METRICS_PILOT_PATH, "r", encoding="utf-8") as f:
                    pilot_metrics = json.load(f)
            except Exception:
                pass

        is_pilot_verified = bool(CHECKPOINT_PILOT_PATH.exists() and METRICS_PILOT_PATH.exists())

        if self.is_ready():
            operational_status = "ready"
            stage_phase = "experimental_prototype" if self.is_experimental else "model_trained"
        elif dataset_info.get("status") == "dataset_missing":
            operational_status = "dataset_missing"
            stage_phase = "architecture_ready"
        elif dataset_info.get("status") in ["ready", "partial"]:
            if CHECKPOINT_LAST_PATH.exists() and not CHECKPOINT_BEST_PATH.exists():
                operational_status = "not_trained"
                stage_phase = "training_in_progress"
            else:
                operational_status = "not_trained"
                stage_phase = "dataset_ready"
        else:
            operational_status = "not_trained"
            stage_phase = "architecture_ready"

        if self.is_experimental:
            disclaimer = (
                "Experimental prototype model trained on limited 10-patient cohort for 3 epochs. "
                "Segmentations are strictly for investigational evaluation and NOT clinically validated or production-ready."
            )
            model_name = "3D U-Net Brain Tumor Segmenter (Experimental 10p/3e Prototype)"
        else:
            disclaimer = (
                "Academic research prototype. Segmentations are for investigational use only "
                "and must not be used as standalone clinical surgical plans."
            )
            model_name = "3D U-Net Volumetric Brain Tumor Segmenter"

        return {
            "status": operational_status,
            "stage_phase": stage_phase,
            "is_pilot": is_pilot_verified,
            "is_experimental": self.is_experimental,
            "model_name": model_name,
            "architecture": "4-Channel 3D Encoder-Decoder U-Net with Skip Connections",
            "checkpoint_path": str(self.checkpoint_path) if self.checkpoint_path.exists() else None,
            "pilot_checkpoint_path": str(CHECKPOINT_PILOT_PATH) if CHECKPOINT_PILOT_PATH.exists() else None,
            "experimental_checkpoint_path": str(CUSTOM_10P_CHECKPOINT_PATH) if CUSTOM_10P_CHECKPOINT_PATH.exists() else None,
            "device": str(self.device),
            "dataset_status": dataset_info,
            "metrics": metrics,
            "pilot_metrics": pilot_metrics,
            "custom_metrics": self.custom_metrics,
            "disclaimer": disclaimer,
        }

    def validate_study_file(self, file_bytes: bytes, filename: str) -> Dict[str, Any]:
        """Validates NIfTI format or ZIP study archive containing MRI sequences."""
        filename_lower = filename.lower()
        if not (filename_lower.endswith(".nii") or filename_lower.endswith(".nii.gz") or filename_lower.endswith(".zip")):
            return {
                "status": "invalid",
                "filename": filename,
                "detected_modalities": [],
                "missing_modalities": MODALITIES,
                "message": "Unsupported file format. Please upload NIfTI (.nii, .nii.gz) or ZIP study archive.",
            }

        if filename_lower.endswith(".zip"):
            try:
                with zipfile.ZipFile(io.BytesIO(file_bytes)) as z:
                    names = z.namelist()
                    detected = []
                    for m in MODALITIES:
                        if any(m in n.lower() for n in names):
                            detected.append(m)
                    missing = [m for m in MODALITIES if m not in detected]
                    status = "valid" if not missing else "incomplete"
                    return {
                        "status": status,
                        "filename": filename,
                        "detected_modalities": detected,
                        "missing_modalities": missing,
                        "message": (
                            f"ZIP archive contains {len(names)} files. Detected sequences: {', '.join(detected)}."
                            if not missing
                            else f"Incomplete study. Missing sequences: {', '.join(missing)}."
                        ),
                    }
            except Exception as exc:
                return {
                    "status": "invalid",
                    "filename": filename,
                    "detected_modalities": [],
                    "missing_modalities": MODALITIES,
                    "message": f"Corrupt ZIP archive: {exc}",
                }

        # Single NIfTI volume
        tmp_path = None
        try:
            suffix = ".nii.gz" if filename_lower.endswith(".gz") else ".nii"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(file_bytes)
                tmp_path = Path(tmp.name)

            nii = nib.load(str(tmp_path))
            shape = list(nii.shape)
            zooms = [float(z) for z in nii.header.get_zooms()[:3]]
            del nii
            if tmp_path and tmp_path.exists():
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

            detected = []
            for m in MODALITIES:
                if m in filename_lower:
                    detected.append(m)
            if not detected:
                detected = ["flair_or_composite"]

            return {
                "status": "valid",
                "filename": filename,
                "detected_modalities": detected,
                "missing_modalities": [m for m in MODALITIES if m not in detected],
                "dimensions": shape,
                "voxel_spacing_mm": zooms,
                "message": f"Valid NIfTI volume with dimensions {shape} and voxel spacing {zooms} mm.",
            }
        except Exception as exc:
            return {
                "status": "invalid",
                "filename": filename,
                "detected_modalities": [],
                "missing_modalities": MODALITIES,
                "message": f"Corrupt or unreadable NIfTI file: {exc}",
            }

    def segment_study(
        self,
        file_bytes: bytes,
        filename: str,
        slice_idx: Optional[int] = None,
        plane: str = "axial",
    ) -> Dict[str, Any]:
        """
        Runs volumetric inference or returns transparent not-trained status.
        Extracts 2D multi-planar slice previews with color-coded tumor subregions.
        """
        # If model is not trained, report clearly without faking output
        if not self.is_ready():
            return {
                "status": "not_trained",
                "model_name": "3D U-Net Volumetric Brain Tumor Segmenter",
                "filename": filename,
                "volumetric_metrics": {},
                "dominant_finding": "Model Not Trained (Stage 2 Pending)",
                "slices": {},
                "message": (
                    "Brain MRI 3D U-Net weights have not been trained yet. "
                    "In strict accordance with project boundaries, no mock predictions are provided. "
                    "Run `python ml/brain_mri/train.py` once the BraTS dataset is in place."
                ),
                "disclaimer": "Stage 2 Brain MRI pipeline requires trained model weights.",
            }

        vol_tensor = None
        mri_display = None
        spacing = (1.0, 1.0, 1.0)

        # Handle ZIP study archive containing multi-parametric sequences
        if filename.lower().endswith(".zip"):
            try:
                with tempfile.TemporaryDirectory() as tmp_dir:
                    zip_dir = Path(tmp_dir)
                    with zipfile.ZipFile(io.BytesIO(file_bytes)) as z:
                        z.extractall(zip_dir)

                    extracted_nii = [
                        p for p in zip_dir.rglob("*")
                        if p.is_file() and (p.name.lower().endswith(".nii") or p.name.lower().endswith(".nii.gz"))
                    ]

                    mod_map = {}
                    for f in extracted_nii:
                        fn = f.name.lower()
                        if "flair" in fn:
                            mod_map["flair"] = f
                        elif "t1ce" in fn or "t1-ce" in fn or "t1post" in fn:
                            mod_map["t1ce"] = f
                        elif "t1" in fn and "t1ce" not in fn and "t1-ce" not in fn and "t1post" not in fn:
                            mod_map["t1"] = f
                        elif "t2" in fn:
                            mod_map["t2"] = f

                    channels_data = []
                    for mod in MODALITIES:
                        f_path = mod_map.get(mod)
                        if f_path is None and mod_map:
                            f_path = next(iter(mod_map.values()))

                        if f_path is not None:
                            nii = nib.load(str(f_path))
                            data = nii.get_fdata().astype(np.float32)
                            zooms = nii.header.get_zooms()[:3]
                            spacing = (float(zooms[0]), float(zooms[1]), float(zooms[2]))
                            norm = normalize_mri_channel(data)
                            vol_d = np.transpose(norm, (2, 0, 1)) if norm.ndim == 3 and norm.shape[2] < norm.shape[0] else norm
                            channels_data.append(vol_d)

                    if len(channels_data) == 4:
                        vol_d = np.stack(channels_data, axis=0)  # [4, D, H, W]
                        mri_display = vol_d[0]
                        vol_tensor = torch.from_numpy(vol_d).unsqueeze(0)  # [1, 4, D, H, W]
                    elif len(channels_data) > 0:
                        vol_d = np.stack([channels_data[0]] * 4, axis=0)
                        mri_display = vol_d[0]
                        vol_tensor = torch.from_numpy(vol_d).unsqueeze(0)
                    else:
                        return {
                            "status": "error",
                            "message": "ZIP archive does not contain readable NIfTI (.nii, .nii.gz) MRI volumes.",
                        }
            except Exception as exc:
                logger.error(f"Failed to process ZIP MRI study: {exc}")
                return {
                    "status": "error",
                    "message": f"Corrupt or unreadable ZIP study archive: {exc}",
                }
        else:
            # Load single NIfTI volume
            is_gzipped = filename.lower().endswith(".gz") and file_bytes[:2] == b"\x1f\x8b"
            suffix = ".nii.gz" if is_gzipped else ".nii"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(file_bytes)
                tmp_path = Path(tmp.name)

            try:
                nii = nib.load(str(tmp_path))
                raw_data = nii.get_fdata().astype(np.float32)
                zooms = nii.header.get_zooms()[:3]
                spacing = (float(zooms[0]), float(zooms[1]), float(zooms[2]))
            finally:
                if tmp_path.exists():
                    os.remove(tmp_path)

            # Standardize volume shape to [4, D, H, W]
            if raw_data.ndim == 3:
                # Replicate single sequence across 4 channels if only 1 channel provided
                norm_channel = normalize_mri_channel(raw_data)
                vol_d = np.transpose(norm_channel, (2, 0, 1)) if norm_channel.shape[2] < norm_channel.shape[0] else norm_channel
                mri_display = vol_d
                vol_tensor = torch.from_numpy(vol_d).unsqueeze(0).repeat(4, 1, 1, 1).unsqueeze(0)  # [1, 4, D, H, W]
            elif raw_data.ndim == 4:
                # 4-channel volume [4, H, W, D] or [H, W, D, 4]
                if raw_data.shape[0] == 4:
                    channels = [normalize_mri_channel(raw_data[c]) for c in range(4)]
                    vol_d = np.stack([np.transpose(c, (2, 0, 1)) for c in channels], axis=0)  # [4, D, H, W]
                elif raw_data.shape[3] == 4:
                    channels = [normalize_mri_channel(raw_data[..., c]) for c in range(4)]
                    vol_d = np.stack([np.transpose(c, (2, 0, 1)) for c in channels], axis=0)  # [4, D, H, W]
                else:
                    vol_d = np.zeros((4, 32, 64, 64), dtype=np.float32)

                mri_display = vol_d[0]  # Display FLAIR channel
                vol_tensor = torch.from_numpy(vol_d).unsqueeze(0)  # [1, 4, D, H, W]
            else:
                return {
                    "status": "error",
                    "message": f"Invalid volume dimensions: {raw_data.shape}. Expected 3D or 4D volume.",
                }

        # Run forward pass through trained 3D U-Net
        self.model.eval()
        with torch.no_grad():
            x = vol_tensor.float().to(self.device)
            logits = self.model(x)
            pred_mask = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.int64)

        # Compute volumetric measurements
        vol_metrics = calculate_volumetric_stats(pred_mask, voxel_spacing=spacing)

        # Cache volume and mask for interactive multi-planar viewing
        self.cached_volume = mri_display
        self.cached_mask = pred_mask

        # Generate slice visualizations (auto-detect most informative slice on initial run)
        effective_slice_idx = None if (slice_idx is None or slice_idx == 0) else slice_idx
        slice_views = extract_slice_visualizations(
            mri_volume=mri_display,
            mask_volume=pred_mask,
            slice_idx=effective_slice_idx,
            plane=plane,
        )

        wt_vol = vol_metrics.get("whole_tumor_cm3", 0.0)
        dominant_prefix = "[Experimental 10p/3e Prototype] " if self.is_experimental else ""
        dominant = (
            f"{dominant_prefix}Volumetric Glioma/Lesion Delineated (Whole Tumor Volume: {wt_vol} cm³)"
            if wt_vol > 0.0
            else f"{dominant_prefix}No Significant Intra-Axial Lesion Detected"
        )

        model_name = (
            "3D U-Net Brain Tumor Segmenter (Experimental 10p/3e Prototype)"
            if self.is_experimental
            else "3D U-Net Volumetric Brain Tumor Segmenter"
        )
        disclaimer = (
            "Experimental prototype model trained on limited 10-patient cohort for 3 epochs. "
            "For investigational evaluation only; NOT clinically validated or production ready."
            if self.is_experimental
            else "Academic research prototype. Results must not be used as standalone clinical diagnoses."
        )

        return {
            "status": "success",
            "model_name": model_name,
            "filename": filename,
            "patient_id": Path(filename).stem.split("_")[0],
            "volumetric_metrics": vol_metrics,
            "dominant_finding": dominant,
            "slices": slice_views,
            "message": "3D U-Net volumetric inference completed successfully.",
            "disclaimer": disclaimer,
            "is_experimental": self.is_experimental,
        }

    def get_slice_visualization(
        self,
        slice_idx: Optional[int] = None,
        plane: str = "axial",
    ) -> Dict[str, Any]:
        """Extracts slice visualization for interactive multi-planar viewing."""
        if self.cached_volume is None:
            return {
                "plane": plane,
                "slice_idx": slice_idx or 0,
                "total_slices": 0,
                "mri_preview": "",
                "mask_preview": None,
                "overlay_preview": None,
                "error": "No segmented study volume loaded in memory. Please run segmentation first.",
            }
        return extract_slice_visualizations(
            mri_volume=self.cached_volume,
            mask_volume=self.cached_mask,
            slice_idx=slice_idx,
            plane=plane,
        )


brain_mri_service = BrainMRIService()
