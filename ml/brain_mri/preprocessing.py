"""
Preprocessing utilities, normalization, volumetric statistics, and slice visualization for Brain MRI.
"""

import base64
import io
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import nibabel as nib
import numpy as np
import torch
from PIL import Image


def normalize_mri_channel(channel: np.ndarray) -> np.ndarray:
    """Channel-wise non-zero intensity z-score normalization."""
    norm = channel.astype(np.float32)
    brain_mask = norm > 0
    if brain_mask.sum() > 0:
        mean = float(norm[brain_mask].mean())
        std = float(norm[brain_mask].std()) + 1e-8
        norm[brain_mask] = (norm[brain_mask] - mean) / std
    return norm


def load_nifti_volume(file_path: Union[str, Path]) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    """Loads a NIfTI volume and extracts its data array and voxel spacing in mm."""
    nii = nib.load(str(file_path))
    data = nii.get_fdata().astype(np.float32)
    zooms = nii.header.get_zooms()[:3]
    voxel_spacing = (float(zooms[0]), float(zooms[1]), float(zooms[2]))
    return data, voxel_spacing


def calculate_volumetric_stats(
    mask: np.ndarray,
    voxel_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> Dict[str, Any]:
    """
    Computes volumetric measurements for BraTS subregions:
      - Whole Tumor (WT: labels 1, 2, 3)
      - Tumor Core (TC: labels 1, 3)
      - Enhancing Tumor (ET: label 3)
      - Peritumoral Edema (ED: label 2)
    Volumes are returned in cm3 (cubic centimeters) and mm3.
    """
    voxel_volume_mm3 = float(voxel_spacing[0] * voxel_spacing[1] * voxel_spacing[2])

    wt_voxels = int((mask >= 1).sum())
    tc_voxels = int(((mask == 1) | (mask == 3)).sum())
    et_voxels = int((mask == 3).sum())
    ed_voxels = int((mask == 2).sum())
    ncr_voxels = int((mask == 1).sum())

    wt_vol_cm3 = round((wt_voxels * voxel_volume_mm3) / 1000.0, 2)
    tc_vol_cm3 = round((tc_voxels * voxel_volume_mm3) / 1000.0, 2)
    et_vol_cm3 = round((et_voxels * voxel_volume_mm3) / 1000.0, 2)
    ed_vol_cm3 = round((ed_voxels * voxel_volume_mm3) / 1000.0, 2)
    ncr_vol_cm3 = round((ncr_voxels * voxel_volume_mm3) / 1000.0, 2)

    return {
        "whole_tumor_cm3": wt_vol_cm3,
        "tumor_core_cm3": tc_vol_cm3,
        "enhancing_tumor_cm3": et_vol_cm3,
        "edema_cm3": ed_vol_cm3,
        "necrotic_core_cm3": ncr_vol_cm3,
        "total_lesion_voxels": wt_voxels,
        "voxel_spacing_mm": list(voxel_spacing),
    }


def find_most_informative_slice(mask: Optional[np.ndarray], total_slices: int) -> int:
    """Finds the slice index with maximum tumor area, or the central slice if empty."""
    if mask is None or mask.sum() == 0:
        return total_slices // 2
    # Count non-zero tumor voxels per slice
    slice_counts = [(mask[s] > 0).sum() for s in range(total_slices)]
    max_idx = int(np.argmax(slice_counts))
    return max_idx if slice_counts[max_idx] > 0 else total_slices // 2


def colorize_mask(mask_slice: np.ndarray) -> np.ndarray:
    """
    Renders integer segmentation labels as an RGBA image:
      Label 0 (Background): Transparent [0, 0, 0, 0]
      Label 1 (NCR/NET): Red / Rose [225, 29, 72, 220]
      Label 2 (Edema): Emerald Green [5, 150, 105, 200]
      Label 3 (Enhancing): Amber / Gold [217, 119, 6, 230]
    """
    h, w = mask_slice.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)

    # Label 1: NCR/NET (Rose)
    m1 = mask_slice == 1
    rgba[m1] = [225, 29, 72, 220]

    # Label 2: Edema (Emerald)
    m2 = mask_slice == 2
    rgba[m2] = [5, 150, 105, 200]

    # Label 3: Enhancing Tumor (Gold/Amber)
    m3 = mask_slice == 3
    rgba[m3] = [217, 119, 6, 230]

    return rgba


def array_to_base64_png(image_array: np.ndarray) -> str:
    """Encodes a uint8 RGB/RGBA or grayscale array to a base64 PNG data URL."""
    pil_img = Image.fromarray(image_array)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64_str}"


def extract_slice_visualizations(
    mri_volume: np.ndarray,  # [D, H, W] single channel (e.g. FLAIR or T1ce)
    mask_volume: Optional[np.ndarray] = None,  # [D, H, W] integer labels
    slice_idx: Optional[int] = None,
    plane: str = "axial",  # axial (D), coronal (H), sagittal (W)
    alpha: float = 0.45,
) -> Dict[str, Any]:
    """
    Extracts 2D MRI slice, color-coded segmentation mask, and blended overlay
    for the chosen anatomical plane and slice index.
    """
    # Reorient volume according to plane
    if plane == "coronal":
        vol = np.transpose(mri_volume, (1, 0, 2))  # [H, D, W]
        msk = np.transpose(mask_volume, (1, 0, 2)) if mask_volume is not None else None
    elif plane == "sagittal":
        vol = np.transpose(mri_volume, (2, 0, 1))  # [W, D, H]
        msk = np.transpose(mask_volume, (2, 0, 1)) if mask_volume is not None else None
    else:
        # Axial
        vol = mri_volume
        msk = mask_volume

    total_slices = vol.shape[0]
    if slice_idx is None or slice_idx < 0 or slice_idx >= total_slices:
        slice_idx = find_most_informative_slice(msk, total_slices)

    # Extract 2D slice and normalize to [0, 255] uint8 for display
    raw_slice = vol[slice_idx]
    v_min, v_max = float(raw_slice.min()), float(raw_slice.max())
    if v_max > v_min:
        mri_uint8 = np.clip(((raw_slice - v_min) / (v_max - v_min)) * 255.0, 0, 255).astype(np.uint8)
    else:
        mri_uint8 = np.zeros_like(raw_slice, dtype=np.uint8)

    # Convert to RGB
    mri_rgb = np.stack([mri_uint8, mri_uint8, mri_uint8], axis=-1)

    mask_base64 = None
    overlay_base64 = None

    if msk is not None:
        mask_slice = msk[slice_idx]
        mask_rgba = colorize_mask(mask_slice)
        mask_base64 = array_to_base64_png(mask_rgba)

        # Blend overlay: MRI + mask
        overlay = mri_rgb.copy()
        tumor_mask = mask_rgba[..., 3] > 0
        if tumor_mask.any():
            color_layer = mask_rgba[tumor_mask, :3]
            overlay[tumor_mask] = (
                (1.0 - alpha) * overlay[tumor_mask].astype(np.float32) +
                alpha * color_layer.astype(np.float32)
            ).astype(np.uint8)

        overlay_base64 = array_to_base64_png(overlay)

    mri_base64 = array_to_base64_png(mri_rgb)

    return {
        "plane": plane,
        "slice_idx": slice_idx,
        "total_slices": total_slices,
        "mri_preview": mri_base64,
        "mask_preview": mask_base64,
        "overlay_preview": overlay_base64,
    }
