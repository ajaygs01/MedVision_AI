"""
FastAPI router for Module 2: Brain MRI Lesion & Tumor Segmentation.
"""

import logging
from typing import Optional
from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status

from app.schemas.brain_mri import (
    BrainMRISegmentResponse,
    BrainMRISliceResponse,
    BrainMRIStatusResponse,
    BrainMRIValidationResponse,
)
from app.services.brain_mri import brain_mri_service

logger = logging.getLogger("medivision.api.brain_mri")
router = APIRouter(prefix="/brain-mri", tags=["Brain MRI Lesion Segmentation"])


@router.get(
    "/status",
    response_model=BrainMRIStatusResponse,
    status_code=status.HTTP_200_OK,
    summary="Retrieve Brain MRI Module Operational Status",
    description="Returns live status (ready, not_trained, or dataset_missing), checkpoint path, device, and dataset audit.",
)
async def get_brain_mri_status():
    res = brain_mri_service.get_status()
    return BrainMRIStatusResponse(**res)


@router.post(
    "/validate",
    response_model=BrainMRIValidationResponse,
    status_code=status.HTTP_200_OK,
    summary="Validate Brain MRI NIfTI Study",
    description="Inspects uploaded NIfTI volume (.nii, .nii.gz) or ZIP study archive without executing segmentation.",
)
async def validate_brain_mri_study(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No filename provided.")

    file_bytes = await file.read()
    if len(file_bytes) == 0:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file is empty.")

    res = brain_mri_service.validate_study_file(file_bytes, file.filename)
    return BrainMRIValidationResponse(**res)


@router.post(
    "/segment",
    response_model=BrainMRISegmentResponse,
    status_code=status.HTTP_200_OK,
    summary="Execute Volumetric 3D U-Net Brain Tumor Segmentation",
    description="Delineates multi-class glioblastoma/glioma subregions (Whole Tumor, Tumor Core, Enhancing Tumor) and persists telemetry to MongoDB Atlas.",
)
async def segment_brain_mri_study(
    file: UploadFile = File(...),
    slice_idx: Optional[int] = Form(None),
    plane: Optional[str] = Form("axial"),
):
    if not file.filename:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No filename provided.")

    file_bytes = await file.read()
    if len(file_bytes) == 0:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file is empty.")

    result = brain_mri_service.segment_study(
        file_bytes=file_bytes,
        filename=file.filename,
        slice_idx=slice_idx,
        plane=plane or "axial",
    )

    if result.get("status") == "error":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=result.get("message", "Volumetric segmentation failed."),
        )

    # Persist in MongoDB Atlas / Local MongoDB if study segmented successfully
    if result.get("status") == "success":
        try:
            from app.db.mongodb import save_prediction_record
            slices = result.get("slices", {})
            record = {
                "modality": "brain_mri",
                "model_name": result.get("model_name", "3D U-Net Volumetric Brain Tumor Segmenter"),
                "filename": file.filename,
                "file_type": "nifti",
                "prediction": result.get("dominant_finding"),
                "predicted_class_id": 1 if result.get("volumetric_metrics", {}).get("whole_tumor_cm3", 0) > 0 else 0,
                "confidence": 0.88,
                "threshold": 0.5,
                "status": "success",
                "is_experimental": result.get("is_experimental", False),
                "disclaimer": result.get("disclaimer"),
                "image_preview": slices.get("mri_preview"),
                "gradcam_preview": slices.get("overlay_preview"),
                "heatmap_preview": slices.get("mask_preview"),
                "volumetric_metrics": result.get("volumetric_metrics"),
                "patient_id": result.get("patient_id"),
                "slices": {
                    "plane": slices.get("plane"),
                    "slice_idx": slices.get("slice_idx"),
                    "total_slices": slices.get("total_slices"),
                },
            }
            inserted_id = await save_prediction_record(record)
            if inserted_id:
                result["id"] = inserted_id
        except Exception as exc:
            logger.warning(f"Could not persist Brain MRI telemetry to MongoDB: {exc}")

    return BrainMRISegmentResponse(**result)


@router.get(
    "/slice",
    response_model=BrainMRISliceResponse,
    status_code=status.HTTP_200_OK,
    summary="Extract Interactive Multi-Planar Slice Visualization",
    description="Returns base64 PNG previews (MRI, Mask, and Blended Overlay) for any anatomical plane and slice index.",
)
async def get_brain_mri_slice(
    slice_idx: Optional[int] = None,
    plane: Optional[str] = "axial",
):
    valid_planes = {"axial", "coronal", "sagittal"}
    target_plane = plane.lower() if plane and plane.lower() in valid_planes else "axial"
    res = brain_mri_service.get_slice_visualization(slice_idx=slice_idx, plane=target_plane)
    if res.get("error"):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=res["error"],
        )
    return BrainMRISliceResponse(**res)
