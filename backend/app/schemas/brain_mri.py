"""
Pydantic Schemas for Module 2: Brain MRI Lesion & Tumor Segmentation.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class BrainMRIStatusResponse(BaseModel):
    status: str = Field(..., description="Operational status: ready, not_trained, or dataset_missing")
    stage_phase: str = Field(
        default="architecture_ready",
        description="Fine-grained lifecycle stage: architecture_ready, dataset_ready, training_in_progress, or model_trained",
    )
    is_pilot: bool = Field(default=False, description="True if trained as pilot / smoke-test verification model")
    model_name: str
    architecture: str
    checkpoint_path: Optional[str] = None
    pilot_checkpoint_path: Optional[str] = None
    is_experimental: bool = Field(default=False, description="True if operating in explicit experimental prototype mode")
    experimental_checkpoint_path: Optional[str] = None
    device: str
    dataset_status: Dict[str, Any]
    metrics: Dict[str, Any]
    pilot_metrics: Dict[str, Any] = Field(default_factory=dict)
    custom_metrics: Dict[str, Any] = Field(default_factory=dict)
    disclaimer: str


class BrainMRIValidationResponse(BaseModel):
    status: str = Field(..., description="Validation outcome: valid, invalid, or incomplete")
    filename: str
    detected_modalities: List[str]
    missing_modalities: List[str]
    dimensions: Optional[List[int]] = None
    voxel_spacing_mm: Optional[List[float]] = None
    message: str


class BrainMRISegmentResponse(BaseModel):
    id: Optional[str] = None
    status: str = Field(..., description="Inference outcome: success, not_trained, or error")
    model_name: str
    filename: str
    patient_id: Optional[str] = None
    volumetric_metrics: Dict[str, Any]
    dominant_finding: str
    slices: Dict[str, Any]
    is_experimental: bool = Field(default=False, description="True if segment was produced by experimental prototype model")
    message: str
    disclaimer: str


class BrainMRISliceResponse(BaseModel):
    plane: str = Field(..., description="Anatomical plane: axial, coronal, or sagittal")
    slice_idx: int = Field(..., description="0-indexed slice coordinate")
    total_slices: int = Field(..., description="Total slices along selected dimension")
    mri_preview: str = Field(..., description="Base64 encoded PNG data URI of the 2D MRI grayscale cross-section")
    mask_preview: Optional[str] = Field(None, description="Base64 encoded PNG data URI of the colorized segmentation mask")
    overlay_preview: Optional[str] = Field(None, description="Base64 encoded PNG data URI of the alpha-blended overlay")

