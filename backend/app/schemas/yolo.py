"""
Pydantic schemas for YOLO Anatomical Lesion Localization requests and responses.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    id: int = Field(..., description="Unique detection index")
    class_name: str = Field(..., description="Detected anatomical finding class name")
    class_id: int = Field(..., description="Target class index")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    bbox_pixels: List[float] = Field(..., min_length=4, max_length=4, description="[x1, y1, x2, y2] in pixels")
    bbox_normalized: List[float] = Field(
        ..., min_length=4, max_length=4, description="[x_center, y_center, width, height] normalized to [0, 1]"
    )
    area_pixels: float = Field(..., ge=0.0, description="Area in square pixels")


class YOLODetectionStatusResponse(BaseModel):
    model_architecture: str = Field(default="YOLOv8n", description="Model architecture")
    trained: bool = Field(default=False, description="Whether valid model weights exist")
    status: str = Field(default="not_trained", description="'ready', 'not_trained', or 'unconfigured'")
    checkpoint_path: Optional[str] = Field(default=None, description="Path to checkpoint if trained")
    classes: List[str] = Field(default=["lung_opacity"], description="Detectable classes")
    metrics: Optional[Dict[str, Any]] = Field(default=None, description="Validation metrics if available")


class YOLODetectionResponse(BaseModel):
    status: str = Field(..., description="'success', 'not_trained', or 'error'")
    message: str = Field(..., description="Diagnostic summary or status message")
    filename: Optional[str] = Field(default=None, description="Uploaded image filename")
    detections: List[BoundingBox] = Field(default_factory=list, description="List of detected bounding boxes")
    num_detections: int = Field(default=0, description="Count of detected objects")
    image_dimensions: Optional[Dict[str, int]] = Field(default=None, description="Width and height of the image")
    conf_threshold: float = Field(default=0.25, description="Confidence threshold used")
    iou_threshold: float = Field(default=0.45, description="IoU threshold used for NMS")
    preview_image: Optional[str] = Field(default=None, description="Base64 data URL of image with detection overlay")
    model_status: YOLODetectionStatusResponse
    persisted: bool = Field(default=False, description="Whether result was stored in analysis history")
    analysis_id: Optional[str] = Field(default=None, description="Database record ID if persisted")
    highest_candidate_confidence: Optional[float] = Field(
        default=None, description="Highest raw candidate confidence score if zero detections passed threshold"
    )
    disclaimer: str = Field(
        default=(
            "Anatomical object detection is an investigational research prototype. "
            "Bounding boxes are indicative of localization patterns and must not be used "
            "as a standalone clinical diagnosis."
        ),
        description="Clinical and research disclaimer",
    )
