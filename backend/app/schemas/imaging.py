from typing import Optional
from pydantic import BaseModel, Field


class ChestXRayClassifyResponse(BaseModel):
    id: Optional[str] = Field(default=None, description="Persisted analysis record ID in MongoDB Atlas")
    status: str = Field(description="'success', 'not_trained', or 'error'")
    model_name: str = Field(default="Chest X-ray ResNet50 Classifier")
    prediction: Optional[str] = Field(default=None, description="Predicted class name (e.g. 'Lung Opacity / Pneumonia')")
    predicted_class_id: Optional[int] = Field(default=None, description="0 for Normal, 1 for Lung Opacity")
    confidence: Optional[float] = Field(default=None, description="Model sigmoid probability [0.0 - 1.0]")
    threshold: float = Field(default=0.5, description="Decision threshold")
    filename: str = Field(description="Uploaded filename")
    file_type: str = Field(description="'dicom' or 'standard_image'")
    message: str = Field(description="Status description or diagnostic details")
    disclaimer: str = Field(
        default="Academic research prototype. Results are not clinically validated and must not be used as a standalone medical diagnosis."
    )
    image_preview: Optional[str] = Field(default=None, description="Base64 data URL preview of analyzed radiograph")
    gradcam_preview: Optional[str] = Field(default=None, description="Base64 data URL preview of Grad-CAM attribution overlay")
    heatmap_preview: Optional[str] = Field(default=None, description="Base64 data URL preview of standalone Jet saliency heatmap")


class PredictionHistoryItem(BaseModel):
    id: str = Field(description="Unique history record identifier")
    timestamp: str = Field(description="ISO-8601 recording timestamp")
    modality: str = Field(default="chest_xray", description="Medical imaging modality ('chest_xray' or 'clinical_ml')")
    model_name: str = Field(description="Inference model name")
    filename: str = Field(description="Original filename analyzed")
    file_type: str = Field(description="'dicom', 'standard_image', or 'tabular_ehr'")
    prediction: Optional[str] = Field(default=None, description="Model classification result")
    predicted_class_id: Optional[int] = Field(default=None, description="Class ID")
    confidence: Optional[float] = Field(default=None, description="Confidence score or risk probability")
    threshold: float = Field(default=0.5, description="Decision threshold")
    status: str = Field(default="success", description="Execution status")
    disclaimer: Optional[str] = None
    image_preview: Optional[str] = Field(default=None, description="Base64 data URL preview of analyzed radiograph")
    gradcam_preview: Optional[str] = Field(default=None, description="Base64 data URL preview of Grad-CAM attribution overlay")
    heatmap_preview: Optional[str] = Field(default=None, description="Base64 data URL preview of standalone Jet saliency heatmap")
    risk_tier: Optional[str] = Field(default=None, description="Clinical risk tier (e.g. 'Low Risk', 'Elevated Risk')")
    risk_probability: Optional[float] = Field(default=None, description="Clinical risk probability [0.0 - 1.0]")
    top_contributing_factors: Optional[list[str]] = Field(default=None, description="Key physiological risk indicators")
    patient_data: Optional[dict] = Field(default=None, description="Structured patient clinical parameters")
    volumetric_metrics: Optional[dict] = Field(default=None, description="Brain MRI volumetric lesion metrics")
    slices: Optional[dict] = Field(default=None, description="Brain MRI multi-planar slice details and previews")
    detections: Optional[list[dict]] = Field(default=None, description="YOLO detection bounding boxes")
    num_detections: Optional[int] = Field(default=None, description="Count of detected objects")


class PredictionHistoryResponse(BaseModel):
    total: int = Field(description="Total records returned")
    records: list[PredictionHistoryItem] = Field(default_factory=list, description="History records list")
    database_connected: bool = Field(description="MongoDB Atlas connection status")
