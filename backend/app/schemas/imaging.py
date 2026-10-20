from typing import Optional
from pydantic import BaseModel, Field


class ChestXRayClassifyResponse(BaseModel):
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
