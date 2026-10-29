from typing import Optional, List, Dict
from pydantic import BaseModel, Field


class SkinLesionClassProbability(BaseModel):
    class_code: str = Field(..., description="Short diagnostic code (e.g. MEL, NV, BCC)")
    class_name: str = Field(..., description="Full clinical diagnostic name")
    probability: float = Field(..., description="Calibrated posterior softmax probability [0.0 - 1.0]")


class SkinLesionClassifyResponse(BaseModel):
    status: str = Field(..., description="Prediction execution status: 'success', 'not_trained', or 'error'")
    model_name: str = Field(..., description="Underlying architecture and model descriptor")
    prediction: Optional[str] = Field(None, description="Primary diagnostic class prediction")
    predicted_class_code: Optional[str] = Field(None, description="Short diagnostic code (e.g. MEL, NV)")
    predicted_class_id: Optional[int] = Field(None, description="Integer class index [0-6]")
    confidence: Optional[float] = Field(None, description="Confidence score for top prediction [0.0 - 1.0]")
    probabilities: Optional[List[SkinLesionClassProbability]] = Field(None, description="Full multi-class probability distribution")
    filename: str = Field(..., description="Uploaded image filename")
    file_type: str = Field(default="standard_image", description="Dermoscopy image encoding")
    message: str = Field(..., description="Clinical interpretation or operational notice")
    disclaimer: str = Field(..., description="Mandatory medical research prototype disclaimer")
    image_preview: Optional[str] = Field(None, description="Base64 data URI of the ingested dermoscopy image")
    id: Optional[str] = Field(None, description="MongoDB Atlas case audit record ID")
