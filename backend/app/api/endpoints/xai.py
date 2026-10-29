import logging
from typing import Any, Dict, Optional
from fastapi import APIRouter, File, HTTPException, UploadFile, status
from pydantic import BaseModel, Field

from app.services.xai import xai_service

logger = logging.getLogger("medivision.api.xai")
router = APIRouter(prefix="/xai", tags=["Explainable AI (XAI)"])


class GradCAMResponse(BaseModel):
    id: Optional[str] = Field(default=None, description="Persisted analysis record ID in MongoDB Atlas")
    status: str = Field(description="'success', 'not_trained', or 'error'")
    attribution_method: str = Field(default="Grad-CAM (Gradient-Weighted Class Activation Mapping)")
    target_layer: str = Field(description="Convolutional feature layer tapped for gradients")
    prediction: Optional[str] = Field(default=None, description="Class label evaluated")
    confidence: Optional[float] = Field(default=None, description="Model prediction probability")
    threshold: float = Field(default=0.5, description="Decision threshold")
    filename: str = Field(description="Uploaded image filename")
    original_preview: Optional[str] = Field(default=None, description="Base64 data URL of original radiograph")
    overlay_preview: Optional[str] = Field(default=None, description="Base64 data URL of radiograph with Grad-CAM Jet heatmap overlay")
    heatmap_preview: Optional[str] = Field(default=None, description="Base64 data URL of standalone Jet saliency heatmap")
    explanation: Optional[str] = Field(default=None, description="Anatomical interpretation of gradient concentration")
    disclaimer: str = Field(
        default="Explainable AI visual saliency maps are research visualizations of CNN gradient flow and do not constitute an anatomical or radiological boundary diagnosis."
    )
    message: Optional[str] = Field(default=None)


class XAIStatusResponse(BaseModel):
    engine: str = "Explainable AI (XAI) Attribution Suite"
    gradcam_ready: bool
    target_architecture: str = "ResNet-50 ImageNet Transfer Learning"
    supported_methods: list[str] = Field(default_factory=lambda: ["Grad-CAM (Layer4 Bottleneck Convolution)"])
    disclaimer: str = "Research prototype for model transparency and interpretability."


@router.get(
    "/status",
    response_model=XAIStatusResponse,
    status_code=status.HTTP_200_OK,
    summary="Get Explainable AI Engine Status",
)
async def get_xai_status():
    ready = xai_service.is_ready()
    return XAIStatusResponse(
        gradcam_ready=ready,
    )


@router.post(
    "/gradcam",
    response_model=GradCAMResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate Grad-CAM Saliency Map for Chest Radiograph",
    description="Calculates gradient-weighted class activation mapping (Grad-CAM) on the trained ResNet-50 model, returning visual attribution overlay and heatmap.",
)
async def generate_gradcam_endpoint(
    file: UploadFile = File(..., description="Chest X-ray image file (.png, .jpg, .jpeg, .dcm)")
):
    if not file or not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No radiograph file was provided.",
        )

    try:
        file_bytes = await file.read()
    except Exception as exc:
        logger.error(f"Failed to read image bytes: {exc}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unable to read file: {str(exc)}",
        )

    result = xai_service.generate_chest_xray_gradcam(file_bytes, file.filename)

    if result.get("status") == "error":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=result.get("message", "Grad-CAM computation failed."),
        )

    # Persist Grad-CAM attribution telemetry in MongoDB Atlas
    if result.get("status") == "success":
        try:
            from app.db.mongodb import save_prediction_record
            record = {
                "modality": "chest_xray",
                "model_name": "Chest X-ray ResNet50 Classifier (Grad-CAM XAI)",
                "filename": file.filename,
                "file_type": "standard_image",
                "prediction": result.get("prediction"),
                "confidence": result.get("confidence"),
                "threshold": 0.5,
                "status": "success",
                "disclaimer": result.get("disclaimer"),
                "image_preview": result.get("original_preview"),
                "gradcam_preview": result.get("overlay_preview"),
                "heatmap_preview": result.get("heatmap_preview"),
                "explanation": result.get("explanation"),
            }
            inserted_id = await save_prediction_record(record)
            if inserted_id:
                result["id"] = inserted_id
        except Exception as exc:
            logger.warning(f"Could not persist Grad-CAM analysis to MongoDB: {exc}")

    return GradCAMResponse(**result)
