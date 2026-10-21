import logging
import sys
from pathlib import Path
from fastapi import APIRouter, File, HTTPException, UploadFile, status

# Ensure ml package is discoverable
project_root = Path(__file__).resolve().parents[4]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from app.schemas.imaging import ChestXRayClassifyResponse
from app.services.chest_xray import chest_xray_service

logger = logging.getLogger("medivision.api.imaging")
router = APIRouter(prefix="/imaging", tags=["Medical Imaging Diagnostics"])


@router.post(
    "/chest-xray/classify",
    response_model=ChestXRayClassifyResponse,
    status_code=status.HTTP_200_OK,
    summary="Classify Chest Radiograph (DICOM or standard image)",
    description="Accepts chest X-ray images in DICOM (.dcm), PNG, or JPEG format. Evaluates using the ResNet-50 CNN. If model weights are unconfigured, returns clean not_trained status.",
)
async def classify_chest_xray(
    file: UploadFile = File(..., description="Chest X-ray image file (.dcm, .png, .jpg, .jpeg)")
):
    if not file or not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No image file was provided in the upload request.",
        )

    try:
        file_bytes = await file.read()
    except Exception as exc:
        logger.error(f"Failed to read uploaded file: {exc}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unable to read uploaded file data: {str(exc)}",
        )

    # Perform inference or safe not_trained check
    result = chest_xray_service.predict(file_bytes, file.filename)

    # If file format/validation error occurred, return 400
    if result.get("status") == "error":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=result.get("message", "Invalid image upload."),
        )

    return ChestXRayClassifyResponse(**result)
