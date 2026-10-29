import logging
from typing import Optional
from fastapi import APIRouter, File, UploadFile, HTTPException, status
from app.schemas.skin_lesion import SkinLesionClassifyResponse
from app.services.skin_lesion import skin_lesion_service

logger = logging.getLogger("medivision.api.skin_lesion")
router = APIRouter(prefix="/skin-lesion", tags=["Skin Lesion Analysis"])


@router.get("/status", summary="Get Skin Lesion Model Status")
async def get_skin_lesion_status():
    """
    Returns live operational status and checkpoint telemetry for Skin Lesion Analysis.
    """
    return skin_lesion_service.get_status()


@router.post(
    "/classify",
    response_model=SkinLesionClassifyResponse,
    status_code=status.HTTP_200_OK,
    summary="Classify Dermoscopy Epiluminescence Image",
    description="Evaluates uploaded skin lesion photograph using the multi-class EfficientNet-B0 classifier. If weights are unconfigured, cleanly returns status='not_trained'.",
)
async def classify_skin_lesion(
    file: UploadFile = File(..., description="Dermoscopy image file (.jpg, .jpeg, .png)")
):
    if not file or not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No dermoscopy image was provided in the upload request.",
        )

    # Validate file extension
    ext = file.filename.split(".")[-1].lower() if "." in file.filename else ""
    if ext not in ["jpg", "jpeg", "png", "bmp"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format '{ext}'. Accepted formats: JPEG, PNG, BMP.",
        )

    try:
        file_bytes = await file.read()
        if len(file_bytes) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded image file is empty (0 bytes).",
            )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Failed to read uploaded skin lesion file: {exc}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unable to read uploaded file: {str(exc)}",
        )

    result = skin_lesion_service.predict(file_bytes, file.filename)

    if result.get("status") == "error":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=result.get("message", "Invalid dermoscopy image."),
        )

    # Persist case telemetry in MongoDB Atlas (non-blocking fallback)
    try:
        from app.db.mongodb import save_prediction_record
        record = {
            "modality": "skin_lesion",
            "model_name": result.get("model_name", "Skin Lesion Multi-Class Classifier"),
            "filename": file.filename,
            "file_type": result.get("file_type", "standard_image"),
            "prediction": result.get("prediction"),
            "predicted_class_id": result.get("predicted_class_id"),
            "confidence": result.get("confidence"),
            "threshold": 0.5,
            "status": result.get("status", "success"),
            "disclaimer": result.get("disclaimer"),
            "image_preview": result.get("image_preview"),
        }
        inserted_id = await save_prediction_record(record)
        if inserted_id:
            result["id"] = inserted_id
    except Exception as exc:
        logger.warning(f"Could not persist skin lesion prediction telemetry: {exc}")

    return SkinLesionClassifyResponse(**result)
