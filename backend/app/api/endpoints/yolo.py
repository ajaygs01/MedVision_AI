"""
FastAPI router for YOLO Anatomical Lesion Localization endpoints.
"""

import logging
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status

from app.schemas.yolo import (
    YOLODetectionResponse,
    YOLODetectionStatusResponse,
)
from app.services.yolo import yolo_service

logger = logging.getLogger("medivision.api.yolo")
router = APIRouter(prefix="/yolo", tags=["YOLO Anatomical Localization"])


@router.get(
    "/status",
    response_model=YOLODetectionStatusResponse,
    status_code=status.HTTP_200_OK,
    summary="Get YOLO Module Status & Weights State",
    description="Returns current status of YOLO detection architecture, checkpoint presence, and genuine evaluation metrics.",
)
async def get_yolo_status():
    stat = yolo_service.get_status()
    return YOLODetectionStatusResponse(**stat)


@router.post(
    "/detect",
    response_model=YOLODetectionResponse,
    status_code=status.HTTP_200_OK,
    summary="Detect Anatomical Lesions and Bounding Boxes",
    description="Accepts chest radiographs (.dcm, .png, .jpg, .jpeg) and outputs localized bounding boxes with confidence scores. If model weights are missing, safely returns not_trained status.",
)
async def detect_lesions(
    file: UploadFile = File(..., description="Radiograph image file (.dcm, .png, .jpg, .jpeg)"),
    conf_threshold: float = Form(0.25, ge=0.01, le=1.0, description="Confidence threshold"),
    iou_threshold: float = Form(0.45, ge=0.01, le=1.0, description="NMS IoU threshold"),
):
    if not file or not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No image file provided in upload request.",
        )

    try:
        file_bytes = await file.read()
    except Exception as exc:
        logger.error(f"Failed to read uploaded file: {exc}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unable to read uploaded file: {exc}",
        )

    result = yolo_service.detect(
        file_bytes=file_bytes,
        filename=file.filename,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
    )

    if result.get("status") == "error":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=result.get("message", "Invalid image file or parameters."),
        )

    # Persist analysis telemetry to MongoDB if connected
    persisted = False
    analysis_id = None
    try:
        from app.db.mongodb import save_prediction_record
        record = {
            "modality": "yolo_detection",
            "model_name": "YOLOv8 Anatomical Lesion Localization",
            "filename": file.filename,
            "file_type": "dicom" if file.filename.lower().endswith(".dcm") else "standard_image",
            "prediction": f"{result.get('num_detections', 0)} Findings",
            "status": result.get("status"),
            "num_detections": result.get("num_detections", 0),
            "detections": result.get("detections", []),
            "conf_threshold": conf_threshold,
            "threshold": conf_threshold,
            "iou_threshold": iou_threshold,
            "disclaimer": result.get("disclaimer"),
            "image_preview": result.get("preview_image"),
        }
        inserted_id = await save_prediction_record(record)
        if inserted_id:
            persisted = True
            analysis_id = inserted_id
    except Exception as exc:
        logger.warning(f"Failed to persist YOLO analysis to MongoDB: {exc}")

    result["persisted"] = persisted
    result["analysis_id"] = analysis_id
    result["id"] = analysis_id

    return YOLODetectionResponse(**result)
