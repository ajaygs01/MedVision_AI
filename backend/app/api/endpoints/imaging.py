import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any
from fastapi import APIRouter, File, HTTPException, UploadFile, status

# Ensure ml package is discoverable
project_root = Path(__file__).resolve().parents[4]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from app.schemas.imaging import (
    ChestXRayClassifyResponse,
    PredictionHistoryItem,
    PredictionHistoryResponse,
)
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

    # Record analysis telemetry in MongoDB Atlas (non-blocking / safe fallback)
    try:
        from app.db.mongodb import save_prediction_record
        record = {
            "modality": "chest_xray",
            "model_name": result.get("model_name", "Chest X-ray ResNet50 Classifier"),
            "filename": file.filename,
            "file_type": result.get("file_type", "standard_image"),
            "prediction": result.get("prediction"),
            "predicted_class_id": result.get("predicted_class_id"),
            "confidence": result.get("confidence"),
            "threshold": result.get("threshold", 0.5),
            "status": result.get("status", "success"),
            "disclaimer": result.get("disclaimer"),
            "image_preview": result.get("image_preview"),
            "gradcam_preview": result.get("gradcam_preview"),
        }
        inserted_id = await save_prediction_record(record)
        if inserted_id:
            result["id"] = inserted_id
    except Exception as exc:
        logger.warning(f"Could not persist prediction history: {exc}")

    return ChestXRayClassifyResponse(**result)


@router.get(
    "/history",
    response_model=PredictionHistoryResponse,
    status_code=status.HTTP_200_OK,
    summary="Retrieve Analysis & Prediction History",
    description="Retrieves chronological case history stored in MongoDB Atlas with graceful empty fallback.",
)
@router.get(
    "/chest-xray/history",
    response_model=PredictionHistoryResponse,
    status_code=status.HTTP_200_OK,
    include_in_schema=False,
)
async def get_imaging_history(limit: int = 50, skip: int = 0):
    from app.db.mongodb import check_mongo_connection, get_prediction_history
    is_connected, _ = await check_mongo_connection()
    records = []
    if is_connected:
        raw_records = await get_prediction_history(limit=limit, skip=skip)
        records = [
            PredictionHistoryItem(
                id=str(r.get("id") or r.get("_id") or ""),
                timestamp=str(r.get("timestamp") or ""),
                modality=str(r.get("modality") or "chest_xray"),
                model_name=str(r.get("model_name") or ""),
                filename=str(r.get("filename") or ""),
                file_type=str(r.get("file_type") or "standard_image"),
                prediction=r.get("prediction"),
                predicted_class_id=r.get("predicted_class_id"),
                confidence=r.get("confidence"),
                threshold=float(r.get("threshold", 0.5)),
                status=str(r.get("status") or "success"),
                disclaimer=r.get("disclaimer"),
                image_preview=r.get("image_preview"),
                gradcam_preview=r.get("gradcam_preview"),
                heatmap_preview=r.get("heatmap_preview"),
                risk_tier=r.get("risk_tier"),
                risk_probability=r.get("risk_probability"),
                top_contributing_factors=r.get("top_contributing_factors"),
                patient_data=r.get("patient_data"),
                volumetric_metrics=r.get("volumetric_metrics"),
                slices=r.get("slices"),
                detections=r.get("detections"),
                num_detections=r.get("num_detections"),
            )
            for r in raw_records
        ]
    return PredictionHistoryResponse(
        total=len(records),
        records=records,
        database_connected=is_connected,
    )


@router.get(
    "/history/{record_id}",
    response_model=Optional[PredictionHistoryItem],
    status_code=status.HTTP_200_OK,
    summary="Retrieve Specific Analysis Record",
    description="Retrieves a specific case history record from MongoDB Atlas by its unique document ID.",
)
async def get_history_record_by_id(record_id: str):
    from app.db.mongodb import get_mongo_client
    from bson import ObjectId
    client = get_mongo_client()
    if client is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Database not configured")

    from app.core.config import get_settings
    settings = get_settings()
    db = client[settings.mongodb_db_name]
    try:
        doc = None
        if ObjectId.is_valid(record_id):
            doc = await db["analysis_history"].find_one({"_id": ObjectId(record_id)})
        if not doc:
            doc = await db["analysis_history"].find_one({"id": record_id})
        if not doc:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Record not found")
        return PredictionHistoryItem(
            id=str(doc.get("_id") or doc.get("id") or record_id),
            timestamp=str(doc.get("timestamp") or ""),
            modality=str(doc.get("modality") or "chest_xray"),
            model_name=str(doc.get("model_name") or ""),
            filename=str(doc.get("filename") or ""),
            file_type=str(doc.get("file_type") or "standard_image"),
            prediction=doc.get("prediction"),
            predicted_class_id=doc.get("predicted_class_id"),
            confidence=doc.get("confidence"),
            threshold=float(doc.get("threshold", 0.5)),
            status=str(doc.get("status") or "success"),
            disclaimer=doc.get("disclaimer"),
            image_preview=doc.get("image_preview"),
            gradcam_preview=doc.get("gradcam_preview"),
            heatmap_preview=doc.get("heatmap_preview"),
            risk_tier=doc.get("risk_tier"),
            risk_probability=doc.get("risk_probability"),
            top_contributing_factors=doc.get("top_contributing_factors"),
            patient_data=doc.get("patient_data"),
            volumetric_metrics=doc.get("volumetric_metrics"),
            slices=doc.get("slices"),
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid record request: {exc}")


@router.get(
    "/history/{record_id}/report",
    summary="Download Official Clinical Analysis PDF Report",
    description="Generates and streams a formal publication-grade PDF research audit report for the specified analysis record.",
)
async def get_history_pdf_report(record_id: str):
    from fastapi.responses import Response
    from app.db.mongodb import get_mongo_client
    from bson import ObjectId
    from app.services.reports import generate_analysis_pdf

    client = get_mongo_client()
    if client is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Database connectivity unavailable.")

    from app.core.config import get_settings
    settings = get_settings()
    db = client[settings.mongodb_db_name]

    doc = None
    try:
        if ObjectId.is_valid(record_id):
            doc = await db["analysis_history"].find_one({"_id": ObjectId(record_id)})
    except Exception:
        pass

    if not doc:
        try:
            doc = await db["analysis_history"].find_one({"id": record_id})
        except Exception:
            pass

    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Analysis record '{record_id}' was not found in database.",
        )

    try:
        pdf_bytes = generate_analysis_pdf(record_id, doc)
    except Exception as exc:
        logger.error(f"Failed to generate analysis PDF report for {record_id}: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to synthesize PDF report: {str(exc)}",
        )

    filename = f"MediVision_Report_{record_id}.pdf"
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Access-Control-Expose-Headers": "Content-Disposition",
        },
    )


@router.post(
    "/reports/generate",
    summary="Generate PDF Clinical Report Directly from Analysis Payload",
    description="Directly synthesizes an official MediVision PDF clinical report from active analysis telemetry.",
)
async def generate_direct_pdf_report_endpoint(payload: Dict[str, Any]):
    import uuid
    from fastapi.responses import Response
    from app.services.reports import generate_analysis_pdf

    record_id = str(payload.get("id") or payload.get("record_id") or f"LIVE_{uuid.uuid4().hex[:12]}")
    try:
        pdf_bytes = generate_analysis_pdf(record_id, payload)
    except Exception as exc:
        logger.error(f"Failed to generate direct analysis PDF report: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to synthesize PDF report: {str(exc)}",
        )

    filename = f"MediVision_Report_{record_id}.pdf"
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Access-Control-Expose-Headers": "Content-Disposition",
        },
    )

