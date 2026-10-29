import logging
from fastapi import APIRouter, HTTPException, status

from app.schemas.clinical_ml import (
    ClinicalModelInfoResponse,
    ClinicalPatientInput,
    ClinicalRiskResponse,
)
from app.services.clinical_ml import clinical_ml_service

logger = logging.getLogger("medivision.api.clinical_ml")
router = APIRouter(prefix="/clinical-ml", tags=["Clinical Tabular Machine Learning"])


@router.post(
    "/predict",
    response_model=ClinicalRiskResponse,
    status_code=status.HTTP_200_OK,
    summary="Predict Cardiovascular Patient Risk",
    description="Evaluates 13 standardized clinical physiological attributes using trained Scikit-Learn Random Forest ensemble. Persists telemetry in MongoDB Atlas.",
)
async def predict_patient_risk(patient: ClinicalPatientInput):
    patient_dict = patient.model_dump()
    result = clinical_ml_service.predict_risk(patient_dict)

    if result.get("status") == "error":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=result.get("message", "Clinical ML evaluation failed."),
        )

    # Persist clinical assessment in MongoDB Atlas / Local MongoDB
    try:
        from app.db.mongodb import save_prediction_record
        record = {
            "modality": "clinical_ml",
            "model_name": result.get("model_name", "Random Forest Cardiovascular Risk Predictor"),
            "filename": f"patient_{patient.patient_id or 'anonymous'}.json",
            "file_type": "tabular_ehr",
            "prediction": result.get("prediction"),
            "predicted_class_id": result.get("predicted_class_id"),
            "confidence": result.get("risk_probability"),
            "threshold": 0.5,
            "status": result.get("status", "success"),
            "disclaimer": result.get("disclaimer"),
            "image_preview": None,
            "risk_tier": result.get("risk_tier"),
            "risk_probability": result.get("risk_probability"),
            "top_contributing_factors": result.get("top_contributing_factors", []),
            "patient_data": patient_dict,
        }
        inserted_id = await save_prediction_record(record)
        if inserted_id:
            result["id"] = inserted_id
    except Exception as exc:
        logger.warning(f"Could not persist clinical ML history: {exc}")

    return ClinicalRiskResponse(**result)


@router.get(
    "/model-info",
    response_model=ClinicalModelInfoResponse,
    status_code=status.HTTP_200_OK,
    summary="Retrieve Clinical ML Model Benchmark Metrics",
    description="Returns cross-validation and held-out test split evaluation metrics (Accuracy, ROC-AUC, Sensitivity, Specificity, Confusion Matrix, and Top Features).",
)
async def get_clinical_model_info():
    info = clinical_ml_service.get_model_info()
    return ClinicalModelInfoResponse(**info)
