import base64
import io
import pytest
from PIL import Image as PILImage
from fastapi.testclient import TestClient

from app.main import app
from app.services.reports import generate_analysis_pdf


def _create_sample_base64_image(width=100, height=100, color=(128, 128, 128)) -> str:
    img = PILImage.new("RGB", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"


def test_generate_pdf_chest_xray_with_images():
    sample_img_b64 = _create_sample_base64_image(128, 128, (60, 60, 60))
    sample_gradcam_b64 = _create_sample_base64_image(128, 128, (200, 50, 50))

    record_data = {
        "modality": "chest_xray",
        "model_name": "Chest X-ray ResNet50 Classifier",
        "filename": "patient_sample_001.png",
        "timestamp": "2026-10-21T12:00:00Z",
        "prediction": "Lung Opacity / Pneumonia",
        "predicted_class_id": 1,
        "confidence": 0.8842,
        "threshold": 0.5,
        "status": "success",
        "image_preview": sample_img_b64,
        "gradcam_preview": sample_gradcam_b64,
        "disclaimer": "Academic research prototype. Results are not clinically validated.",
    }

    pdf_bytes = generate_analysis_pdf("test_rec_cxr_01", record_data)
    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 1000
    # Standard PDF magic byte signature
    assert pdf_bytes.startswith(b"%PDF-")


def test_generate_pdf_chest_xray_without_images():
    record_data = {
        "modality": "chest_xray",
        "model_name": "Chest X-ray ResNet50 Classifier",
        "filename": "study_no_preview.png",
        "timestamp": "2026-10-21T12:00:00Z",
        "prediction": "Normal",
        "predicted_class_id": 0,
        "confidence": 0.125,
        "threshold": 0.5,
        "status": "success",
        "image_preview": None,
        "gradcam_preview": None,
        "disclaimer": "Academic research prototype.",
    }

    pdf_bytes = generate_analysis_pdf("test_rec_cxr_02", record_data)
    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 1000
    assert pdf_bytes.startswith(b"%PDF-")


def test_generate_pdf_clinical_ml():
    record_data = {
        "modality": "clinical_ml",
        "model_name": "Random Forest Cardiovascular Risk Predictor",
        "filename": "patient_CLEVELAND_ELEVATED_02.json",
        "timestamp": "2026-10-21T12:30:00Z",
        "prediction": "Elevated Cardiovascular Risk Detected",
        "predicted_class_id": 1,
        "confidence": 0.8267,
        "risk_tier": "Elevated Risk",
        "risk_probability": 0.8267,
        "status": "success",
        "top_contributing_factors": [
            "Fluoroscopy Vessels (ca=3 colored): Strong anatomical predictor of coronary calcification",
            "Thallium Defect (thal=7.0): Stress scintigraphy indicates reversible ischemia",
            "ST Depression (oldpeak=2.6mm): Significant exercise-induced ischemic marker",
        ],
        "patient_data": {
            "age": 65,
            "sex": 1,
            "cp": 4,
            "trestbps": 160,
            "chol": 286,
            "fbs": 1,
            "restecg": 2,
            "thalach": 108,
            "exang": 1,
            "oldpeak": 2.6,
            "slope": 2,
            "ca": 3,
            "thal": 7,
            "patient_id": "CLEVELAND_ELEVATED_02",
        },
        "disclaimer": "Academic research prototype. Results are not clinically validated.",
    }

    pdf_bytes = generate_analysis_pdf("test_rec_clin_01", record_data)
    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 1000
    assert pdf_bytes.startswith(b"%PDF-")


def test_report_endpoint_not_found():
    client = TestClient(app)
    # Using an invalid / non-existent ObjectId
    response = client.get("/api/imaging/history/000000000000000000000000/report")
    # Should cleanly return 404 or 503 if db not reachable, not crash
    assert response.status_code in (404, 503)
    if response.status_code == 404:
        assert "not found" in response.json()["detail"].lower()


def test_direct_generate_pdf_report_endpoint():
    client = TestClient(app)
    payload = {
        "modality": "chest_xray",
        "model_name": "Chest X-ray ResNet50 Classifier",
        "filename": "test_direct_image.png",
        "prediction": "Normal / No Lung Opacity",
        "confidence": 0.25,
        "threshold": 0.5,
        "view_mode": "split",
        "explanation": "Test gradient concentration",
    }
    response = client.post("/api/reports/generate", json=payload)
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/pdf"
    assert response.content.startswith(b"%PDF-")
    assert len(response.content) > 1000

