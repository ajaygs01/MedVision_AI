"""
Backend integration tests for Brain MRI FastAPI endpoints, validation, and PDF reporting.
"""

import io
import pytest
from fastapi.testclient import TestClient
import numpy as np
import nibabel as nib

from app.main import create_application
from app.services.reports import generate_analysis_pdf


@pytest.fixture
def client():
    app = create_application()
    return TestClient(app)


def create_dummy_nifti_bytes(shape=(16, 32, 32)) -> bytes:
    """Helper creating an in-memory NIfTI .nii.gz file."""
    data = np.random.randn(*shape).astype(np.float32)
    affine = np.eye(4)
    img = nib.Nifti1Image(data, affine)
    buf = io.BytesIO()
    # Save using nibabel file_map or temporary write
    with io.BytesIO() as bio:
        file_map = img.make_file_map({'image': bio, 'header': bio})
        img.to_file_map(file_map)
        return bio.getvalue()


def test_brain_mri_status_endpoint(client):
    """Verify /api/brain-mri/status returns proper schema and transparent operational status."""
    resp = client.get("/api/brain-mri/status")
    assert resp.status_code == 200
    data = resp.json()
    assert "status" in data
    assert data["status"] in ["ready", "not_trained", "dataset_missing"]
    assert "model_name" in data
    assert "architecture" in data
    assert "dataset_status" in data
    assert "disclaimer" in data


def test_brain_mri_validate_invalid_extension(client):
    """Verify validator rejects non-NIfTI / non-ZIP file types."""
    files = {"file": ("test.txt", b"plain text content", "text/plain")}
    resp = client.post("/api/brain-mri/validate", files=files)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "invalid"
    assert "Unsupported file format" in data["message"]


def test_brain_mri_validate_empty_file(client):
    """Verify validator rejects 0-byte upload."""
    files = {"file": ("empty.nii.gz", b"", "application/octet-stream")}
    resp = client.post("/api/brain-mri/validate", files=files)
    assert resp.status_code == 400


def test_brain_mri_segment_not_trained_response(client):
    """Verify segment endpoint handles unconfigured/untrained state transparently."""
    dummy_nii = create_dummy_nifti_bytes()
    files = {"file": ("patient_test_flair.nii.gz", dummy_nii, "application/octet-stream")}
    resp = client.post("/api/brain-mri/segment", files=files)
    assert resp.status_code == 200
    data = resp.json()
    assert "status" in data
    assert data["status"] in ["success", "not_trained"]
    if data["status"] == "not_trained":
        assert "not been trained yet" in data["message"]
        assert "train.py" in data["message"]


def test_generate_pdf_report_brain_mri():
    """Verify PDF generator synthesizes a valid Brain MRI report with volumetric table."""
    record_data = {
        "modality": "brain_mri",
        "model_name": "3D U-Net Volumetric Brain Tumor Segmenter",
        "filename": "BraTS20_Training_001_study.nii.gz",
        "patient_id": "BraTS20_001",
        "prediction": "Volumetric Glioma/Lesion Delineated (Whole Tumor Volume: 32.4 cm³)",
        "confidence": 0.88,
        "status": "success",
        "volumetric_metrics": {
            "whole_tumor_cm3": 32.4,
            "tumor_core_cm3": 18.2,
            "enhancing_tumor_cm3": 9.5,
            "edema_cm3": 14.2,
            "necrotic_core_cm3": 8.7,
        },
        "slices": {
            "plane": "axial",
            "slice_idx": 78,
            "total_slices": 155,
        },
        "disclaimer": "Academic research prototype. Results are not clinically validated.",
        "timestamp": "2026-10-21T21:00:00Z",
    }

    pdf_bytes = generate_analysis_pdf("TEST_BRAIN_MRI_01", record_data)
    assert isinstance(pdf_bytes, bytes)
    assert pdf_bytes.startswith(b"%PDF-")
    assert len(pdf_bytes) > 2000


def test_brain_mri_status_endpoint_experimental_mode(client, monkeypatch):
    """Verify /api/brain-mri/status reflects experimental prototype status when experimental mode is enabled."""
    from app.services.brain_mri import BrainMRIService
    import app.api.endpoints.brain_mri as brain_mri_ep

    monkeypatch.setenv("BRAIN_MRI_EXPERIMENTAL_MODE", "true")
    exp_service = BrainMRIService()
    monkeypatch.setattr(brain_mri_ep, "brain_mri_service", exp_service)

    resp = client.get("/api/brain-mri/status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ready"
    assert data["stage_phase"] == "experimental_prototype"
    assert data["is_experimental"] is True
    assert "Experimental 10p/3e Prototype" in data["model_name"]
    assert "NOT clinically validated" in data["disclaimer"]
    assert "best_val_dice" in data["custom_metrics"]


def test_generate_pdf_report_brain_mri_experimental():
    """Verify PDF generator synthesizes report with explicit experimental prototype non-clinical label."""
    record_data = {
        "modality": "brain_mri",
        "model_name": "3D U-Net Brain Tumor Segmenter (Experimental 10p/3e Prototype)",
        "filename": "BraTS20_Exp_Study_01.nii.gz",
        "patient_id": "BraTS20_Exp",
        "prediction": "[Experimental 10p/3e Prototype] Volumetric Glioma/Lesion Delineated (Whole Tumor Volume: 28.1 cm³)",
        "confidence": 0.88,
        "status": "success",
        "is_experimental": True,
        "volumetric_metrics": {
            "whole_tumor_cm3": 28.1,
            "tumor_core_cm3": 14.5,
            "enhancing_tumor_cm3": 6.2,
            "edema_cm3": 13.6,
            "necrotic_core_cm3": 8.3,
        },
        "slices": {
            "plane": "axial",
            "slice_idx": 80,
            "total_slices": 155,
        },
        "disclaimer": "Experimental prototype model trained on limited 10-patient cohort for 3 epochs. For investigational evaluation only; NOT clinically validated or production ready.",
        "timestamp": "2026-10-22T10:00:00Z",
    }

    pdf_bytes = generate_analysis_pdf("TEST_EXP_REPORT_01", record_data)
    assert isinstance(pdf_bytes, bytes)
    assert pdf_bytes.startswith(b"%PDF-")
    assert len(pdf_bytes) > 2000


def test_brain_mri_slice_endpoint(client, monkeypatch):
    """Verify GET /api/brain-mri/slice returns accurate multi-planar slice previews."""
    from app.services.brain_mri import brain_mri_service
    # Seed cached volume with synthetic 3D data for testing slice extraction
    brain_mri_service.cached_volume = np.ones((32, 64, 64), dtype=np.float32) * 0.5
    brain_mri_service.cached_mask = np.zeros((32, 64, 64), dtype=np.int64)
    brain_mri_service.cached_mask[16, 20:30, 20:30] = 3

    # 1. Axial
    r_ax = client.get("/api/brain-mri/slice?plane=axial&slice_idx=16")
    assert r_ax.status_code == 200
    ax_json = r_ax.json()
    assert ax_json["plane"] == "axial"
    assert ax_json["slice_idx"] == 16
    assert ax_json["total_slices"] == 32
    assert ax_json["mri_preview"].startswith("data:image/png;base64,")

    # 2. Coronal
    r_cor = client.get("/api/brain-mri/slice?plane=coronal&slice_idx=25")
    assert r_cor.status_code == 200
    cor_json = r_cor.json()
    assert cor_json["plane"] == "coronal"
    assert cor_json["total_slices"] == 64

    # 3. Sagittal
    r_sag = client.get("/api/brain-mri/slice?plane=sagittal&slice_idx=25")
    assert r_sag.status_code == 200
    sag_json = r_sag.json()
    assert sag_json["plane"] == "sagittal"
    assert sag_json["total_slices"] == 64
