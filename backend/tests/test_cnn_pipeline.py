import io
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from PIL import Image

from ml.chest_xray.cnn.config import CNNConfig
from ml.chest_xray.cnn.dataset import (
    get_transforms,
    load_and_split_rsna_metadata,
    read_dicom_or_image,
)
from ml.chest_xray.cnn.predict import ChestXRayInferenceService


@pytest.fixture
def sample_png_bytes(tmp_path: Path):
    """Creates a temporary synthetic 512x512 grayscale chest radiograph PNG."""
    img_array = (np.random.rand(512, 512) * 255).astype(np.uint8)
    img = Image.fromarray(img_array, mode="L")
    img_path = tmp_path / "synthetic_xray.png"
    img.save(img_path)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return img_path, buf.getvalue()


def test_read_image_and_transforms(sample_png_bytes):
    img_path, png_bytes = sample_png_bytes
    pil_img = read_dicom_or_image(img_path)
    assert pil_img.mode == "RGB"
    assert pil_img.size == (512, 512)

    transform = get_transforms(img_size=(224, 224), is_train=False)
    if transform is not None:
        tensor = transform(pil_img)
        assert tensor.shape == (3, 224, 224)


def test_metadata_grouping_and_patient_split(tmp_path: Path):
    """
    Verifies that patients with multiple bounding box rows are collapsed to
    a single unique patient record, and that patient-level split has 0% leakage.
    """
    data = {
        "patientId": [
            "patient-A",
            "patient-A",  # duplicate row (2nd bounding box)
            "patient-B",
            "patient-C",
            "patient-C",  # duplicate row
            "patient-D",
            "patient-E",
            "patient-F",
            "patient-G",
            "patient-H",
        ],
        "x": [100, 200, None, 150, 250, None, 50, None, 80, None],
        "y": [100, 200, None, 150, 250, None, 50, None, 80, None],
        "width": [50, 60, None, 70, 80, None, 90, None, 40, None],
        "height": [50, 60, None, 70, 80, None, 90, None, 40, None],
        "Target": [1, 1, 0, 1, 1, 0, 1, 0, 1, 0],
    }
    csv_path = tmp_path / "mock_labels.csv"
    pd.DataFrame(data).to_csv(csv_path, index=False)

    train_df, val_df, test_df, pos_weight = load_and_split_rsna_metadata(
        csv_path, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, seed=42
    )

    # Unique patients: A, B, C, D, E, F, G, H (8 total)
    total_split_patients = len(train_df) + len(val_df) + len(test_df)
    assert total_split_patients == 8

    # Verify zero leakage across splits
    train_patients = set(train_df["patientId"])
    val_patients = set(val_df["patientId"])
    test_patients = set(test_df["patientId"])

    assert len(train_patients.intersection(val_patients)) == 0
    assert len(train_patients.intersection(test_patients)) == 0
    assert len(val_patients.intersection(test_patients)) == 0
    assert pos_weight > 0


def test_missing_checkpoint_returns_not_trained(tmp_path: Path, sample_png_bytes):
    """
    Guarantees that an absent checkpoint safely returns 'not_trained'
    and NEVER fabricates mock predictions.
    """
    _, png_bytes = sample_png_bytes
    fake_config = CNNConfig(
        artifacts_dir=tmp_path / "non_existent_artifacts",
        checkpoint_filename="absent_model.pth",
    )
    service = ChestXRayInferenceService(config=fake_config)

    assert service.is_model_trained() is False

    result = service.predict(png_bytes, "chest_scan.png")
    assert result["status"] == "not_trained"
    assert result["prediction"] is None
    assert result["confidence"] is None
    assert "not found" in result["message"].lower() or "not trained" in result["message"].lower()
    assert "Academic research prototype" in result["disclaimer"]


def test_upload_validation_rejects_invalid_inputs(sample_png_bytes):
    _, png_bytes = sample_png_bytes
    service = ChestXRayInferenceService()

    # Empty payload
    is_valid, _, msg = service.validate_upload(b"", "empty.png")
    assert is_valid is False
    assert "empty" in msg.lower()

    # Unsupported file extension
    is_valid, _, msg = service.validate_upload(b"fake text content", "notes.txt")
    assert is_valid is False
    assert "unsupported" in msg.lower()

    # Valid PNG
    is_valid, ftype, _ = service.validate_upload(png_bytes, "scan.png")
    assert is_valid is True
    assert ftype == "standard_image"


def test_chest_xray_service_startup_and_inference(sample_png_bytes):
    """
    Tests that ChestXRayService loads the trained checkpoint and executes
    valid inference without fabricating scores.
    """
    from app.services.chest_xray import ChestXRayService
    _, png_bytes = sample_png_bytes

    service = ChestXRayService()
    loaded = service.load_model()

    if service.is_model_trained():
        assert loaded is True
        assert service.is_ready() is True

        result = service.predict(png_bytes, "synthetic_test.png")
        assert result["status"] == "success"
        assert result["prediction"] in ["Lung Opacity / Pneumonia", "Normal / No Lung Opacity"]
        assert result["predicted_class_id"] in [0, 1]
        assert isinstance(result["confidence"], float)
        assert 0.0 <= result["confidence"] <= 1.0
        assert "Academic research prototype" in result["disclaimer"]
    else:
        assert service.is_ready() is False
        result = service.predict(png_bytes, "synthetic_test.png")
        assert result["status"] == "not_trained"
        assert result["prediction"] is None


def test_cpu_inference_reliability(sample_png_bytes):
    """
    Guarantees that ChestXRayService reliably performs CPU inference
    when running on CPU device.
    """
    from app.services.chest_xray import ChestXRayService
    _, png_bytes = sample_png_bytes

    service = ChestXRayService()
    service.device = "cpu"
    loaded = service.load_model()

    if service.is_model_trained():
        assert loaded is True
        assert service.is_ready() is True
        assert service.device == "cpu"

        result = service.predict(png_bytes, "cpu_test.png")
        assert result["status"] == "success"
        assert isinstance(result["confidence"], float)
        assert 0.0 <= result["confidence"] <= 1.0


def test_corrupt_upload_rejected():
    """
    Guarantees that corrupted or truncated image bytes are rejected
    with descriptive error messages.
    """
    from app.services.chest_xray import ChestXRayService

    service = ChestXRayService()
    # Random unparseable bytes with .png extension
    is_valid, ftype, msg = service.validate_upload(b"NOT_A_VALID_PNG_DATA", "corrupted.png")
    assert is_valid is False
    assert "corrupted" in msg.lower() or "invalid" in msg.lower()

    # Random unparseable bytes with .dcm extension
    is_valid, ftype, msg = service.validate_upload(b"NOT_A_VALID_DICOM_DATA", "corrupted.dcm")
    assert is_valid is False
    assert "corrupted" in msg.lower() or "invalid" in msg.lower()


