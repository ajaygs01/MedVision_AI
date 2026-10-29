import pytest
import torch
import pandas as pd
from pathlib import Path
from httpx import ASGITransport, AsyncClient

from app.main import app
from app.services.skin_lesion import skin_lesion_service
from ml.skin_lesion.config import (
    SkinLesionConfig,
    HAM10000_CLASSES,
    CLASS_TO_IDX,
    IDX_TO_CLASS,
)
from ml.skin_lesion.model import build_skin_lesion_model
from ml.skin_lesion.dataset import split_lesions_zero_leakage, validate_dataset_presence


def test_skin_lesion_config_classes():
    assert len(HAM10000_CLASSES) == 7
    assert "MEL" in HAM10000_CLASSES
    assert "NV" in HAM10000_CLASSES
    assert "BCC" in HAM10000_CLASSES
    assert CLASS_TO_IDX["MEL"] == IDX_TO_CLASS.index("MEL")


def test_skin_lesion_model_forward_pass_shape():
    model = build_skin_lesion_model(num_classes=7, pretrained=False)
    model.eval()
    dummy_input = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        output = model(dummy_input)
    assert output.shape == (2, 7), f"Expected shape (2, 7), got {output.shape}"


def test_skin_lesion_dataset_split_zero_leakage():
    # Synthetic metadata with 10 lesions across 20 images
    records = []
    for lesion_idx in range(10):
        lesion_id = f"HAM_LESION_{lesion_idx:03d}"
        dx = HAM10000_CLASSES[lesion_idx % 7]
        # 2 images per lesion
        for img_sub in range(2):
            records.append({
                "lesion_id": lesion_id,
                "image_id": f"{lesion_id}_img{img_sub}",
                "dx": dx,
            })
    df = pd.DataFrame(records)

    train_df, val_df = split_lesions_zero_leakage(df, val_ratio=0.2, seed=42)

    train_lesions = set(train_df["lesion_id"].unique())
    val_lesions = set(val_df["lesion_id"].unique())

    # Zero leakage assertion: Intersection must be strictly empty
    overlap = train_lesions.intersection(val_lesions)
    assert len(overlap) == 0, f"Found leaking lesions across train and val: {overlap}"
    assert len(train_df) + len(val_df) == len(df)


def test_skin_lesion_dataset_validation_missing(tmp_path):
    empty_dir = tmp_path / "empty_dataset"
    empty_dir.mkdir()
    cfg = SkinLesionConfig(dataset_dir=empty_dir)
    status = validate_dataset_presence(cfg)
    assert status["available"] is False
    assert "missing" in status["message"].lower()


def test_skin_lesion_service_unconfigured_when_no_weights(tmp_path):
    # Ensure fresh state
    non_existent = tmp_path / "non_existent_weights.pth"
    service = skin_lesion_service
    # Service checkpoint defaults to non-existent file
    assert service.is_ready() is False
    status = service.get_status()
    assert status["trained"] is False
    assert status["status"] == "unconfigured"


def test_skin_lesion_service_safe_not_trained_inference():
    dummy_bytes = b"fake-image-bytes"
    result = skin_lesion_service.predict(dummy_bytes, "test.jpg")
    assert result["status"] == "not_trained"
    assert result["prediction"] is None
    assert result["confidence"] is None
    assert "not configured" in result["message"].lower()


@pytest.mark.asyncio
async def test_skin_lesion_models_status_api_endpoint():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.get("/api/models/status")
    assert response.status_code == 200
    data = response.json()
    assert "cnn_skin_lesion" in data["models"]
    model_info = data["models"]["cnn_skin_lesion"]
    assert model_info["trained"] is False
    assert model_info["status"] == "unconfigured"
    assert model_info["weights_path"] is None


@pytest.mark.asyncio
async def test_skin_lesion_dedicated_status_endpoint():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.get("/api/skin-lesion/status")
    assert response.status_code == 200
    data = response.json()
    assert data["trained"] is False
    assert data["status"] == "unconfigured"
    assert data["name"] == "Skin Lesion Multi-Class Classifier"


@pytest.mark.asyncio
async def test_skin_lesion_classify_endpoint_untrained():
    import io
    from PIL import Image

    # Create dummy 100x100 JPEG
    img = Image.new("RGB", (100, 100), color=(200, 100, 80))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)

    files = {"file": ("test_lesion.jpg", buf, "image/jpeg")}
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.post("/api/skin-lesion/classify", files=files)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "not_trained"
    assert data["prediction"] is None
    assert data["confidence"] is None
    assert "not configured" in data["message"].lower()


@pytest.mark.asyncio
async def test_skin_lesion_classify_invalid_file_type():
    files = {"file": ("invalid_notes.txt", b"plain text content", "text/plain")}
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.post("/api/skin-lesion/classify", files=files)
    assert response.status_code == 400
    assert "unsupported file format" in response.json()["detail"].lower()
