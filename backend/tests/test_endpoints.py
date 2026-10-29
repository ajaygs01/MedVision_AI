import pytest
from httpx import ASGITransport, AsyncClient
from app.main import app


@pytest.mark.asyncio
async def test_root_endpoint():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["stage"] == 1
    assert data["status"] == "online"
    assert "MediVision" in data["name"]


@pytest.mark.asyncio
async def test_health_endpoint():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert data["version"] == "1.0.0-stage1"
    assert "database" in data
    assert data["database"]["engine"] in ["MongoDB Atlas", "Local MongoDB"]
    assert data["database"]["status"] in ["connected", "disconnected"]


@pytest.mark.asyncio
async def test_models_status_endpoint():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.get("/api/models/status")
    assert response.status_code == 200
    data = response.json()
    assert data["stage"] in [1, 2]
    assert data["system_status"] in ["foundation_ready", "stage2_pipeline_active", "stage_2_cnn_pipeline_ready"]
    assert data["active_trained_count"] in [0, 1, 2, 3, 4, 5]
    assert "academic" in data["disclaimer"].lower() or "stage" in data["disclaimer"].lower()

    models = data["models"]
    expected_keys = [
        "cnn_chest_xray",
        "unet_brain_mri",
        "cnn_skin_lesion",
        "yolo_detection",
        "clinical_ml",
        "xai_engine",
        "rag_assistant",
    ]
    for key in expected_keys:
        assert key in models, f"Model key '{key}' missing from models status"
        model_info = models[key]
        if key in ["cnn_chest_xray", "clinical_ml", "xai_engine", "yolo_detection", "cnn_skin_lesion"]:
            assert model_info["trained"] in [True, False]
            assert model_info["status"] in ["ready", "unconfigured", "not_trained"]
        elif key == "unet_brain_mri":
            assert model_info["trained"] in [True, False]
            assert model_info["status"] in ["ready", "dataset_missing", "not_trained", "unconfigured"]
        else:
            assert model_info["trained"] is False, f"Model '{key}' cannot be marked as trained"
            assert model_info["status"] == "unconfigured", f"Model '{key}' must be unconfigured"
            assert model_info["weights_path"] is None, f"Model '{key}' cannot have active weights"
