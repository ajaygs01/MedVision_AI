import pytest
from app.db.mongodb import (
    sanitize_mongodb_uri,
    save_prediction_record,
    get_prediction_history,
    check_mongo_connection,
    get_mongo_client,
)
from app.core.config import get_settings
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_sanitize_mongodb_uri():
    # Test stripping angle brackets
    raw_with_brackets = "mongodb+srv://<mock_user>:<mock_password_123>@cluster0.example.mongodb.net/"
    clean = sanitize_mongodb_uri(raw_with_brackets)
    assert "<" not in clean
    assert ">" not in clean
    assert "mock_user:mock_password_123@cluster0" in clean

    # Test already clean URI
    already_clean = "mongodb+srv://user:pwd@cluster0.mongodb.net/?retryWrites=true"
    assert sanitize_mongodb_uri(already_clean) == already_clean

    # Test empty
    assert sanitize_mongodb_uri("") == ""


@pytest.mark.asyncio
async def test_history_unconfigured_fallback():
    settings = get_settings()
    original_uri = settings.mongodb_uri
    settings.mongodb_uri = ""
    try:
        # Saving when unconfigured should return None and not crash
        record_id = await save_prediction_record({
            "modality": "chest_xray",
            "model_name": "Test Model",
            "filename": "test.png",
            "prediction": "NORMAL",
            "confidence": 0.12,
        })
        assert record_id is None

        # Fetching when unconfigured should return []
        history = await get_prediction_history()
        assert history == []
    finally:
        settings.mongodb_uri = original_uri


def test_imaging_history_endpoint_structure():
    response = client.get("/api/imaging/history?limit=10")
    assert response.status_code == 200
    data = response.json()
    assert "total" in data
    assert "records" in data
    assert "database_connected" in data
    assert isinstance(data["records"], list)
    assert isinstance(data["database_connected"], bool)


@pytest.mark.asyncio
async def test_live_mongo_atlas_history_lifecycle():
    settings = get_settings()
    if not settings.mongodb_uri:
        pytest.skip("MONGODB_URI not configured for live Atlas test")

    is_connected, msg = await check_mongo_connection()
    if not is_connected:
        pytest.skip(f"Atlas cluster unreachable in this environment: {msg}")

    test_doc = {
        "modality": "chest_xray",
        "model_name": "ResNet-50 Test Suite",
        "filename": "pytest_verification_study.dcm",
        "file_type": "dicom",
        "prediction": "NORMAL",
        "predicted_class_id": 0,
        "confidence": 0.042,
        "threshold": 0.5,
        "status": "success",
        "disclaimer": "Automated test record",
    }

    # Test save
    inserted_id = await save_prediction_record(test_doc)
    assert inserted_id is not None
    assert isinstance(inserted_id, str)

    # Test retrieve
    records = await get_prediction_history(limit=5)
    assert len(records) > 0
    found = any(r.get("id") == inserted_id or r.get("filename") == "pytest_verification_study.dcm" for r in records)
    assert found is True

    # Test single-record API retrieval endpoint
    with TestClient(app) as client:
        rec_res = client.get(f"/api/imaging/history/{inserted_id}")
        assert rec_res.status_code == 200
        rec_data = rec_res.json()
        assert rec_data["id"] == inserted_id
        assert rec_data["filename"] == "pytest_verification_study.dcm"

    # Cleanup test record
    mongo_client = get_mongo_client()
    if mongo_client:
        from bson import ObjectId
        db = mongo_client[settings.mongodb_db_name]
        await db["analysis_history"].delete_one({"_id": ObjectId(inserted_id)})
