import io
import pytest
from httpx import ASGITransport, AsyncClient
from PIL import Image
from app.main import app


@pytest.fixture
def dummy_png_bytes():
    img = Image.new("L", (128, 128), color=128)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.mark.asyncio
async def test_classify_chest_xray_endpoint(dummy_png_bytes):
    """
    Verifies that calling POST /api/imaging/chest-xray/classify
    returns 200 OK with proper status ('success' when trained, 'not_trained' when absent)
    and valid schema without fabricated values.
    """
    from app.services.chest_xray import chest_xray_service

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        files = {
            "file": ("test_radiograph.png", dummy_png_bytes, "image/png"),
        }
        response = await ac.post("/api/imaging/chest-xray/classify", files=files)

    assert response.status_code == 200
    data = response.json()
    assert "Academic research prototype" in data["disclaimer"]

    if chest_xray_service.is_ready():
        assert data["status"] == "success"
        assert data["prediction"] in ["Lung Opacity / Pneumonia", "Normal / No Lung Opacity"]
        assert data["predicted_class_id"] in [0, 1]
        assert isinstance(data["confidence"], float)
        assert 0.0 <= data["confidence"] <= 1.0
    else:
        assert data["status"] == "not_trained"
        assert data["prediction"] is None
        assert data["confidence"] is None


@pytest.mark.asyncio
async def test_classify_chest_xray_invalid_file():
    """
    Verifies that uploading an invalid file format returns HTTP 400 Bad Request.
    """
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        files = {
            "file": ("report.txt", b"plain text report", "text/plain"),
        }
        response = await ac.post("/api/imaging/chest-xray/classify", files=files)

    assert response.status_code == 400
    data = response.json()
    assert "Unsupported file format" in data["detail"]


@pytest.mark.asyncio
async def test_classify_chest_xray_empty_file():
    """
    Verifies that uploading an empty file returns HTTP 400 Bad Request.
    """
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        files = {
            "file": ("empty.png", b"", "image/png"),
        }
        response = await ac.post("/api/imaging/chest-xray/classify", files=files)

    assert response.status_code == 400
    data = response.json()
    assert "empty" in data["detail"].lower()


@pytest.mark.asyncio
async def test_models_status_reports_stage2_pipeline():
    """
    Verifies that GET /api/models/status reports stage 2
    and correctly reflects that CNN model is unconfigured until trained.
    """
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.get("/api/models/status")

    assert response.status_code == 200
    data = response.json()
    assert data["stage"] == 2
    assert "cnn_chest_xray" in data["models"]
    cnn_model = data["models"]["cnn_chest_xray"]
    assert cnn_model["status"] in ["unconfigured", "ready"]
    assert cnn_model["trained"] in [True, False]
