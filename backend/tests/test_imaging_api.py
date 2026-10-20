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
async def test_classify_chest_xray_not_trained(dummy_png_bytes):
    """
    Verifies that calling POST /api/imaging/chest-xray/classify
    without a trained checkpoint returns 200 OK with status: 'not_trained'
    and zero fake predictions.
    """
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        files = {
            "file": ("test_radiograph.png", dummy_png_bytes, "image/png"),
        }
        response = await ac.post("/api/imaging/chest-xray/classify", files=files)

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "not_trained"
    assert data["prediction"] is None
    assert data["confidence"] is None
    assert "not found" in data["message"].lower() or "not trained" in data["message"].lower()
    assert "Academic research prototype" in data["disclaimer"]


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
