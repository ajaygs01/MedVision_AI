import io
import pytest
from PIL import Image
from fastapi.testclient import TestClient

from app.main import app
from app.services.xai import xai_service


@pytest.fixture
def client():
    return TestClient(app)


def test_xai_service_is_ready():
    assert xai_service.is_ready() is True


def test_xai_status_endpoint(client):
    response = client.get("/api/xai/status")
    assert response.status_code == 200
    data = response.json()
    assert data["gradcam_ready"] is True
    assert "Grad-CAM" in data["supported_methods"][0]


def test_xai_gradcam_computation():
    # Create simple synthetic test image
    img = Image.new("RGB", (224, 224), color=(100, 100, 100))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    file_bytes = buf.getvalue()

    result = xai_service.generate_chest_xray_gradcam(file_bytes, "test_saliency.png")
    assert result["status"] == "success"
    assert result["overlay_preview"] is not None
    assert result["overlay_preview"].startswith("data:image/jpeg;base64,")
    assert result["heatmap_preview"] is not None
    assert result["heatmap_preview"].startswith("data:image/jpeg;base64,")
    assert "Grad-CAM" in result["attribution_method"]


def test_xai_gradcam_endpoint(client):
    # Upload synthetic PNG to /api/xai/gradcam
    img = Image.new("RGB", (224, 224), color=(120, 120, 120))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    files = {"file": ("test_radiograph.png", buf.getvalue(), "image/png")}
    response = client.post("/api/xai/gradcam", files=files)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["overlay_preview"] is not None
    assert data["heatmap_preview"] is not None
    assert data["attribution_method"] is not None


def test_classify_chest_xray_includes_gradcam(client):
    # Test that classify endpoint also returns gradcam_preview
    img = Image.new("RGB", (224, 224), color=(90, 90, 90))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    files = {"file": ("cxr_sample.png", buf.getvalue(), "image/png")}
    response = client.post("/api/imaging/chest-xray/classify", files=files)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["gradcam_preview"] is not None
    assert data["gradcam_preview"].startswith("data:image/jpeg;base64,")
