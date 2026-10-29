"""
Unit and integration tests for MediVision-AI YOLO Anatomical Lesion Localization module.
Verifies RSNA dataset verification, coordinate transformation math, untrained state safety,
API endpoint responses, and bounding box constraints.
"""

import io
import pytest
from httpx import ASGITransport, AsyncClient
from PIL import Image

from app.main import app
from app.services.yolo import yolo_service
from ml.yolo_detection.dataset import convert_box_to_yolo, verify_rsna_source_dataset


def create_synthetic_png(width: int = 256, height: int = 256) -> bytes:
    """Creates an in-memory PNG image for testing."""
    img = Image.new("RGB", (width, height), color=(128, 128, 128))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_rsna_source_dataset_inspection():
    """Verifies that the existing RSNA dataset is correctly identified and inspected."""
    res = verify_rsna_source_dataset()
    assert res["status"] == "ready"
    assert res["total_rows"] == 30227
    assert res["unique_patients"] == 26684
    assert res["num_positive_boxes"] == 9555
    assert res["num_positive_patients"] == 6012


def test_bounding_box_yolo_conversion_math():
    """
    Verifies that absolute (x, y, w, h) boxes convert accurately to normalized YOLO format
    (xc, yc, nw, nh) clamped to [0.0, 1.0].
    """
    # Centered box: 512, 512, 256, 256 on 1024x1024
    xc, yc, nw, nh = convert_box_to_yolo(512.0, 512.0, 256.0, 256.0, 1024.0, 1024.0)
    assert pytest.approx(xc, 0.001) == 0.625  # (512 + 128) / 1024 = 640 / 1024 = 0.625
    assert pytest.approx(yc, 0.001) == 0.625
    assert pytest.approx(nw, 0.001) == 0.25   # 256 / 1024 = 0.25
    assert pytest.approx(nh, 0.001) == 0.25

    # Edge clamping test: box exceeding boundary
    xc_c, yc_c, nw_c, nh_c = convert_box_to_yolo(900.0, 900.0, 300.0, 300.0, 1024.0, 1024.0)
    assert 0.0 <= xc_c <= 1.0
    assert 0.0 <= yc_c <= 1.0
    assert 0.0 <= nw_c <= 1.0
    assert 0.0 <= nh_c <= 1.0


def test_untrained_model_safety():
    """
    Verifies that when model weights are not trained, the service:
    1. Reports trained=False and status='not_trained'
    2. Does NOT fabricate predictions or hallucinate bounding boxes
    3. Produces zero bounding boxes
    """
    if not yolo_service.is_ready():
        stat = yolo_service.get_status()
        assert stat["trained"] is False
        assert stat["status"] == "not_trained"
        assert stat["checkpoint_path"] is None

        img_bytes = create_synthetic_png()
        result = yolo_service.detect(img_bytes, "test_cxr.png")
        assert result["status"] == "not_trained"
        assert result["num_detections"] == 0
        assert result["detections"] == []
        assert "untrained" in result["message"].lower() or "not trained" in result["message"].lower()
        assert result["preview_image"] is not None


@pytest.mark.asyncio
async def test_yolo_status_api_endpoint():
    """Tests GET /api/yolo/status endpoint."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.get("/api/yolo/status")
    assert response.status_code == 200
    data = response.json()
    assert "model_architecture" in data
    assert "trained" in data
    assert "status" in data
    assert "classes" in data
    assert "lung_opacity" in data["classes"]


@pytest.mark.asyncio
async def test_yolo_detect_endpoint_untrained():
    """Tests POST /api/yolo/detect endpoint returns truthful status."""
    img_bytes = create_synthetic_png()
    files = {"file": ("radiograph.png", img_bytes, "image/png")}
    data = {"conf_threshold": "0.30", "iou_threshold": "0.50"}

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.post("/api/yolo/detect", files=files, data=data)

    assert response.status_code == 200
    res = response.json()
    assert "status" in res
    assert "detections" in res
    assert "disclaimer" in res

    if not yolo_service.is_ready():
        assert res["status"] == "not_trained"
        assert res["num_detections"] == 0
        assert len(res["detections"]) == 0


@pytest.mark.asyncio
async def test_yolo_detect_invalid_file_type():
    """Tests POST /api/yolo/detect with unsupported file extension."""
    files = {"file": ("document.txt", b"not an image", "text/plain")}
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.post("/api/yolo/detect", files=files)
    assert response.status_code == 400
    assert "unsupported" in response.json()["detail"].lower() or "allowed formats" in response.json()["detail"].lower()


def test_yolo_training_argparser():
    """Verifies that the YOLO training CLI parser supports warmup, mosaic, and no-promote arguments."""
    from ml.yolo_detection.train import build_parser

    parser = build_parser()

    # 1. Defaults verification
    default_args = parser.parse_args([])
    assert default_args.warmup_epochs == 1.0
    assert default_args.mosaic == 0.0
    assert default_args.no_promote is False

    # 2. Custom values with hyphenated flags
    cmd_args = parser.parse_args([
        "--data", "datasets/yolo_rsna/data.yaml",
        "--weights", "yolov8n.pt",
        "--epochs", "10",
        "--batch", "2",
        "--imgsz", "320",
        "--workers", "0",
        "--device", "cpu",
        "--name", "rsna_cpu_exp2",
        "--warmup-epochs", "1.0",
        "--mosaic", "0.0",
        "--no-promote",
    ])
    assert cmd_args.epochs == 10
    assert cmd_args.name == "rsna_cpu_exp2"
    assert cmd_args.warmup_epochs == 1.0
    assert cmd_args.mosaic == 0.0
    assert cmd_args.no_promote is True

    # 3. Custom values with underscored flags
    alt_args = parser.parse_args([
        "--warmup_epochs", "2.5",
        "--mosaic", "0.5",
        "--no_promote",
    ])
    assert alt_args.warmup_epochs == 2.5
    assert alt_args.mosaic == 0.5
    assert alt_args.no_promote is True


@pytest.mark.asyncio
async def test_yolo_detect_endpoint_returns_preview_and_schema():
    """Verifies that POST /api/yolo/detect returns a valid preview image and correct detection structure."""
    img_bytes = create_synthetic_png(width=128, height=128)
    files = {"file": ("test_study.png", img_bytes, "image/png")}
    data = {"conf_threshold": "0.25", "iou_threshold": "0.45"}

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.post("/api/yolo/detect", files=files, data=data)

    assert response.status_code == 200
    res = response.json()
    assert res["status"] in ("success", "not_trained")
    assert "preview_image" in res
    assert res["preview_image"] is not None
    assert res["preview_image"].startswith("data:image/jpeg;base64,")
    assert res["conf_threshold"] == 0.25
    assert res["iou_threshold"] == 0.45
    assert isinstance(res["detections"], list)
    assert isinstance(res["num_detections"], int)
    assert res["num_detections"] == len(res["detections"])
    if res["num_detections"] == 0:
        assert len(res["detections"]) == 0

