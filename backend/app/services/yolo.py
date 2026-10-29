"""
YOLO detection service for FastAPI backend.
Coordinates image preprocessing, DICOM conversion, YOLOModelWrapper inference,
and result formatting with safe unconfigured handling.
"""

import base64
import io
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
from PIL import Image

try:
    import pydicom
except ImportError:
    pydicom = None

from ml.yolo_detection.config import (
    CHECKPOINT_PATH,
    CLASSES,
    DEFAULT_CONF_THRESHOLD,
    DEFAULT_IOU_THRESHOLD,
    MODEL_ARCH,
)
from ml.yolo_detection.model import YOLOModelWrapper

logger = logging.getLogger("medivision.services.yolo")

ALLOWED_EXTENSIONS = {".dcm", ".png", ".jpg", ".jpeg"}
MAX_FILE_SIZE_BYTES = 25 * 1024 * 1024  # 25 MB
RESEARCH_DISCLAIMER = (
    "Anatomical object detection is an investigational research prototype. "
    "Bounding boxes are indicative of localization patterns and must not be used "
    "as a standalone clinical diagnosis."
)


class YOLOService:
    """
    Dedicated YOLO Anatomical Lesion Localization inference service.
    Maintains clean separation between trained and untrained states.
    Never fabricates predictions when weights are missing.
    """

    def __init__(self, checkpoint_path: Optional[Path] = None):
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else CHECKPOINT_PATH
        self.wrapper = YOLOModelWrapper(self.checkpoint_path)

    def is_ready(self) -> bool:
        """Returns True only if valid model weights exist on disk."""
        return self.wrapper.is_trained

    def is_model_trained(self) -> bool:
        """Returns True only if valid model weights exist on disk."""
        return self.wrapper.is_trained

    def get_status(self) -> Dict[str, Any]:
        """Returns current model status, architecture, and validation metrics."""
        return self.wrapper.get_status()

    def validate_upload(self, file_bytes: bytes, filename: str) -> Tuple[bool, str, str]:
        """
        Validates the uploaded file extension and byte size.
        Returns: (is_valid, file_type, error_message)
        """
        if not file_bytes:
            return False, "unknown", "Uploaded file is empty (0 bytes)."

        if len(file_bytes) > MAX_FILE_SIZE_BYTES:
            max_mb = MAX_FILE_SIZE_BYTES // (1024 * 1024)
            return False, "unknown", f"File size exceeds maximum allowed limit of {max_mb} MB."

        suffix = Path(filename).suffix.lower() if filename else ""
        if suffix not in ALLOWED_EXTENSIONS:
            allowed = ", ".join(sorted(ALLOWED_EXTENSIONS))
            return False, "unknown", f"Unsupported file type '{suffix}'. Allowed formats: {allowed}"

        file_type = "dicom" if suffix == ".dcm" else "standard_image"
        return True, file_type, ""

    def process_image_bytes(self, file_bytes: bytes, filename: str) -> Image.Image:
        """Decodes DICOM (.dcm) or standard image bytes into an RGB PIL Image."""
        suffix = Path(filename).suffix.lower() if filename else ""
        if suffix == ".dcm":
            if pydicom is None:
                raise RuntimeError("pydicom is required to parse DICOM files.")
            ds = pydicom.dcmread(io.BytesIO(file_bytes))
            arr = ds.pixel_array.astype(np.float32)

            if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
                arr = np.max(arr) - arr

            min_val, max_val = arr.min(), arr.max()
            if max_val > min_val:
                arr = (arr - min_val) / (max_val - min_val) * 255.0
            else:
                arr = np.zeros_like(arr)

            return Image.fromarray(arr.astype(np.uint8)).convert("RGB")
        else:
            return Image.open(io.BytesIO(file_bytes)).convert("RGB")

    def generate_preview_data_url(self, pil_image: Image.Image) -> str:
        """Generates a base64 JPEG data URL for display."""
        buffered = io.BytesIO()
        pil_image.save(buffered, format="JPEG", quality=85)
        encoded = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{encoded}"

    def detect(
        self,
        file_bytes: bytes,
        filename: str,
        conf_threshold: float = DEFAULT_CONF_THRESHOLD,
        iou_threshold: float = DEFAULT_IOU_THRESHOLD,
    ) -> Dict[str, Any]:
        """
        Executes YOLO anatomical lesion localization on uploaded image.
        If the model is untrained, returns an explicit not_trained response with 0 detections.
        """
        is_valid, file_type, err_msg = self.validate_upload(file_bytes, filename)
        if not is_valid:
            return {
                "status": "error",
                "message": err_msg,
                "filename": filename,
                "detections": [],
                "num_detections": 0,
                "preview_image": None,
                "conf_threshold": conf_threshold,
                "iou_threshold": iou_threshold,
                "model_status": self.get_status(),
                "disclaimer": RESEARCH_DISCLAIMER,
            }

        try:
            pil_image = self.process_image_bytes(file_bytes, filename)
            preview_url = self.generate_preview_data_url(pil_image)
        except Exception as exc:
            logger.error(f"Image processing failed for '{filename}': {exc}")
            return {
                "status": "error",
                "message": f"Failed to process image: {exc}",
                "filename": filename,
                "detections": [],
                "num_detections": 0,
                "preview_image": None,
                "conf_threshold": conf_threshold,
                "iou_threshold": iou_threshold,
                "model_status": self.get_status(),
                "disclaimer": RESEARCH_DISCLAIMER,
            }

        # If model is not trained, return truthful not_trained response
        if not self.is_ready():
            return {
                "status": "not_trained",
                "message": (
                    f"YOLOv8 lesion detection model is untrained. No weights found at '{self.checkpoint_path.name}'. "
                    "Dataset preparation and training must be completed before detections can be produced."
                ),
                "filename": filename,
                "detections": [],
                "num_detections": 0,
                "image_dimensions": {"width": pil_image.width, "height": pil_image.height},
                "conf_threshold": conf_threshold,
                "iou_threshold": iou_threshold,
                "preview_image": preview_url,
                "model_status": self.get_status(),
                "disclaimer": RESEARCH_DISCLAIMER,
            }

        # Run model inference
        pred_result = self.wrapper.predict(
            pil_image,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
        )

        return {
            "status": pred_result.get("status", "success"),
            "message": pred_result.get("message", "Detection completed successfully."),
            "filename": filename,
            "detections": pred_result.get("detections", []),
            "num_detections": pred_result.get("num_detections", 0),
            "image_dimensions": pred_result.get("image_dimensions", {"width": pil_image.width, "height": pil_image.height}),
            "conf_threshold": conf_threshold,
            "iou_threshold": iou_threshold,
            "preview_image": pred_result.get("preview_image") or preview_url,
            "highest_candidate_confidence": pred_result.get("highest_candidate_confidence"),
            "model_status": pred_result.get("model_status", self.get_status()),
            "disclaimer": RESEARCH_DISCLAIMER,
        }


# Global singleton instance
yolo_service = YOLOService()
