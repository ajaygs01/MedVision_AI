import base64
import io
import logging
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
from PIL import Image

try:
    import torch
    import torchvision.transforms as transforms
except ImportError:
    torch = None
    transforms = None

try:
    import pydicom
except ImportError:
    pydicom = None

from ml.chest_xray.cnn.config import CNNConfig, default_config
from ml.chest_xray.cnn.train import build_resnet50_model

logger = logging.getLogger("medivision.services.chest_xray")

ALLOWED_EXTENSIONS = {".dcm", ".png", ".jpg", ".jpeg"}
MAX_FILE_SIZE_BYTES = 25 * 1024 * 1024  # 25 MB
RESEARCH_DISCLAIMER = (
    "Academic research prototype. Results are not clinically validated "
    "and must not be used as a standalone medical diagnosis."
)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class ChestXRayService:
    """
    Dedicated Chest X-ray inference service for FastAPI.
    Loads and caches the ResNet-50 checkpoint once at application startup
    and performs validated inference on DICOM (.dcm) and standard images.
    """

    def __init__(self, config: Optional[CNNConfig] = None, device: Optional[str] = "cpu"):
        self.config = config or default_config
        self.device = device or "cpu"
        self._model: Optional["torch.nn.Module"] = None
        self._is_loaded: bool = False
        self._transform = transforms.Compose([
            transforms.Resize(self.config.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]) if transforms is not None else None

    @property
    def checkpoint_path(self) -> Path:
        return self.config.checkpoint_path

    def is_model_trained(self) -> bool:
        """Checks if a valid, non-empty checkpoint file exists on disk."""
        return self.checkpoint_path.exists() and self.checkpoint_path.stat().st_size > 0

    def is_ready(self) -> bool:
        """Returns True if the model has been loaded into memory and is ready for inference."""
        return self._is_loaded and self._model is not None

    def load_model(self, device: Optional[str] = None) -> bool:
        """
        Loads the trained PyTorch ResNet-50 model into memory once at application startup.
        Safe against missing or unconfigured checkpoints (logs status without crashing).
        """
        if device is not None:
            self.device = device

        if not self.is_model_trained():
            logger.warning(
                f"[ChestXRayService] Checkpoint not found at '{self.checkpoint_path}'. "
                "Service operating in unconfigured/not_trained state."
            )
            self._is_loaded = False
            self._model = None
            return False

        if torch is None:
            logger.error("[ChestXRayService] PyTorch is not installed in current environment.")
            return False

        try:
            logger.info(f"[ChestXRayService] Loading ResNet-50 weights from '{self.checkpoint_path}' on {self.device}...")
            checkpoint_data = torch.load(self.checkpoint_path, map_location=self.device)

            if isinstance(checkpoint_data, dict) and "model_state_dict" in checkpoint_data:
                state_dict = checkpoint_data["model_state_dict"]
            else:
                state_dict = checkpoint_data

            # Detect whether output layer is direct Linear(2048, 1) or Sequential Dropout+Linear
            has_fc_weight = "fc.weight" in state_dict
            model = build_resnet50_model(pretrained=False, use_dropout=not has_fc_weight)
            model.load_state_dict(state_dict)

            model.to(self.device)
            model.eval()
            self._model = model
            self._is_loaded = True
            logger.info("[ChestXRayService] ResNet-50 model loaded and cached successfully.")
            return True
        except Exception as exc:
            logger.error(f"[ChestXRayService] Failed to load checkpoint: {exc}", exc_info=True)
            self._is_loaded = False
            self._model = None
            return False

    def validate_upload(self, file_bytes: bytes, filename: str) -> Tuple[bool, str, str]:
        """
        Validates file size, extension, and data integrity.
        Returns (is_valid, file_type, error_message).
        """
        if not file_bytes:
            return False, "unknown", "Uploaded file is empty (0 bytes)."

        if len(file_bytes) > MAX_FILE_SIZE_BYTES:
            return False, "unknown", f"File size exceeds limit of {MAX_FILE_SIZE_BYTES // (1024 * 1024)} MB."

        ext = Path(filename).suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            return (
                False,
                "unknown",
                f"Unsupported file format '{ext}'. Allowed formats: {', '.join(sorted(ALLOWED_EXTENSIONS))}.",
            )

        file_type = "dicom" if ext == ".dcm" else "standard_image"

        try:
            if file_type == "dicom":
                if pydicom is None:
                    return False, file_type, "pydicom library not installed on backend."
                pydicom.dcmread(io.BytesIO(file_bytes))
            else:
                img = Image.open(io.BytesIO(file_bytes))
                img.verify()
        except Exception as exc:
            return False, file_type, f"Corrupted or invalid image data: {str(exc)}"

        return True, file_type, ""

    def process_image_bytes(self, file_bytes: bytes, filename: str) -> Image.Image:
        """
        Decodes raw bytes into a 3-channel RGB PIL Image.
        Handles DICOM MONOCHROME1 inversion and min-max uint8 intensity scaling.
        """
        ext = Path(filename).suffix.lower()

        if ext == ".dcm":
            if pydicom is None:
                raise ImportError("pydicom is required to decode DICOM files.")
            dcm = pydicom.dcmread(io.BytesIO(file_bytes))
            pixel_array = dcm.pixel_array.astype(np.float32)

            # Invert MONOCHROME1 where 0 is brightest
            if getattr(dcm, "PhotometricInterpretation", "") == "MONOCHROME1":
                pixel_array = np.max(pixel_array) - pixel_array

            # Intensity normalization to [0, 255] uint8
            p_min, p_max = np.min(pixel_array), np.max(pixel_array)
            if p_max > p_min:
                norm = ((pixel_array - p_min) / (p_max - p_min) * 255.0).astype(np.uint8)
            else:
                norm = np.zeros_like(pixel_array, dtype=np.uint8)

            return Image.fromarray(norm).convert("RGB")
        else:
            img = Image.open(io.BytesIO(file_bytes))
            return img.convert("RGB")

    def generate_preview_data_url(self, pil_image: Image.Image, max_dim: int = 512) -> str:
        """
        Generates a lightweight, compressed base64 JPEG data URL
        suitable for fast browser display and state persistence without heavy payloads.
        """
        try:
            thumb = pil_image.copy()
            thumb.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
            buf = io.BytesIO()
            thumb.save(buf, format="JPEG", quality=85)
            b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")
            return f"data:image/jpeg;base64,{b64_str}"
        except Exception as exc:
            logger.warning(f"Could not generate image preview: {exc}")
            return ""

    def predict(self, file_bytes: bytes, filename: str, threshold: float = 0.5) -> Dict:
        """
        Executes binary pneumonia classification on an uploaded radiograph.
        If the model is unconfigured or not loaded, returns a clean 'not_trained' response.
        Never fabricates predictions or confidence scores.
        """
        is_valid, file_type, err_msg = self.validate_upload(file_bytes, filename)
        if not is_valid:
            return {
                "status": "error",
                "model_name": "Chest X-ray ResNet50 Classifier",
                "prediction": None,
                "confidence": None,
                "threshold": threshold,
                "filename": filename,
                "file_type": file_type,
                "message": err_msg,
                "disclaimer": RESEARCH_DISCLAIMER,
                "image_preview": None,
                "gradcam_preview": None,
            }

        # Generate preview data URL from valid upload
        try:
            pil_image = self.process_image_bytes(file_bytes, filename)
            preview_data_url = self.generate_preview_data_url(pil_image)
        except Exception as exc:
            logger.error(f"[ChestXRayService] Could not process image bytes: {exc}")
            pil_image = None
            preview_data_url = ""

        # Check if model is trained and loaded
        if not self.is_ready():
            # Attempt a lazy load if checkpoint became available after startup
            if self.is_model_trained():
                self.load_model()

        if not self.is_ready():
            return {
                "status": "not_trained",
                "model_name": "Chest X-ray ResNet50 Classifier",
                "prediction": None,
                "predicted_class_id": None,
                "confidence": None,
                "threshold": threshold,
                "filename": filename,
                "file_type": file_type,
                "message": (
                    f"Model weights checkpoint '{self.checkpoint_path.name}' is not loaded. "
                    "The ResNet-50 CNN model must be trained or configured before inference."
                ),
                "disclaimer": RESEARCH_DISCLAIMER,
                "image_preview": preview_data_url or None,
                "gradcam_preview": None,
            }

        try:
            if pil_image is None:
                pil_image = self.process_image_bytes(file_bytes, filename)
                preview_data_url = self.generate_preview_data_url(pil_image)

            if self._transform is None:
                raise RuntimeError("Torchvision image transform pipeline is uninitialized.")

            tensor = self._transform(pil_image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                logits = self._model(tensor)
                prob = float(torch.sigmoid(logits[0, 0]).item())

            has_opacity = prob >= threshold
            predicted_class = "Lung Opacity / Pneumonia" if has_opacity else "Normal / No Lung Opacity"
            predicted_class_id = 1 if has_opacity else 0

            # Generate Explainable AI Grad-CAM saliency overlay and standalone heatmap
            gradcam_url = None
            heatmap_url = None
            try:
                from app.services.xai import xai_service
                cam_res = xai_service.generate_chest_xray_gradcam(file_bytes, filename)
                if cam_res.get("status") == "success":
                    gradcam_url = cam_res.get("overlay_preview")
                    heatmap_url = cam_res.get("heatmap_preview")
            except Exception as cam_exc:
                logger.warning(f"Grad-CAM generation skipped/failed: {cam_exc}")

            return {
                "status": "success",
                "model_name": "Chest X-ray ResNet50 Classifier",
                "prediction": predicted_class,
                "predicted_class_id": predicted_class_id,
                "confidence": round(prob, 4),
                "threshold": threshold,
                "filename": filename,
                "file_type": file_type,
                "message": "Inference completed successfully using trained ResNet50 checkpoint.",
                "disclaimer": RESEARCH_DISCLAIMER,
                "image_preview": preview_data_url or None,
                "gradcam_preview": gradcam_url,
                "heatmap_preview": heatmap_url,
            }
        except Exception as exc:
            logger.error(f"[ChestXRayService] Inference failure on '{filename}': {exc}", exc_info=True)
            return {
                "status": "error",
                "model_name": "Chest X-ray ResNet50 Classifier",
                "prediction": None,
                "confidence": None,
                "threshold": threshold,
                "filename": filename,
                "file_type": file_type,
                "message": f"Inference execution failed: {str(exc)}",
                "disclaimer": RESEARCH_DISCLAIMER,
                "image_preview": preview_data_url or None,
                "gradcam_preview": None,
            }


# Global singleton instance
chest_xray_service = ChestXRayService()
