import io
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
from PIL import Image

try:
    import torch
except ImportError:
    torch = None

try:
    import pydicom
except ImportError:
    pydicom = None

from ml.chest_xray.cnn.config import CNNConfig, default_config
from ml.chest_xray.cnn.dataset import get_transforms, read_dicom_or_image
from ml.chest_xray.cnn.train import build_resnet50_model

logger = logging.getLogger("medivision.ml.predict")

ALLOWED_EXTENSIONS = {".dcm", ".png", ".jpg", ".jpeg"}
MAX_FILE_SIZE_BYTES = 25 * 1024 * 1024  # 25MB
RESEARCH_DISCLAIMER = (
    "Academic research prototype. Results are not clinically validated "
    "and must not be used as a standalone medical diagnosis."
)


class ChestXRayInferenceService:
    """
    Inference service for the ResNet-50 Chest X-ray Pneumonia Classifier.
    Maintains clean separation between trained model inference and untrained status.
    """

    def __init__(self, config: Optional[CNNConfig] = None):
        self.config = config or default_config
        self.device = self.config.get_device()
        self._model: Optional["torch.nn.Module"] = None
        self._transform = get_transforms(self.config.img_size, is_train=False)

    @property
    def checkpoint_path(self) -> Path:
        return self.config.checkpoint_path

    def is_model_trained(self) -> bool:
        """Checks if a valid trained checkpoint file exists on disk."""
        return self.checkpoint_path.exists() and self.checkpoint_path.stat().st_size > 0

    def get_model_status_info(self) -> Dict:
        """Returns metadata about the CNN model status."""
        trained = self.is_model_trained()
        return {
            "model_name": "Chest X-ray ResNet50 Classifier",
            "modality": "Chest Radiograph (CXR)",
            "architecture": "ResNet-50 ImageNet Transfer Learning",
            "trained": trained,
            "status": "ready" if trained else "unconfigured",
            "weights_path": str(self.checkpoint_path) if trained else None,
            "device": self.device,
            "disclaimer": RESEARCH_DISCLAIMER,
        }

    def load_model(self) -> Optional["torch.nn.Module"]:
        """Loads and caches the trained PyTorch ResNet-50 model at startup if checkpoint exists."""
        if not self.is_model_trained():
            logger.info(f"Model checkpoint not found at '{self.checkpoint_path}'. Service initialized in not_trained state.")
            return None
        return self._load_model()

    def _load_model(self) -> "torch.nn.Module":
        """Loads and caches the trained PyTorch ResNet-50 model."""
        if not self.is_model_trained():
            raise FileNotFoundError(
                f"Model weights checkpoint not found at '{self.checkpoint_path}'. "
                "Training must be completed before inference."
            )

        if self._model is None:
            if torch is None:
                raise ImportError("PyTorch is required for model inference.")

            logger.info(f"Loading ResNet-50 checkpoint from '{self.checkpoint_path}' on {self.device}...")
            checkpoint_data = torch.load(self.checkpoint_path, map_location=self.device)

            if isinstance(checkpoint_data, dict) and "model_state_dict" in checkpoint_data:
                state_dict = checkpoint_data["model_state_dict"]
            else:
                state_dict = checkpoint_data

            # Support both direct Linear(2048, 1) and Sequential Dropout+Linear
            has_fc_weight = "fc.weight" in state_dict
            model = build_resnet50_model(pretrained=False, use_dropout=not has_fc_weight)
            model.load_state_dict(state_dict)

            model.to(self.device)
            model.eval()
            self._model = model
            logger.info("ResNet-50 model loaded successfully.")

        return self._model

    def validate_upload(
        self, file_bytes: bytes, filename: str
    ) -> Tuple[bool, str, str]:
        """
        Validates file size, extension, and integrity (DICOM or standard image).
        Returns (is_valid, file_type, error_message).
        """
        if not file_bytes:
            return False, "unknown", "Uploaded file is empty (0 bytes)."

        if len(file_bytes) > MAX_FILE_SIZE_BYTES:
            return (
                False,
                "unknown",
                f"File size exceeds limit of {MAX_FILE_SIZE_BYTES // (1024 * 1024)} MB.",
            )

        ext = Path(filename).suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            return (
                False,
                "unknown",
                f"Unsupported file format '{ext}'. Allowed formats: {', '.join(sorted(ALLOWED_EXTENSIONS))}.",
            )

        file_type = "dicom" if ext == ".dcm" else "standard_image"

        # Integrity verification
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

    def process_image_bytes(
        self, file_bytes: bytes, filename: str
    ) -> Image.Image:
        """
        Converts raw uploaded bytes (DICOM or PNG/JPG) into an RGB PIL Image.
        """
        ext = Path(filename).suffix.lower()

        if ext == ".dcm":
            if pydicom is None:
                raise ImportError("pydicom is required to decode DICOM bytes.")
            dcm = pydicom.dcmread(io.BytesIO(file_bytes))
            pixel_array = dcm.pixel_array.astype(np.float32)

            if getattr(dcm, "PhotometricInterpretation", "") == "MONOCHROME1":
                pixel_array = np.max(pixel_array) - pixel_array

            p_min, p_max = np.min(pixel_array), np.max(pixel_array)
            if p_max > p_min:
                norm = ((pixel_array - p_min) / (p_max - p_min) * 255.0).astype(np.uint8)
            else:
                norm = np.zeros_like(pixel_array, dtype=np.uint8)

            return Image.fromarray(norm).convert("RGB")
        else:
            img = Image.open(io.BytesIO(file_bytes))
            return img.convert("RGB")

    def predict(
        self, file_bytes: bytes, filename: str, threshold: float = 0.5
    ) -> Dict:
        """
        Classifies an uploaded chest radiograph.
        If no checkpoint exists, returns clean 'not_trained' response without fake inference.
        """
        is_valid, file_type, err_msg = self.validate_upload(file_bytes, filename)
        if not is_valid:
            return {
                "status": "error",
                "model_name": "Chest X-ray ResNet50 Classifier",
                "prediction": None,
                "confidence": None,
                "filename": filename,
                "file_type": file_type,
                "message": err_msg,
                "disclaimer": RESEARCH_DISCLAIMER,
            }

        # Checkpoint check: if model is NOT trained, report truthfully
        if not self.is_model_trained():
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
                    f"Model weights checkpoint '{self.checkpoint_path.name}' not found. "
                    "The ResNet-50 CNN model has not been trained yet. "
                    "Please execute Stage 2 training using the RSNA dataset."
                ),
                "disclaimer": RESEARCH_DISCLAIMER,
            }

        # Model is trained: execute real inference
        try:
            model = self._load_model()
            pil_image = self.process_image_bytes(file_bytes, filename)

            if self._transform is None:
                raise RuntimeError("Image transform pipeline not initialized.")

            tensor = self._transform(pil_image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                logits = model(tensor)
                prob = float(torch.sigmoid(logits).item())

            has_opacity = prob >= threshold
            predicted_class = "Lung Opacity / Pneumonia" if has_opacity else "Normal / No Lung Opacity"
            predicted_class_id = 1 if has_opacity else 0

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
            }
        except Exception as exc:
            logger.error(f"Inference error on {filename}: {exc}", exc_info=True)
            return {
                "status": "error",
                "model_name": "Chest X-ray ResNet50 Classifier",
                "prediction": None,
                "confidence": None,
                "filename": filename,
                "file_type": file_type,
                "message": f"Inference failed during execution: {str(exc)}",
                "disclaimer": RESEARCH_DISCLAIMER,
            }


# Singleton service instance
inference_service = ChestXRayInferenceService()
