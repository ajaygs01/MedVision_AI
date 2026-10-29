import io
import base64
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
import torch
import torch.nn.functional as F
from PIL import Image

from ml.skin_lesion.config import (
    SkinLesionConfig,
    HAM10000_CLASSES,
    CLASS_NAMES,
    IDX_TO_CLASS,
)
from ml.skin_lesion.model import build_skin_lesion_model
from ml.skin_lesion.dataset import get_skin_transforms

logger = logging.getLogger("medivision.services.skin_lesion")


class SkinLesionService:
    """
    Singleton service managing model weights, inference execution,
    and safe untrained fallbacks for Skin Lesion Analysis.
    """

    def __init__(self, checkpoint_path: Optional[Path] = None):
        self.config = SkinLesionConfig()
        self.checkpoint_path = checkpoint_path or self.config.get_checkpoint_path()
        self.device = torch.device(self.config.device)
        self.model: Optional[torch.nn.Module] = None
        self._transforms = get_skin_transforms(self.config.image_size, is_train=False)

    def is_model_trained(self) -> bool:
        """Returns True if a valid checkpoint file exists on disk."""
        return self.checkpoint_path.exists() and self.checkpoint_path.is_file()

    def is_ready(self) -> bool:
        """Returns True if the model is currently instantiated and loaded in memory."""
        if self.model is not None:
            return True
        if self.is_model_trained():
            return self.load_model()
        return False

    def load_model(self) -> bool:
        """Safely loads model weights from disk into memory."""
        if not self.is_model_trained():
            logger.info("Skin Lesion checkpoint not present on disk; service in unconfigured mode.")
            return False

        try:
            logger.info(f"Loading Skin Lesion EfficientNet-B0 weights from {self.checkpoint_path}...")
            model = build_skin_lesion_model(num_classes=self.config.num_classes, pretrained=False)
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
            state_dict = checkpoint.get("model_state_dict", checkpoint)
            model.load_state_dict(state_dict)
            model.to(self.device)
            model.eval()
            self.model = model
            logger.info("Skin Lesion model loaded successfully.")
            return True
        except Exception as exc:
            logger.error(f"Failed to load Skin Lesion model: {exc}")
            self.model = None
            return False

    def get_status(self) -> Dict[str, Any]:
        """Returns comprehensive status telemetry adhering to ModelItemStatus schema."""
        trained = self.is_ready()
        return {
            "name": "Skin Lesion Multi-Class Classifier",
            "modality": "Dermoscopy / Epiluminescence",
            "architecture": "EfficientNet-B0 Multi-Class",
            "framework": "PyTorch 2.x / Torchvision",
            "status": "ready" if trained else "unconfigured",
            "trained": trained,
            "weights_path": str(self.checkpoint_path) if trained else None,
            "next_stage": "Stage 2 Lesion Color Augmentation & Preprocessing (HAM10000 / ISIC)",
            "description": "High-resolution dermatological classifier for melanoma, basal cell carcinoma, and benign keratosis.",
        }

    def predict(self, file_bytes: bytes, filename: str) -> Dict[str, Any]:
        """
        Executes multi-class inference on uploaded dermoscopy image.
        If weights are unconfigured, safely returns status='not_trained'.
        """
        disclaimer = (
            "Academic research prototype. Results are generated for investigative and educational "
            "purposes only and must not be used as standalone clinical dermatological diagnoses."
        )

        if not self.is_ready():
            return {
                "status": "not_trained",
                "model_name": "EfficientNet-B0 (Unconfigured)",
                "prediction": None,
                "predicted_class_code": None,
                "predicted_class_id": None,
                "confidence": None,
                "probabilities": None,
                "filename": filename,
                "file_type": "standard_image",
                "message": (
                    "Skin Lesion Classifier model weights are not configured. "
                    "Complete Stage 2 training using the HAM10000 / ISIC dataset before initiating clinical inferences."
                ),
                "disclaimer": disclaimer,
                "image_preview": None,
            }

        try:
            image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
            
            # Generate Base64 thumbnail preview for frontend display
            preview_img = image.copy()
            preview_img.thumbnail((300, 300))
            preview_buf = io.BytesIO()
            preview_img.save(preview_buf, format="JPEG", quality=85)
            preview_base64 = f"data:image/jpeg;base64,{base64.b64encode(preview_buf.getvalue()).decode('utf-8')}"

            # Preprocess tensor
            input_tensor = self._transforms(image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                logits = self.model(input_tensor)
                probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()

            top_idx = int(probs.argmax())
            top_code = IDX_TO_CLASS[top_idx]
            top_name = CLASS_NAMES.get(top_code, top_code)
            confidence = float(probs[top_idx])

            probabilities: List[Dict[str, Any]] = []
            for idx, code in enumerate(IDX_TO_CLASS):
                probabilities.append({
                    "class_code": code,
                    "class_name": CLASS_NAMES.get(code, code),
                    "probability": round(float(probs[idx]), 4),
                })

            # Sort probabilities descending
            probabilities.sort(key=lambda x: x["probability"], reverse=True)

            return {
                "status": "success",
                "model_name": "EfficientNet-B0 Multi-Class Classifier",
                "prediction": top_name,
                "predicted_class_code": top_code,
                "predicted_class_id": top_idx,
                "confidence": round(confidence, 4),
                "probabilities": probabilities,
                "filename": filename,
                "file_type": "standard_image",
                "message": f"Dermoscopy evaluation completed: {top_name} ({confidence * 100:.2f}% confidence).",
                "disclaimer": disclaimer,
                "image_preview": preview_base64,
            }

        except Exception as exc:
            logger.error(f"Inference execution failed on {filename}: {exc}")
            return {
                "status": "error",
                "model_name": "EfficientNet-B0",
                "prediction": None,
                "predicted_class_code": None,
                "predicted_class_id": None,
                "confidence": None,
                "probabilities": None,
                "filename": filename,
                "file_type": "standard_image",
                "message": f"Dermoscopy image processing failed: {str(exc)}",
                "disclaimer": disclaimer,
                "image_preview": None,
            }


# Global singleton instance
skin_lesion_service = SkinLesionService()
