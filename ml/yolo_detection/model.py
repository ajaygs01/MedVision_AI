"""
YOLO detection model wrapper for MediVision-AI.
Provides safe model loading, real inference, coordinate normalization,
and bounding-box rendering for frontend display.
"""

import base64
import io
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ml.yolo_detection.config import (
    CHECKPOINT_PATH,
    CLASSES,
    DEFAULT_CONF_THRESHOLD,
    DEFAULT_IOU_THRESHOLD,
    METRICS_PATH,
    MODEL_ARCH,
)

logger = logging.getLogger("medivision.ml.yolo_model")


class YOLOModelWrapper:
    """
    Wrapper for Ultralytics YOLOv8 lesion localization model.
    Maintains rigorous separation between trained and untrained states.
    Never fabricates detections when weights are missing.
    """

    def __init__(self, checkpoint_path: Optional[Path] = None):
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else CHECKPOINT_PATH
        self._model = None
        self._is_loaded = False

    @property
    def is_trained(self) -> bool:
        """Returns True only if valid weights file exists on disk."""
        return self.checkpoint_path.exists() and self.checkpoint_path.is_file() and self.checkpoint_path.stat().st_size > 1024

    def get_status(self) -> Dict[str, Any]:
        """Returns the current module and weights status."""
        trained = self.is_trained
        metrics = None
        if METRICS_PATH.exists():
            try:
                import json
                with open(METRICS_PATH, "r", encoding="utf-8") as f:
                    metrics = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to read YOLO metrics: {e}")

        return {
            "model_architecture": MODEL_ARCH,
            "trained": trained,
            "status": "ready" if trained else "not_trained",
            "checkpoint_path": str(self.checkpoint_path) if trained else None,
            "classes": CLASSES,
            "metrics": metrics,
        }

    def load_model(self) -> bool:
        """Loads the YOLO model if weights exist."""
        if not self.is_trained:
            logger.info("YOLO weights not found; model remains uninitialized/untrained.")
            self._model = None
            self._is_loaded = False
            return False

        try:
            from ultralytics import YOLO
            self._model = YOLO(str(self.checkpoint_path))
            self._is_loaded = True
            logger.info(f"Successfully loaded YOLO model from {self.checkpoint_path}")
            return True
        except Exception as exc:
            logger.error(f"Failed to load YOLO model from {self.checkpoint_path}: {exc}")
            self._model = None
            self._is_loaded = False
            return False

    def predict(
        self,
        image: Union[Image.Image, np.ndarray, bytes, Path, str],
        conf_threshold: float = DEFAULT_CONF_THRESHOLD,
        iou_threshold: float = DEFAULT_IOU_THRESHOLD,
    ) -> Dict[str, Any]:
        """
        Runs object detection on the input image.
        If the model is untrained, returns an explicit untrained response
        with 0 detections and zero mock coordinates.
        """
        # If untrained or checkpoint missing, return safe untrained status
        if not self.is_trained:
            return {
                "status": "not_trained",
                "message": "YOLO detection model is not yet trained. No weights found.",
                "detections": [],
                "num_detections": 0,
                "preview_image": None,
                "model_status": self.get_status(),
            }

        if not self._is_loaded:
            loaded = self.load_model()
            if not loaded or self._model is None:
                return {
                    "status": "error",
                    "message": "Failed to load YOLO model weights.",
                    "detections": [],
                    "num_detections": 0,
                    "preview_image": None,
                    "model_status": self.get_status(),
                }

        # Process input image to PIL Image for rendering
        pil_img = self._to_pil_image(image)
        img_w, img_h = pil_img.size

        # Run inference
        results = self._model.predict(
            source=pil_img,
            conf=conf_threshold,
            iou=iou_threshold,
            device="cpu",
            verbose=False,
        )

        detections = []
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            for i in range(len(boxes)):
                xyxy = boxes.xyxy[i].cpu().numpy().tolist()
                conf = float(boxes.conf[i].cpu().numpy().item())
                cls_id = int(boxes.cls[i].cpu().numpy().item())
                cls_name = CLASSES[cls_id] if 0 <= cls_id < len(CLASSES) else f"class_{cls_id}"

                x1, y1, x2, y2 = xyxy
                # Clamp coordinates to image boundaries
                x1 = max(0.0, min(float(x1), float(img_w)))
                y1 = max(0.0, min(float(y1), float(img_h)))
                x2 = max(0.0, min(float(x2), float(img_w)))
                y2 = max(0.0, min(float(y2), float(img_h)))

                # Compute normalized coordinates
                norm_xc = ((x1 + x2) / 2.0) / img_w
                norm_yc = ((y1 + y2) / 2.0) / img_h
                norm_w = (x2 - x1) / img_w
                norm_h = (y2 - y1) / img_h

                detections.append({
                    "id": i + 1,
                    "class_name": cls_name,
                    "class_id": cls_id,
                    "confidence": round(conf, 4),
                    "bbox_pixels": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                    "bbox_normalized": [round(norm_xc, 4), round(norm_yc, 4), round(norm_w, 4), round(norm_h, 4)],
                    "area_pixels": round((x2 - x1) * (y2 - y1), 1),
                })

        # Render preview image with bounding boxes
        preview_base64 = self._render_detection_overlay(pil_img, detections)

        # If no detections passed threshold, probe peak raw candidate confidence
        highest_candidate_conf = None
        if len(detections) == 0:
            try:
                raw_probe = self._model.predict(
                    source=pil_img,
                    imgsz=640,
                    conf=0.001,
                    iou=1.0,
                    verbose=False,
                )
                if len(raw_probe) > 0 and raw_probe[0].boxes is not None and len(raw_probe[0].boxes) > 0:
                    highest_candidate_conf = round(float(raw_probe[0].boxes.conf.max().item()), 4)
            except Exception:
                pass

        return {
            "status": "success",
            "message": f"Detection complete. Found {len(detections)} anatomical findings.",
            "detections": detections,
            "num_detections": len(detections),
            "image_dimensions": {"width": img_w, "height": img_h},
            "conf_threshold": conf_threshold,
            "iou_threshold": iou_threshold,
            "preview_image": preview_base64,
            "highest_candidate_confidence": highest_candidate_conf,
            "model_status": self.get_status(),
        }

    def _to_pil_image(self, image: Union[Image.Image, np.ndarray, bytes, Path, str]) -> Image.Image:
        """Converts diverse image formats to PIL Image in RGB format."""
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        if isinstance(image, (bytes, bytearray)):
            return Image.open(io.BytesIO(image)).convert("RGB")
        if isinstance(image, (str, Path)):
            return Image.open(str(image)).convert("RGB")
        if isinstance(image, np.ndarray):
            if image.ndim == 2:
                return Image.fromarray(image).convert("RGB")
            if image.ndim == 3:
                return Image.fromarray(image)
        raise ValueError(f"Unsupported image type: {type(image)}")

    def _render_detection_overlay(self, image: Image.Image, detections: List[Dict[str, Any]]) -> str:
        """Draws bounding boxes and confidence labels on image and returns a base64 JPEG string."""
        draw_img = image.copy().convert("RGB")
        draw = ImageDraw.Draw(draw_img)

        # Primary palette color: MediVision Amber/Red for lung lesions
        box_color = "#E05638"
        text_color = "#FFFFFF"
        label_bg = "#E05638"

        for det in detections:
            x1, y1, x2, y2 = det["bbox_pixels"]
            conf = det["confidence"]
            cls_name = det["class_name"]

            # Draw outer rectangle
            draw.rectangle([x1, y1, x2, y2], outline=box_color, width=3)

            # Draw label banner
            label_text = f"{cls_name} {conf:.1%}"
            # Approximate font measurement
            text_w = len(label_text) * 7 + 8
            text_h = 16
            draw.rectangle([x1, max(0, y1 - text_h), x1 + text_w, y1], fill=label_bg)
            draw.text((x1 + 4, max(0, y1 - text_h) + 1), label_text, fill=text_color)

        buffered = io.BytesIO()
        draw_img.save(buffered, format="JPEG", quality=85)
        encoded = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{encoded}"
