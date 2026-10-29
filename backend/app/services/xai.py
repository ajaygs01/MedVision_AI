import base64
import io
import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np
from PIL import Image

try:
    import torch
    import torchvision.transforms as transforms
except ImportError:
    torch = None
    transforms = None

try:
    import matplotlib
except ImportError:
    matplotlib = None

from app.services.chest_xray import chest_xray_service

logger = logging.getLogger("medivision.services.xai")


class XAIService:
    """
    Explainable AI (XAI) Service providing Gradient-weighted Class Activation Mapping (Grad-CAM)
    for deep convolutional neural networks (specifically ResNet-50 Chest Radiograph Classifier).
    """

    def __init__(self):
        self.target_layer_name = "layer4"

    def is_ready(self) -> bool:
        """Returns True if the underlying Chest X-ray ResNet50 model is ready and PyTorch/Matplotlib are available."""
        return (
            torch is not None
            and matplotlib is not None
            and (chest_xray_service.is_ready() or chest_xray_service.is_model_trained())
        )

    def generate_chest_xray_gradcam(
        self,
        file_bytes: bytes,
        filename: str,
        alpha: float = 0.45,
    ) -> Dict[str, Any]:
        """
        Executes Grad-CAM attribution on an uploaded chest X-ray image using the trained ResNet-50 model.
        Returns base64 data URLs for both the standalone heatmap and the radiograph overlay.
        """
        # Ensure Chest X-ray model is loaded
        if not chest_xray_service.is_ready():
            if chest_xray_service.is_model_trained():
                chest_xray_service.load_model()

        if not chest_xray_service.is_ready() or chest_xray_service._model is None:
            return {
                "status": "not_trained",
                "message": "ResNet-50 model is unconfigured or not loaded.",
                "overlay_preview": None,
                "heatmap_preview": None,
                "target_layer": self.target_layer_name,
                "attribution_method": "Grad-CAM (Gradient-Weighted Class Activation Mapping)",
            }

        if matplotlib is None:
            return {
                "status": "error",
                "message": "Matplotlib is required for Grad-CAM colormap rendering.",
                "overlay_preview": None,
                "heatmap_preview": None,
            }

        try:
            # 1. Decode image
            pil_image = chest_xray_service.process_image_bytes(file_bytes, filename)
            orig_size = pil_image.size # (width, height)

            # 2. Prepare tensor
            if chest_xray_service._transform is None:
                raise RuntimeError("Transform pipeline uninitialized.")

            tensor = chest_xray_service._transform(pil_image).unsqueeze(0).to(chest_xray_service.device)
            tensor.requires_grad = True

            model = chest_xray_service._model
            model.eval()

            # 3. Register forward and backward hooks on target layer
            activations = []
            gradients = []

            def forward_hook(module, input, output):
                activations.append(output)

            def backward_hook(module, grad_in, grad_out):
                gradients.append(grad_out[0])

            target_layer = getattr(model, self.target_layer_name)
            f_handle = target_layer.register_forward_hook(forward_hook)
            b_handle = target_layer.register_full_backward_hook(backward_hook)

            # 4. Forward pass
            model.zero_grad()
            logits = model(tensor)
            logit_val = logits[0, 0]
            prob = float(torch.sigmoid(logit_val).item())

            # 5. Backward pass on class-relevant output score
            # When prob >= 0.5 (Lung Opacity), backpropagate logit_val
            # When prob < 0.5 (Normal), backpropagate -logit_val to capture features driving normal classification
            target_score = logit_val if prob >= 0.5 else -logit_val
            target_score.backward()

            # Remove hook handles to prevent memory leak
            f_handle.remove()
            b_handle.remove()

            if not activations or not gradients:
                raise RuntimeError("Failed to capture feature activations or gradients from target layer.")

            act = activations[0].detach()  # [1, 2048, H, W]
            grad = gradients[0].detach()    # [1, 2048, H, W]

            # 6. Global average pooling on gradients -> channel importance weights
            weights = torch.mean(grad, dim=[2, 3], keepdim=True)  # [1, 2048, 1, 1]
            cam = torch.sum(weights * act, dim=1, keepdim=True)    # [1, 1, H, W]
            cam = torch.relu(cam).squeeze().cpu().numpy()          # [H, W]

            # 7. Normalize CAM to [0.0, 1.0]
            if cam.max() > cam.min():
                cam_normalized = (cam - cam.min()) / (cam.max() - cam.min())
            else:
                cam_normalized = np.zeros_like(cam, dtype=np.float32)

            # 8. Upsample CAM to original image dimensions
            orig_rgb = pil_image.convert("RGB")
            cam_pil = Image.fromarray((cam_normalized * 255).astype(np.uint8)).resize(
                orig_size, resample=Image.Resampling.BILINEAR
            )
            cam_resized = np.array(cam_pil).astype(np.float32) / 255.0

            # 9. Apply Jet colormap
            cmap = matplotlib.colormaps["jet"]
            heatmap_rgba = cmap(cam_resized)  # [H, W, 4] in float [0, 1]
            heatmap_rgb = (heatmap_rgba[:, :, :3] * 255).astype(np.uint8)
            heatmap_pil = Image.fromarray(heatmap_rgb)

            # 10. Blend overlay: alpha * heatmap + (1 - alpha) * original (exact matching dimensions)
            overlay_pil = Image.blend(orig_rgb, heatmap_pil, alpha=alpha)

            # 11. Encode to base64 JPEG data URLs (RGB safe, identical bounding box)
            def to_data_url(img: Image.Image, max_dim: int = 512) -> str:
                thumb = img.convert("RGB")
                thumb.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
                buf = io.BytesIO()
                thumb.save(buf, format="JPEG", quality=88)
                b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
                return f"data:image/jpeg;base64,{b64}"

            overlay_url = to_data_url(overlay_pil)
            heatmap_url = to_data_url(heatmap_pil)
            orig_url = to_data_url(orig_rgb)

            predicted_class = "Lung Opacity / Pneumonia" if prob >= 0.5 else "Normal / No Lung Opacity"

            return {
                "status": "success",
                "attribution_method": "Grad-CAM (Layer4 Feature Saliency)",
                "target_layer": f"ResNet-50 {self.target_layer_name} (Bottleneck Conv)",
                "prediction": predicted_class,
                "confidence": round(prob, 4),
                "threshold": 0.5,
                "filename": filename,
                "original_preview": orig_url,
                "overlay_preview": overlay_url,
                "heatmap_preview": heatmap_url,
                "explanation": (
                    "Warm red and yellow regions highlight localized anatomical zones driving the "
                    f"ResNet-50 prediction towards '{predicted_class}'. Cooler blue regions indicate "
                    "neutral background or normal parenchymal tissue."
                ),
                "disclaimer": (
                    "Explainable AI visual saliency maps are research visualizations of CNN gradient flow "
                    "and do not constitute an anatomical or radiological boundary diagnosis."
                ),
            }

        except Exception as exc:
            logger.error(f"[XAIService] Grad-CAM generation failed: {exc}", exc_info=True)
            return {
                "status": "error",
                "message": f"Grad-CAM computation failed: {str(exc)}",
                "overlay_preview": None,
                "heatmap_preview": None,
                "target_layer": self.target_layer_name,
                "attribution_method": "Grad-CAM",
            }


xai_service = XAIService()
