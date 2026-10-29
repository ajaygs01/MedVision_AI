import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict

HAM10000_CLASSES: List[str] = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]

CLASS_NAMES: Dict[str, str] = {
    "MEL": "Melanoma (Malignant)",
    "NV": "Melanocytic Nevus (Benign Mole)",
    "BCC": "Basal Cell Carcinoma (Malignant)",
    "AKIEC": "Actinic Keratosis / Intraepithelial Carcinoma (Pre-cancerous)",
    "BKL": "Benign Keratosis (Solar Lentigo / Seborrheic)",
    "DF": "Dermatofibroma (Benign)",
    "VASC": "Vascular Lesion (Angioma / Hemangioma)",
}

CLASS_TO_IDX: Dict[str, int] = {cls: idx for idx, cls in enumerate(HAM10000_CLASSES)}
IDX_TO_CLASS: List[str] = HAM10000_CLASSES


@dataclass
class SkinLesionConfig:
    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parents[2])
    dataset_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parents[2] / "datasets" / "ham10000")
    artifacts_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parents[2] / "backend" / "model_artifacts" / "skin_lesion")
    
    # Image & Training Hyperparameters (optimized for CPU execution and <=2GB free RAM)
    image_size: int = 224
    batch_size: int = 2
    num_workers: int = 0
    lr: float = 1e-4
    epochs: int = 5
    num_classes: int = 7
    device: str = "cpu"

    # Preprocessing constants (ImageNet standard)
    mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    std: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])

    def get_images_dir(self) -> Path:
        return self.dataset_dir / "images"

    def get_metadata_path(self) -> Path:
        return self.dataset_dir / "HAM10000_metadata.csv"

    def get_checkpoint_path(self) -> Path:
        return self.artifacts_dir / "efficientnet_best.pth"

    def get_metrics_path(self) -> Path:
        return self.artifacts_dir / "metrics.json"
