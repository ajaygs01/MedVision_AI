import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

_project_root = Path(__file__).resolve().parents[3]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))


@dataclass
class CNNConfig:
    """Hyperparameters and paths for Chest X-ray ResNet50 classifier."""

    # Project root is 4 levels up from this file (ml/chest_xray/cnn/config.py -> root)
    project_root: Path = Path(__file__).resolve().parents[3]

    # Dataset paths
    dataset_dir: Path = project_root / "datasets" / "rsna_pneumonia"
    labels_csv_name: str = "stage_2_train_labels.csv"
    detailed_info_csv_name: str = "stage_2_detailed_class_info.csv"
    images_dir_name: str = "stage_2_train_images"

    # Artifact paths
    artifacts_dir: Path = project_root / "model_artifacts" / "chest_xray_cnn"
    checkpoint_filename: str = "best_model.pth"
    metrics_filename: str = "metrics.json"

    # Preprocessing and model parameters
    img_size: Tuple[int, int] = (224, 224)
    batch_size: int = 32
    num_workers: int = 0  # 0 is safe for Windows multiprocessing
    num_epochs: int = 10
    learning_rate: float = 1e-4
    weight_decay: float = 1e-2
    seed: int = 42

    # Patient-level dataset split proportions
    train_ratio: float = 0.80
    val_ratio: float = 0.10
    test_ratio: float = 0.10

    # Early stopping
    early_stopping_patience: int = 3

    # Device
    device_override: str = ""

    @property
    def labels_csv_path(self) -> Path:
        return self.dataset_dir / self.labels_csv_name

    @property
    def images_dir_path(self) -> Path:
        return self.dataset_dir / self.images_dir_name

    checkpoint_path_override: Optional[Path] = None

    @property
    def checkpoint_path(self) -> Path:
        if self.checkpoint_path_override is not None:
            return self.checkpoint_path_override

        backend_ckpt = self.project_root / "backend" / "model_artifacts" / "chest_xray" / "resnet50_best.pth"
        root_ckpt = self.artifacts_dir / self.checkpoint_filename

        # If custom artifacts_dir was provided, respect it
        default_artifacts = self.project_root / "model_artifacts" / "chest_xray_cnn"
        if self.artifacts_dir != default_artifacts:
            return root_ckpt

        if backend_ckpt.exists() and backend_ckpt.stat().st_size > 0:
            return backend_ckpt
        if root_ckpt.exists() and root_ckpt.stat().st_size > 0:
            return root_ckpt
        return backend_ckpt

    @property
    def metrics_path(self) -> Path:
        return self.artifacts_dir / self.metrics_filename

    def get_device(self) -> str:
        if self.device_override:
            return self.device_override
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    def ensure_artifact_dir(self) -> None:
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)


default_config = CNNConfig()
