"""
Configuration constants for YOLO Anatomical Lesion Localization.
"""

from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASETS_DIR = PROJECT_ROOT / "datasets"
RSNA_DIR = DATASETS_DIR / "rsna_pneumonia"
RSNA_CSV_PATH = RSNA_DIR / "stage_2_train_labels.csv"
RSNA_IMAGES_DIR = RSNA_DIR / "stage_2_train_images"

# Converted YOLO dataset directory (distinct from original RSNA dataset)
YOLO_DATASET_DIR = DATASETS_DIR / "yolo_rsna"
YOLO_DATA_YAML = YOLO_DATASET_DIR / "data.yaml"

# Artifact and checkpoint paths
ARTIFACTS_DIR = PROJECT_ROOT / "backend" / "model_artifacts" / "yolo_detection"
CHECKPOINT_PATH = ARTIFACTS_DIR / "yolov8_best.pt"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"

# Model hyperparameters and target classes
MODEL_ARCH = "YOLOv8n"
DEFAULT_CONF_THRESHOLD = 0.25
DEFAULT_IOU_THRESHOLD = 0.45
CLASSES = ["lung_opacity"]
NUM_CLASSES = len(CLASSES)
DEFAULT_IMGSZ = 512
DEFAULT_BATCH_SIZE = 8
DEFAULT_EPOCHS = 10
DEFAULT_WORKERS = 4
