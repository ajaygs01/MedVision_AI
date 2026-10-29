"""
Configuration parameters for Module 2: Brain MRI Lesion & Tumor Segmentation.
"""

from pathlib import Path
import sys

_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

DATASET_DIR = _project_root / "datasets" / "brats"
ARTIFACTS_DIR = _project_root / "backend" / "model_artifacts" / "brain_mri"
PILOT_ARTIFACTS_DIR = ARTIFACTS_DIR / "pilot"

CHECKPOINT_BEST_PATH = ARTIFACTS_DIR / "unet3d_best.pth"
CHECKPOINT_LAST_PATH = ARTIFACTS_DIR / "last_checkpoint.pth"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"

CHECKPOINT_PILOT_PATH = PILOT_ARTIFACTS_DIR / "unet3d_pilot.pth"
CHECKPOINT_PILOT_LAST_PATH = PILOT_ARTIFACTS_DIR / "last_checkpoint.pth"
METRICS_PILOT_PATH = PILOT_ARTIFACTS_DIR / "metrics_pilot.json"

# Modality sequencing (BraTS standard)
MODALITIES = ["flair", "t1", "t1ce", "t2"]
IN_CHANNELS = 4
NUM_CLASSES = 4

# Class label definitions
LABEL_BACKGROUND = 0
LABEL_NCR_NET = 1    # Necrotic and non-enhancing tumor core
LABEL_EDEMA = 2      # Peritumoral edema
LABEL_ENHANCING = 3  # Enhancing tumor (remapped from BraTS label 4 to 3)

CLASS_NAMES = {
    0: "Background / Normal Tissue",
    1: "Necrotic & Non-Enhancing Core (NCR/NET)",
    2: "Peritumoral Edema (ED)",
    3: "Active Enhancing Tumor (ET)",
}

# Subregion clinical composite groupings
SUBREGION_NAMES = {
    "WT": "Whole Tumor (WT: NCR/NET + ED + ET)",
    "TC": "Tumor Core (TC: NCR/NET + ET)",
    "ET": "Enhancing Tumor (ET)",
}

# Default training hyperparameters (CPU-safe defaults)
DEFAULT_BATCH_SIZE = 1
DEFAULT_EPOCHS = 10
DEFAULT_LR = 1e-3
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_PATCH_SIZE = (64, 128, 128)  # Depth, Height, Width
RANDOM_SEED = 42

TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10
