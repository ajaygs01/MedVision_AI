import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

_project_root = Path(__file__).resolve().parents[3]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import numpy as np
import pandas as pd
from PIL import Image

try:
    import torch
    from torch.utils.data import Dataset
    from torchvision import transforms
except ImportError:
    torch = None
    Dataset = object
    transforms = None

try:
    import pydicom
except ImportError:
    pydicom = None

from ml.chest_xray.cnn.config import CNNConfig, default_config

logger = logging.getLogger("medivision.ml.chest_xray")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def read_dicom_or_image(file_path: Union[str, Path]) -> Image.Image:
    """
    Safely reads either a DICOM (.dcm) file or standard image (.png, .jpg, .jpeg)
    and returns a 3-channel RGB PIL Image normalized to 8-bit [0, 255].
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    suffix = file_path.suffix.lower()

    if suffix == ".dcm":
        if pydicom is None:
            raise ImportError("pydicom is required to decode DICOM images. Install via `pip install pydicom`.")
        dicom_data = pydicom.dcmread(str(file_path))
        pixel_array = dicom_data.pixel_array.astype(np.float32)

        # Handle DICOM Photometric Interpretation (Invert MONOCHROME1 where 0 is brightest)
        if getattr(dicom_data, "PhotometricInterpretation", "") == "MONOCHROME1":
            pixel_array = np.max(pixel_array) - pixel_array

        # Min-Max normalization to [0, 255] uint8
        p_min, p_max = np.min(pixel_array), np.max(pixel_array)
        if p_max > p_min:
            norm_array = ((pixel_array - p_min) / (p_max - p_min) * 255.0).astype(np.uint8)
        else:
            norm_array = np.zeros_like(pixel_array, dtype=np.uint8)

        # Convert 2D grayscale to 3-channel RGB
        return Image.fromarray(norm_array).convert("RGB")
    else:
        # Standard image reading via PIL
        img = Image.open(file_path)
        return img.convert("RGB")


def get_transforms(img_size: Tuple[int, int] = (224, 224), is_train: bool = True, **kwargs):
    """
    Standard torchvision transform pipeline for ResNet-50.
    """
    if "is_training" in kwargs:
        is_train = kwargs["is_training"]

    if transforms is None:
        return None

    if is_train:
        return transforms.Compose([
            transforms.Resize(img_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])


class RSNAPneumoniaDataset(Dataset):
    """
    PyTorch Dataset for the RSNA Pneumonia Detection Challenge.
    Expects patientId and binary Target (0 = Normal, 1 = Lung Opacity).
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        images_dir: Union[str, Path],
        transform=None,
    ):
        self.df = dataframe.reset_index(drop=True)
        self.images_dir = Path(images_dir)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        patient_id = str(row["patientId"])
        target = float(row["Target"])

        # RSNA images are stored as <patientId>.dcm (or fallback to .png / .jpg if pre-converted)
        dcm_path = self.images_dir / f"{patient_id}.dcm"
        png_path = self.images_dir / f"{patient_id}.png"
        jpg_path = self.images_dir / f"{patient_id}.jpg"

        if dcm_path.exists():
            image_path = dcm_path
        elif png_path.exists():
            image_path = png_path
        elif jpg_path.exists():
            image_path = jpg_path
        else:
            raise FileNotFoundError(f"Image for patient {patient_id} not found in {self.images_dir}")

        image = read_dicom_or_image(image_path)

        if self.transform is not None:
            image_tensor = self.transform(image)
        else:
            if transforms is not None:
                image_tensor = transforms.ToTensor()(image)
            else:
                image_tensor = np.array(image)

        if torch is not None:
            target_tensor = torch.tensor(target, dtype=torch.float32)
        else:
            target_tensor = target

        return image_tensor, target_tensor, patient_id


def load_and_split_rsna_metadata(
    labels_csv_path: Union[str, Path],
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    test_ratio: float = 0.10,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, float]:
    """
    Reads RSNA labels CSV, collapses multiple bounding boxes per patient,
    and performs patient-level stratified split (guaranteeing zero data leakage).
    Returns (train_df, val_df, test_df, pos_weight).
    """
    from sklearn.model_selection import train_test_split

    if hasattr(labels_csv_path, "labels_csv_path"):
        labels_csv_path = labels_csv_path.labels_csv_path

    df = pd.read_csv(labels_csv_path)

    if "patientId" not in df.columns or "Target" not in df.columns:
        raise ValueError("CSV must contain 'patientId' and 'Target' columns.")

    # Group by unique patientId: Target=1 if any bounding box was annotated for this patient
    patient_df = df.groupby("patientId")["Target"].max().reset_index()

    # Stratified split: Train vs Temp (Val + Test)
    temp_ratio = val_ratio + test_ratio
    train_df, temp_df = train_test_split(
        patient_df,
        test_size=temp_ratio,
        stratify=patient_df["Target"],
        random_state=seed,
    )

    # Stratified split: Val vs Test
    val_test_proportion = test_ratio / temp_ratio
    val_df, test_df = train_test_split(
        temp_df,
        test_size=val_test_proportion,
        stratify=temp_df["Target"],
        random_state=seed,
    )

    # Calculate class imbalance weighting for BCE loss: num_neg / num_pos
    num_pos = train_df["Target"].sum()
    num_neg = len(train_df) - num_pos
    pos_weight = float(num_neg / max(num_pos, 1))

    logger.info(
        f"[RSNA Dataset Split] Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)} | PosWeight: {pos_weight:.2f}"
    )

    return train_df, val_df, test_df, pos_weight


def verify_dataset_presence(config: Optional[CNNConfig] = None) -> Tuple[bool, str]:
    """
    Checks if the official RSNA dataset files exist locally.
    """
    cfg = config or default_config
    if not cfg.labels_csv_path.exists():
        return (
            False,
            f"Labels CSV missing at '{cfg.labels_csv_path}'. Please review datasets/README.md for download instructions.",
        )
    if not cfg.images_dir_path.exists():
        return (
            False,
            f"Images directory missing at '{cfg.images_dir_path}'. Please review datasets/README.md for download instructions.",
        )
    return True, f"RSNA dataset verified at '{cfg.dataset_dir}'."
