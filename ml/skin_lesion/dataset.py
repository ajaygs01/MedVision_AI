import os
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from sklearn.model_selection import train_test_split

from ml.skin_lesion.config import SkinLesionConfig, CLASS_TO_IDX


def validate_dataset_presence(config: SkinLesionConfig) -> Dict[str, Any]:
    """
    Validates whether the HAM10000 dataset is present locally with both
    metadata CSV and image files.
    """
    meta_path = config.get_metadata_path()
    img_dir = config.get_images_dir()

    if not meta_path.exists():
        return {
            "available": False,
            "total_images": 0,
            "message": f"Dataset metadata missing at {meta_path}.",
        }

    if not img_dir.exists() or not img_dir.is_dir():
        return {
            "available": False,
            "total_images": 0,
            "message": f"Dataset image directory missing at {img_dir}.",
        }

    # Count image files
    image_files = [f for f in img_dir.iterdir() if f.suffix.lower() in [".jpg", ".jpeg", ".png"]]
    if len(image_files) == 0:
        return {
            "available": False,
            "total_images": 0,
            "message": f"No valid images (.jpg/.png) found in {img_dir}.",
        }

    return {
        "available": True,
        "total_images": len(image_files),
        "message": f"Found {len(image_files)} dermatoscopy images and metadata at {meta_path}.",
    }


def split_lesions_zero_leakage(
    df: pd.DataFrame, val_ratio: float = 0.2, seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits dataframe strictly by unique `lesion_id`.
    Zero data leakage: No lesion or patient will ever appear in both train and validation sets.
    """
    if "lesion_id" not in df.columns:
        raise ValueError("DataFrame must contain 'lesion_id' column for leakage-free splitting.")

    # Group by lesion_id and determine the dominant diagnosis for stratified sampling
    lesion_groups = df.groupby("lesion_id").agg({
        "dx": "first",
        "image_id": "count"
    }).reset_index()

    # If any class has only 1 lesion, fallback to simple random split for those rare classes
    class_counts = lesion_groups["dx"].value_counts()
    stratifiable = lesion_groups["dx"].map(lambda x: class_counts[x] > 1)

    if stratifiable.all():
        train_lesions, val_lesions = train_test_split(
            lesion_groups["lesion_id"],
            test_size=val_ratio,
            random_state=seed,
            stratify=lesion_groups["dx"],
        )
    else:
        train_lesions, val_lesions = train_test_split(
            lesion_groups["lesion_id"],
            test_size=val_ratio,
            random_state=seed,
        )

    train_set = set(train_lesions)
    val_set = set(val_lesions)

    train_df = df[df["lesion_id"].isin(train_set)].reset_index(drop=True)
    val_df = df[df["lesion_id"].isin(val_set)].reset_index(drop=True)

    return train_df, val_df


def get_skin_transforms(image_size: int = 224, is_train: bool = True) -> transforms.Compose:
    """
    Standard preprocessing transforms for 224x224 RGB dermoscopy images.
    """
    if is_train:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])


class HAM10000Dataset(Dataset):
    """
    PyTorch Dataset for HAM10000 / ISIC dermatoscopy classification.
    """
    def __init__(self, df: pd.DataFrame, images_dir: Path, transform: Optional[transforms.Compose] = None):
        self.df = df.reset_index(drop=True)
        self.images_dir = Path(images_dir)
        self.transform = transform or get_skin_transforms(is_train=False)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]
        image_id = row["image_id"]
        # Match image with or without extension
        img_filename = f"{image_id}.jpg"
        img_path = self.images_dir / img_filename
        if not img_path.exists():
            img_path = self.images_dir / f"{image_id}.png"

        if not img_path.exists():
            raise FileNotFoundError(f"Dermoscopy image not found: {img_path}")

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        dx = row["dx"]
        label = CLASS_TO_IDX.get(dx, 0)
        return image, label
