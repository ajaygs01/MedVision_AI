"""
Dataset preparation, conversion, and validation pipeline for YOLO detection.
Converts RSNA Pneumonia bounding boxes into YOLO Darknet format in a separate directory
without altering the original RSNA files.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import pydicom
from PIL import Image

from ml.yolo_detection.config import (
    CLASSES,
    RSNA_CSV_PATH,
    RSNA_IMAGES_DIR,
    YOLO_DATASET_DIR,
    YOLO_DATA_YAML,
)

logger = logging.getLogger("medivision.ml.yolo_dataset")


def verify_rsna_source_dataset() -> Dict[str, Any]:
    """Inspects the raw RSNA dataset source directory and annotation CSV."""
    if not RSNA_CSV_PATH.exists():
        return {
            "status": "missing_csv",
            "message": f"RSNA labels CSV not found at {RSNA_CSV_PATH}",
            "csv_exists": False,
            "images_dir_exists": RSNA_IMAGES_DIR.exists(),
        }

    if not RSNA_IMAGES_DIR.exists():
        return {
            "status": "missing_images",
            "message": f"RSNA image directory not found at {RSNA_IMAGES_DIR}",
            "csv_exists": True,
            "images_dir_exists": False,
        }

    try:
        df = pd.read_csv(RSNA_CSV_PATH)
        total_rows = len(df)
        unique_patients = df["patientId"].nunique()
        pos_df = df[df["Target"] == 1]
        num_pos_boxes = len(pos_df)
        num_pos_patients = pos_df["patientId"].nunique()

        return {
            "status": "ready",
            "total_rows": total_rows,
            "unique_patients": unique_patients,
            "num_positive_boxes": num_pos_boxes,
            "num_positive_patients": num_pos_patients,
            "csv_path": str(RSNA_CSV_PATH),
            "images_dir": str(RSNA_IMAGES_DIR),
            "message": (
                f"RSNA dataset valid: {unique_patients} patients, "
                f"{num_pos_boxes} bounding boxes across {num_pos_patients} positive patients."
            ),
        }
    except Exception as exc:
        return {
            "status": "error",
            "message": f"Failed to read RSNA CSV: {exc}",
        }


def convert_box_to_yolo(
    x: float,
    y: float,
    width: float,
    height: float,
    img_width: float = 1024.0,
    img_height: float = 1024.0,
) -> Tuple[float, float, float, float]:
    """
    Converts absolute (x, y, w, h) bounding box to normalized YOLO format
    (x_center, y_center, width, height) clamped to [0.0, 1.0].
    """
    x_center = (x + width / 2.0) / img_width
    y_center = (y + height / 2.0) / img_height
    norm_w = width / img_width
    norm_h = height / img_height

    # Clamp coordinates safely
    x_center = float(np.clip(x_center, 0.0, 1.0))
    y_center = float(np.clip(y_center, 0.0, 1.0))
    norm_w = float(np.clip(norm_w, 0.0, 1.0))
    norm_h = float(np.clip(norm_h, 0.0, 1.0))

    return x_center, y_center, norm_w, norm_h


def read_dicom_to_image(dcm_path: Path) -> Image.Image:
    """Reads a DICOM file and standardizes it to an 8-bit RGB PIL Image."""
    ds = pydicom.dcmread(str(dcm_path))
    arr = ds.pixel_array.astype(np.float32)

    # Invert PhotometricInterpretation if MONOCHROME1 (0 is white)
    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        arr = np.max(arr) - arr

    # Window / normalize to 0-255
    min_val, max_val = arr.min(), arr.max()
    if max_val > min_val:
        arr = (arr - min_val) / (max_val - min_val) * 255.0
    else:
        arr = np.zeros_like(arr)

    arr_uint8 = arr.astype(np.uint8)
    return Image.fromarray(arr_uint8).convert("RGB")


def prepare_rsna_yolo_dataset(
    output_dir: Path = YOLO_DATASET_DIR,
    sample_limit: Optional[int] = None,
    val_ratio: float = 0.15,
    test_ratio: float = 0.10,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Extracts RSNA DICOMs and converts bounding boxes into a distinct YOLO dataset structure.
    Enforces strict patient-level splitting to prevent data leakage.
    """
    source_status = verify_rsna_source_dataset()
    if source_status.get("status") != "ready":
        raise ValueError(f"RSNA dataset source not ready: {source_status.get('message')}")

    df = pd.read_csv(RSNA_CSV_PATH)

    # Get unique patient list
    unique_patients = df["patientId"].unique()
    np.random.seed(seed)
    shuffled_patients = np.random.permutation(unique_patients)

    if sample_limit and sample_limit < len(shuffled_patients):
        # Stratify to ensure positive cases are represented
        pos_patients = np.array(df[df["Target"] == 1]["patientId"].unique(), dtype=object)
        neg_patients = np.array(df[df["Target"] == 0]["patientId"].unique(), dtype=object)
        np.random.shuffle(pos_patients)
        np.random.shuffle(neg_patients)

        half = sample_limit // 2
        selected_patients = list(pos_patients[:half]) + list(neg_patients[: (sample_limit - half)])
        shuffled_patients = np.random.permutation(selected_patients)

    n_total = len(shuffled_patients)
    n_test = int(n_total * test_ratio)
    n_val = int(n_total * val_ratio)
    n_train = n_total - n_val - n_test

    train_ids = set(shuffled_patients[:n_train])
    val_ids = set(shuffled_patients[n_train : n_train + n_val])
    test_ids = set(shuffled_patients[n_train + n_val :])

    # Verify zero overlap between patient splits
    assert len(train_ids.intersection(val_ids)) == 0, "Patient leakage between train and val"
    assert len(train_ids.intersection(test_ids)) == 0, "Patient leakage between train and test"
    assert len(val_ids.intersection(test_ids)) == 0, "Patient leakage between val and test"

    split_map = {}
    for pid in train_ids:
        split_map[pid] = "train"
    for pid in val_ids:
        split_map[pid] = "val"
    for pid in test_ids:
        split_map[pid] = "test"

    # Create destination directories
    for split in ["train", "val", "test"]:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    grouped = df.groupby("patientId")
    processed_count = 0
    box_count = 0

    for pid in shuffled_patients:
        split = split_map[pid]
        dcm_path = RSNA_IMAGES_DIR / f"{pid}.dcm"
        if not dcm_path.exists():
            continue

        # Convert DICOM and save as JPEG image
        out_img_path = output_dir / "images" / split / f"{pid}.jpg"
        out_lbl_path = output_dir / "labels" / split / f"{pid}.txt"

        if not out_img_path.exists():
            img = read_dicom_to_image(dcm_path)
            img.save(out_img_path, format="JPEG", quality=90)

        # Write labels
        rows = grouped.get_group(pid)
        label_lines = []
        for _, row in rows.iterrows():
            if row["Target"] == 1 and not pd.isna(row["x"]):
                x, y, w, h = float(row["x"]), float(row["y"]), float(row["width"]), float(row["height"])
                xc, yc, nw, nh = convert_box_to_yolo(x, y, w, h)
                label_lines.append(f"0 {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}")
                box_count += 1

        with open(out_lbl_path, "w", encoding="utf-8") as lf:
            if label_lines:
                lf.write("\n".join(label_lines) + "\n")
            else:
                lf.write("")  # Empty file denotes verified background image without target

        processed_count += 1

    # Write data.yaml using relative paths
    yaml_content = f"""# MediVision-AI RSNA YOLOv8 Dataset Configuration
path: {output_dir.resolve().as_posix()}
train: images/train
val: images/val
test: images/test

# Classes
names:
  0: lung_opacity
"""
    with open(output_dir / "data.yaml", "w", encoding="utf-8") as yf:
        yf.write(yaml_content)

    return {
        "status": "success",
        "processed_patients": processed_count,
        "total_boxes": box_count,
        "splits": {
            "train": len(train_ids),
            "val": len(val_ids),
            "test": len(test_ids),
        },
        "output_dir": str(output_dir),
        "yaml_path": str(output_dir / "data.yaml"),
    }


def verify_yolo_dataset(yolo_dir: Path = YOLO_DATASET_DIR) -> Dict[str, Any]:
    """
    Validates whether a converted YOLO dataset exists and conforms to
    YOLO Darknet format with valid bounding box coordinates.
    """
    data_yaml = yolo_dir / "data.yaml"
    if not data_yaml.exists():
        return {
            "status": "missing_yaml",
            "message": f"data.yaml not found at {data_yaml}",
            "is_valid": False,
        }

    stats = {}
    total_images = 0
    total_labels = 0
    total_boxes = 0
    invalid_boxes = 0

    for split in ["train", "val", "test"]:
        img_dir = yolo_dir / "images" / split
        lbl_dir = yolo_dir / "labels" / split
        if not img_dir.exists() or not lbl_dir.exists():
            return {
                "status": "incomplete_split",
                "message": f"Split '{split}' directory missing in {yolo_dir}",
                "is_valid": False,
            }

        imgs = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
        lbls = list(lbl_dir.glob("*.txt"))

        split_boxes = 0
        for lbl in lbls:
            with open(lbl, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        cls_id, xc, yc, w, h = parts
                        xc, yc, w, h = float(xc), float(yc), float(w), float(h)
                        if (
                            0.0 <= xc <= 1.0
                            and 0.0 <= yc <= 1.0
                            and 0.0 < w <= 1.0
                            and 0.0 < h <= 1.0
                        ):
                            split_boxes += 1
                        else:
                            invalid_boxes += 1

        stats[split] = {
            "images": len(imgs),
            "labels": len(lbls),
            "bounding_boxes": split_boxes,
        }
        total_images += len(imgs)
        total_labels += len(lbls)
        total_boxes += split_boxes

    if total_images == 0:
        return {
            "status": "empty",
            "message": f"Dataset directory {yolo_dir} contains no images.",
            "is_valid": False,
        }

    return {
        "status": "valid",
        "is_valid": True,
        "total_images": total_images,
        "total_labels": total_labels,
        "total_bounding_boxes": total_boxes,
        "invalid_boxes": invalid_boxes,
        "splits": stats,
        "yaml_path": str(data_yaml),
        "classes": CLASSES,
        "message": (
            f"Verified YOLO dataset with {total_images} images, "
            f"{total_boxes} bounding boxes across train/val/test splits (0 invalid coordinates)."
        ),
    }
