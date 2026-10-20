# RSNA Pneumonia Detection Challenge Dataset Setup

This directory contains instructions for downloading and configuring the official **RSNA Pneumonia Detection Challenge** dataset for **MediVision AI Stage 2: CNN Chest X-ray Classification**.

> **Note on Data Access & Security:**  
> In accordance with medical data governance and competition terms, dataset files must not be committed to Git. Never download restricted or licensed patient data without authorization.

---

## 1. Official Dataset Source

- **Challenge Platform:** Kaggle
- **Competition URL:** [https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge/data](https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge/data)
- **Modality:** Chest Radiographs (DICOM `.dcm` format)
- **Resolution:** 1024 x 1024 pixels, 12-bit / 8-bit grayscale

---

## 2. Directory Structure

Download and extract the competition archive so that files are placed into `datasets/rsna_pneumonia/` following this structure:

```text
MediVision-AI/
└── datasets/
    ├── README.md
    └── rsna_pneumonia/
        ├── stage_2_train_labels.csv           # Bounding boxes and binary Target (0 or 1)
        ├── stage_2_detailed_class_info.csv    # Class labels: Normal, No Lung Opacity / Not Normal, Lung Opacity
        └── stage_2_train_images/              # DICOM images (.dcm)
            ├── 0004cfab-14fd-4e49-80ba-63a80b6bddd6.dcm
            ├── 00313ee0-9ea5-44f1-9f85-fa9695819770.dcm
            └── ...
```

---

## 3. CSV Schema & Patient ID Handling

### `stage_2_train_labels.csv`
| Column | Type | Description |
|---|---|---|
| `patientId` | string | Unique alphanumeric identifier for each patient |
| `x` | float / empty | X-coordinate of top-left corner of opacity bounding box (null if Target=0) |
| `y` | float / empty | Y-coordinate of top-left corner of opacity bounding box (null if Target=0) |
| `width` | float / empty | Width of bounding box |
| `height` | float / empty | Height of bounding box |
| `Target` | int | Binary indicator: `1` = Lung Opacity (Pneumonia), `0` = Normal / No Lung Opacity |

### Critical Loader Notes:
1. **Multiple Rows per Patient:** Patients with multiple lung opacity findings have **multiple rows** in `stage_2_train_labels.csv` with the same `patientId`. The Stage 2 CNN loader groups by `patientId` and assigns `Target = max(Target)` to yield a single binary classification label per image.
2. **Patient-Level Separation:** Splitting between training (80%), validation (10%), and test (10%) splits is performed strictly by unique `patientId`. Images from the same patient NEVER appear in multiple splits, eliminating data leakage.

---

## 4. How to Download via Kaggle CLI

```powershell
# 1. Install Kaggle CLI (if not installed)
pip install kaggle

# 2. Ensure your kaggle.json API token is placed in:
# C:\Users\<YourUsername>\.kaggle\kaggle.json

# 3. Download competition files:
kaggle competitions download -c rsna-pneumonia-detection-challenge -p datasets/rsna_pneumonia/

# 4. Unzip files inside datasets/rsna_pneumonia/:
Expand-Archive -Path datasets/rsna_pneumonia/rsna-pneumonia-detection-challenge.zip -DestinationPath datasets/rsna_pneumonia/
```

---

## 5. Verification Command

Once the dataset is in place, verify the dataset loader by running:

```powershell
cd C:\Users\gsaja\OneDrive\Desktop\MediVision-AI\backend
.\.venv\Scripts\Activate.ps1
python -m ml.chest_xray.cnn.dataset --verify
```
