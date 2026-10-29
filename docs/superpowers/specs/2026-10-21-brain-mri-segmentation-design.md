# Design Specification: Module 2 - Brain MRI Lesion & Tumor Segmentation

**Author:** DeepMind Antigravity Pair Programming Agent  
**Date:** 2026-10-21  
**Project:** MediVision AI Clinical Research Platform  
**Target Module:** Module 2 — Volumetric Brain MRI Lesion Segmentation  
**Status:** Validated Design Spec (Pre-Implementation)

---

## 1. System Overview & Objective

The objective of Module 2 is to expand the MediVision AI multi-modal clinical platform by integrating volumetric Brain MRI lesion and tumor delineation. The system evaluates multi-parametric Magnetic Resonance Imaging (mpMRI) scans to identify and segment glioblastoma / glioma subregions following the international Brain Tumor Segmentation (BraTS) benchmark standard.

### Core Principles
1. **Zero Fabrication**: No synthetic training data, fabricated segmentation masks, or mock evaluation metrics.
2. **CPU & Accelerator Compatibility**: Training and inference pipelines must function reliably on CPU (Intel UHD 630 / x86_64) while supporting CUDA acceleration when available.
3. **Data Leakage Prevention**: All training, validation, and held-out test splits are partitioned strictly at the unique patient level.
4. **Preserved UI & Multi-Module Persistence**: Retain the existing MediVision blue sidebar, card architecture, and styling. The active MRI study, slice view, and segmented contours must survive navigation across sidebar modules.
5. **Full Lifecycle Auditability**: Every segmented study is assigned a unique case ID, persisted in MongoDB Atlas, and equipped with a publication-grade downloadable PDF clinical report.

---

## 2. Dataset Protocol & Missing Prerequisite Handling

### 2.1 The BraTS Dataset Benchmark Standard
- **Target Dataset**: BraTS (Brain Tumor Segmentation Challenge — BraTS 2020 / 2021 / 2023 / BraTS-GLI).
- **Imaging Sequences (4 Co-registered Channels)**:
  1. **FLAIR**: Fluid Attenuated Inversion Recovery (highlights peritumoral edema)
  2. **T1**: T1-weighted native
  3. **T1ce**: T1-weighted contrast-enhanced with Gadolinium (highlights active vascularized tumor border)
  4. **T2**: T2-weighted (delineates overall edema and fluid cavities)
- **Segmentation Subregions (BraTS Label Hierarchy)**:
  - `Label 0`: Background (healthy brain tissue / CSF / air)
  - `Label 1`: Necrotic and Non-Enhancing Tumor core (NCR/NET)
  - `Label 2`: Peritumoral Edematous tissue (ED)
  - `Label 4` (remapped to `3` in tensor space): GD-Enhancing Tumor (ET)
- **Standard Clinical Composite Evaluation Tiers**:
  - **Whole Tumor (WT)**: Labels 1 + 2 + 4
  - **Tumor Core (TC)**: Labels 1 + 4
  - **Enhancing Tumor (ET)**: Label 4

### 2.2 Expected Directory Layout
```text
MediVision-AI/
└── datasets/
    ├── README.md
    ├── clinical_heart/
    ├── rsna_pneumonia/
    └── brats/
        ├── README.md
        └── BraTS2020_TrainingData/
            ├── BraTS20_Training_001/
            │   ├── BraTS20_Training_001_flair.nii.gz
            │   ├── BraTS20_Training_001_t1.nii.gz
            │   ├── BraTS20_Training_001_t1ce.nii.gz
            │   ├── BraTS20_Training_001_t2.nii.gz
            │   └── BraTS20_Training_001_seg.nii.gz
            └── ...
```

### 2.3 Status When Dataset is Missing
- If `datasets/brats/` is absent or contains no patient directories, the system reports status `dataset_missing` or `not_trained`.
- The CLI command `python -m ml.brain_mri.dataset --verify` inspects directory paths, file integrity, and patient counts.
- The platform provides clear, reproducible setup instructions (Kaggle BraTS dataset or Synapse challenge portal) without pretending that data has been downloaded.

---

## 3. Preprocessing & Input Validation Pipeline

The preprocessing pipeline is encapsulated in `ml/brain_mri/preprocessing.py` and `ml/brain_mri/dataset.py`:

1. **Volume Ingestion**:
   - Loads `.nii` / `.nii.gz` volumes using `nibabel`.
   - Supports ingestion of either 4 separate sequence files (`_flair.nii.gz`, `_t1.nii.gz`, `_t1ce.nii.gz`, `_t2.nii.gz`), a stacked 4D volume $[4, H, W, D]$, or a `.zip` archive containing the study.
2. **Dimension & Alignment Validation**:
   - Asserts all 4 modalities share identical spatial dimensions $(H, W, D)$ (typically $240 \times 240 \times 155$) and matching affine matrices.
   - Flags missing channels or malformed voxel headers.
3. **Non-Zero Tissue Masking & Z-Score Normalization**:
   - Background voxels with value $0$ across all modalities are masked out.
   - For each modality channel $c \in \{0, 1, 2, 3\}$, calculates mean $\mu_c$ and standard deviation $\sigma_c$ across non-zero brain voxels only:
     $$x_{c, norm} = \frac{x_c - \mu_c}{\sigma_c + \epsilon}$$
4. **Spatial Cropping & Patch Resampling**:
   - Crops empty peripheral background to tight bounding box of non-zero tissue.
   - For training: random spatial patch extraction (e.g., $128 \times 128 \times 64$ or $64 \times 128 \times 128$) to fit CPU RAM budgets and prevent out-of-memory errors.
   - For inference: sliding window inference or resampled whole-brain volume inference.

---

## 4. Model Architecture & Training Protocol

### 4.1 PyTorch 3D U-Net Architecture (`ml/brain_mri/model.py`)
A modular 3D Encoder-Decoder U-Net implemented in standard PyTorch with zero heavy external dependencies:
- **Input Channels**: 4 (FLAIR, T1, T1ce, T2).
- **Encoder (Contracting Path)**:
  - Stage 1: Conv3d(4 -> 16, kernel=3, pad=1), InstanceNorm3d, LeakyReLU -> Conv3d(16 -> 16), InstanceNorm3d, LeakyReLU.
  - MaxPool3d(2).
  - Stage 2: Conv3d(16 -> 32) -> Conv3d(32 -> 32), InstanceNorm3d, LeakyReLU.
  - MaxPool3d(2).
  - Stage 3: Conv3d(32 -> 64) -> Conv3d(64 -> 64), InstanceNorm3d, LeakyReLU.
  - MaxPool3d(2).
  - Stage 4: Conv3d(64 -> 128) -> Conv3d(128 -> 128), InstanceNorm3d, LeakyReLU.
- **Bottleneck**:
  - Stage 5: Conv3d(128 -> 256) -> Conv3d(256 -> 256), Dropout3d(0.2), InstanceNorm3d, LeakyReLU.
- **Decoder (Expanding Path)**:
  - ConvTranspose3d(256 -> 128, kernel=2, stride=2) + Concatenation of Stage 4 skip connection -> Conv3d(256 -> 128) -> Conv3d(128 -> 128).
  - ConvTranspose3d(128 -> 64, kernel=2, stride=2) + Stage 3 skip -> Conv3d(128 -> 64) -> Conv3d(64 -> 64).
  - ConvTranspose3d(64 -> 32, kernel=2, stride=2) + Stage 2 skip -> Conv3d(64 -> 32) -> Conv3d(32 -> 32).
  - ConvTranspose3d(32 -> 16, kernel=2, stride=2) + Stage 1 skip -> Conv3d(32 -> 16) -> Conv3d(16 -> 16).
- **Output Head**:
  - Conv3d(16 -> 4, kernel=1) yielding class logits for `[Background, NCR/NET, ED, ET]`.

### 4.2 Loss Function & Metrics
- **Generalized Dice + Cross-Entropy Loss**:
  $$\mathcal{L}_{total} = \mathcal{L}_{Dice} + \mathcal{L}_{CE}$$
  where Dice is computed across one-hot class channels with smooth factor $\epsilon=1.0$.
- **Validation & Test Metrics**:
  - **Whole Tumor (WT) Dice Score**: $\text{DSC}(P_{WT}, G_{WT})$
  - **Tumor Core (TC) Dice Score**: $\text{DSC}(P_{TC}, G_{TC})$
  - **Enhancing Tumor (ET) Dice Score**: $\text{DSC}(P_{ET}, G_{ET})$
  - **Mean Dice**: Average of WT, TC, and ET.
  - Volumetric measurements: Predicted lesion volumes in cubic centimeters ($\text{cm}^3$).

### 4.3 Training & Checkpointing Lifecycle (`ml/brain_mri/train.py`)
- **Device Support**: `--device cpu` or `--device cuda`.
- **Optimizer**: AdamW ($\text{lr}=10^{-3}$, weight decay $10^{-4}$) with CosineAnnealingLR.
- **Patient-Level Split**: 80% train, 10% validation, 10% test with deterministic random seed.
- **Checkpointing**:
  - `backend/model_artifacts/brain_mri/unet3d_best.pth`: Stores `model_state_dict`, `optimizer_state_dict`, `epoch`, `best_val_dice`, and `config`.
  - `backend/model_artifacts/brain_mri/last_checkpoint.pth`: Facilitates `--resume` for interrupted training sessions.
  - `backend/model_artifacts/brain_mri/metrics.json`: Records training history, validation curves, and test split Dice scores.

---

## 5. Backend FastAPI Architecture & Services

### 5.1 Service Layer (`backend/app/services/brain_mri.py`)
- Handles model lifecycle: loads checkpoint from `backend/model_artifacts/brain_mri/unet3d_best.pth` on startup.
- `is_ready()`: Returns `True` only when valid weights are loaded.
- `validate_mri_study()`: Validates uploaded file headers, dimensions, and sequences.
- `segment_study()`: Executes 3D U-Net forward pass, reconstructs volumetric masks, extracts axial/coronal/sagittal slice previews with overlay, and calculates tumor volumes in $\text{cm}^3$.

### 5.2 API Router (`backend/app/api/endpoints/brain_mri.py`)
1. `GET /api/brain-mri/status`:
   - Returns status (`ready`, `not_trained`, `dataset_missing`), model path, device (`cpu`/`cuda`), test Dice metrics if available, and prerequisite checklist.
2. `POST /api/brain-mri/validate`:
   - Multipart file upload validation for `.nii`, `.nii.gz`, or `.zip` study archives.
3. `POST /api/brain-mri/segment`:
   - Executes volumetric segmentation inference.
   - Automatically persists telemetry record to MongoDB Atlas / Local MongoDB.
   - Returns segmentation results, volumetric tumor breakdown, slice previews (base64), and unique database record ID.

### 5.3 PDF Report Generation (`backend/app/services/reports.py`)
- Extends `generate_analysis_pdf()` with a dedicated `modality == "brain_mri"` section:
  - Embeds axial slice views: Original MRI (FLAIR / T1ce) alongside Color-Coded Lesion Segmentation Overlay.
  - Subregion Volume Breakdown Table: Whole Tumor, Tumor Core, and Enhancing Tumor volumes in $\text{cm}^3$.
  - Research disclaimer, patient ID, timestamp, and model architecture metadata.

---

## 6. Frontend Workbench (`BrainMRIWorkbench.tsx`)

### 6.1 UI & Layout Compliance
- Replaces the placeholder blueprint in `ModuleWorkbench.tsx` when `tabId === 'brain-mri'`.
- Preserves the existing blue sidebar, header metrics, and card design.
- If the model is not yet trained or the dataset is missing:
  - Displays an informative operational banner with the prerequisite status.
  - Outlines exact download and training CLI commands.
  - Allows testing pre-processed sample MRI cases when available or uploads to validate NIfTI format.

### 6.2 Key UI Components
1. **Status & Model Benchmark Banner**:
   - Shows live model status (`OPERATIONAL • 3D U-Net` or `DATASET REQUIRED`).
   - Displays held-out test metrics: WT Dice, TC Dice, ET Dice when trained.
2. **Multi-Sequence Study Uploader**:
   - Drag-and-drop zone accepting `.nii.gz` or `.zip` MRI studies.
   - Sequence checklist validating FLAIR, T1, T1ce, T2.
3. **Interactive 2.5D Slice Viewer**:
   - Plane selector: Axial, Coronal, Sagittal.
   - Interactive Slice Slider navigating through brain volume depth.
   - View mode toggle: Original MRI, Segmentation Mask Only, or Alpha-blended Overlay.
4. **Volumetric Quantitative Lesion Card**:
   - Whole Tumor Volume ($\text{cm}^3$) with emerald indicator.
   - Tumor Core Volume ($\text{cm}^3$) with amber indicator.
   - Enhancing Lesion Volume ($\text{cm}^3$) with rose indicator.
5. **PDF Report Download & History Integration**:
   - Visible "Download Patient Report (PDF)" button.
   - Integrates with `AnalysisContext` via `activeBrainMRICase` to preserve state across tab switches.

---

## 7. Verification & Test Plan

1. **Unit & Pipeline Tests (`backend/tests/test_brain_mri.py`)**:
   - Test 1: Model instantiation and 3D tensor shape integrity ($[1, 4, 32, 64, 64] \to [1, 4, 32, 64, 64]$).
   - Test 2: Dice and Cross-Entropy loss computation with known target tensors.
   - Test 3: NIfTI parsing, z-score normalization, and patient-level split logic.
   - Test 4: Model status endpoint behavior with missing vs. present checkpoint.
   - Test 5: Rejection of malformed / invalid upload payloads.
   - Test 6: Database history logging and retrieval of brain MRI cases.
   - Test 7: PDF clinical report synthesis for brain MRI records.
2. **Regression & Fast Refresh Verification**:
   - Ensure all existing 38 backend tests for Chest X-ray, Grad-CAM, and Clinical ML continue to pass (100%).
   - Frontend lint: `oxlint` with 0 warnings, 0 errors.
   - Frontend production build: `npm run build` (`tsc -b && vite build`) exit code 0.
