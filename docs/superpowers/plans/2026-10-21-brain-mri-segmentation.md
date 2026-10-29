# Brain MRI Lesion & Tumor Segmentation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement Module 2: Brain MRI Lesion & Tumor Segmentation in MediVision AI, featuring genuine 3D U-Net architecture, BraTS multi-parametric MRI (FLAIR, T1, T1ce, T2) preprocessing, patient-level data splits, training & evaluation CLI with checkpoint resume, FastAPI endpoints, MongoDB Atlas history persistence, PDF report synthesis, and an interactive multi-planar slice viewer frontend workbench with cross-module state preservation.

**Architecture:** A standalone PyTorch 3D U-Net with InstanceNorm3d, Generalized Dice + Cross-Entropy loss, and patient-level splitting. A FastAPI backend service handling study validation, inference, volumetric tumor quantification ($\text{cm}^3$), and MongoDB persistence. A React frontend workbench with interactive slice navigation (Axial/Coronal/Sagittal), overlay blending, and sessionStorage-backed state persistence.

**Tech Stack:** Python 3.14 / PyTorch 2.x, Nibabel, NumPy, Scikit-Learn, FastAPI, Motor/MongoDB, ReportLab, React 19, TypeScript, Tailwind CSS, Lucide React.

**Spec:** [`docs/superpowers/specs/2026-10-21-brain-mri-segmentation-design.md`](file:///c:/Users/gsaja/OneDrive/Desktop/MediVision-AI/docs/superpowers/specs/2026-10-21-brain-mri-segmentation-design.md)

## Global Constraints
- Do not create a new project or redesign the existing UI; preserve the blue sidebar, layout, and color palette.
- Do not break Chest X-ray ResNet-50, Grad-CAM, Clinical ML, MongoDB history, or PDF report downloads.
- Real ML only: Never fabricate synthetic patient volumes, fake segmentation masks, or mock evaluation metrics.
- CPU compatibility: All pipelines must run reliably on CPU without requiring CUDA/GPU.
- Patient-level splitting: Images/volumes from the same patient must never cross train/validation/test splits.

---

### Task 1: BraTS Dataset Verification CLI & Setup Guide

**Files:**
- Create: `ml/brain_mri/__init__.py`
- Create: `ml/brain_mri/dataset.py`
- Create: `datasets/brats/README.md`
- Test: `ml/brain_mri/dataset.py --verify`

**Interfaces:**
- Produces: `verify_brats_dataset(data_dir: Path) -> Dict[str, Any]` returning status (`ready` | `dataset_missing`), patient count, sequence checks, and resolution instructions.
- CLI: `python -m ml.brain_mri.dataset --verify`

- [ ] **Step 1: Write `ml/brain_mri/__init__.py`**
- [ ] **Step 2: Implement `ml/brain_mri/dataset.py` with dataset scanner and verification CLI**
- [ ] **Step 3: Create `datasets/brats/README.md` with reproducible download guide**
- [ ] **Step 4: Run verification command to test dataset verification output**
  Run: `.\backend\.venv\Scripts\python.exe -m ml.brain_mri.dataset --verify`
  Expected: Reports `dataset_missing` with clear instructions when dataset directory is empty.

---

### Task 2: PyTorch 3D U-Net Model, Generalized Dice Loss & Preprocessing

**Files:**
- Create: `ml/brain_mri/model.py`
- Create: `ml/brain_mri/losses.py`
- Create: `ml/brain_mri/preprocessing.py`
- Test: `backend/tests/test_brain_mri_model.py`

**Interfaces:**
- Produces: `UNet3D(in_channels=4, num_classes=4)`
- Produces: `DiceCELoss(smooth=1.0, ce_weight=0.5)`
- Produces: `preprocess_mri_volume(flair, t1, t1ce, t2) -> (tensor, brain_mask)`
- Produces: `get_patient_splits(data_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42)`

- [ ] **Step 1: Write test for 3D U-Net tensor shapes and Dice loss in `backend/tests/test_brain_mri_model.py`**
- [ ] **Step 2: Run test to verify it fails before implementation**
- [ ] **Step 3: Implement `ml/brain_mri/model.py` (3D U-Net architecture)**
- [ ] **Step 4: Implement `ml/brain_mri/losses.py` (Dice + Cross-Entropy loss)**
- [ ] **Step 5: Implement `ml/brain_mri/preprocessing.py` (NIfTI loading, z-score normalization, slice extraction)**
- [ ] **Step 6: Run test to verify 3D U-Net and losses pass**
  Run: `.\backend\.venv\Scripts\python.exe -m pytest backend/tests/test_brain_mri_model.py -v`
  Expected: PASS

---

### Task 3: Training & Evaluation Pipeline with Resume Support

**Files:**
- Create: `ml/brain_mri/train.py`
- Create: `ml/brain_mri/evaluate.py`
- Test: `backend/tests/test_brain_mri_pipeline.py`

**Interfaces:**
- Produces: `train(config: BrainMRITrainConfig)` supporting `--epochs`, `--batch-size`, `--lr`, `--device`, `--resume`.
- Produces: `evaluate(checkpoint_path, data_dir, device)` computing WT, TC, ET Dice scores.
- Artifacts: `backend/model_artifacts/brain_mri/unet3d_best.pth`, `last_checkpoint.pth`, `metrics.json`.

- [ ] **Step 1: Implement `ml/brain_mri/train.py` with standalone execution, checkpoint resume, and dataset missing handling**
- [ ] **Step 2: Implement `ml/brain_mri/evaluate.py` with held-out test split evaluation**
- [ ] **Step 3: Test CLI help and missing dataset execution for train & evaluate**
  Run: `.\backend\.venv\Scripts\python.exe ml/brain_mri/train.py --help`
  Expected: Displays options for epochs, batch-size, device, resume.

---

### Task 4: Backend Service & FastAPI Endpoints

**Files:**
- Create: `backend/app/schemas/brain_mri.py`
- Create: `backend/app/services/brain_mri.py`
- Create: `backend/app/api/endpoints/brain_mri.py`
- Modify: `backend/app/api/endpoints/models.py`
- Modify: `backend/app/main.py`
- Test: `backend/tests/test_brain_mri_api.py`

**Interfaces:**
- Produces: `GET /api/brain-mri/status` -> `BrainMRIStatusResponse`
- Produces: `POST /api/brain-mri/validate` -> `BrainMRIValidationResponse`
- Produces: `POST /api/brain-mri/segment` -> `BrainMRISegmentResponse` (persists to MongoDB)
- Modifies: `models.py` to report `unet_brain_mri` dynamic status.

- [ ] **Step 1: Create `backend/app/schemas/brain_mri.py` (Pydantic request/response schemas)**
- [ ] **Step 2: Implement `backend/app/services/brain_mri.py` (model loader, inference, volumetric calculator)**
- [ ] **Step 3: Implement `backend/app/api/endpoints/brain_mri.py` and register in `backend/app/main.py`**
- [ ] **Step 4: Update `backend/app/api/endpoints/models.py` to link `brain_mri_service` status**
- [ ] **Step 5: Run API endpoint tests**
  Run: `.\backend\.venv\Scripts\python.exe -m pytest backend/tests/test_brain_mri_api.py -v`
  Expected: PASS

---

### Task 5: PDF Clinical Report Generation Extension

**Files:**
- Modify: `backend/app/services/reports.py:410-480`
- Test: `backend/tests/test_brain_mri_report.py`

**Interfaces:**
- Modifies: `generate_analysis_pdf()` to support `modality == "brain_mri"`.
- Embeds: Multi-planar slice previews (Axial FLAIR, T1ce, segmentation overlay) and quantitative tumor volume table.

- [ ] **Step 1: Add Brain MRI reporting section in `backend/app/services/reports.py`**
- [ ] **Step 2: Test PDF generation with Brain MRI study payload**
  Run: `.\backend\.venv\Scripts\python.exe -m pytest backend/tests/test_brain_mri_report.py -v`
  Expected: Valid PDF bytes starting with `%PDF-` with embedded MRI images and volumetric table.

---

### Task 6: Frontend Brain MRI Workbench & Navigation State Preservation

**Files:**
- Modify: `frontend/src/types/index.ts`
- Modify: `frontend/src/services/api.ts`
- Modify: `frontend/src/context/AnalysisContextDefinition.ts`
- Modify: `frontend/src/context/AnalysisContext.tsx`
- Create: `frontend/src/components/workbench/BrainMRIWorkbench.tsx`
- Modify: `frontend/src/components/workbench/ModuleWorkbench.tsx`

**Interfaces:**
- Produces: `ActiveBrainMRICase` interface and storage in `AnalysisContext`.
- Produces: `BrainMRIWorkbench.tsx` with slice slider, plane selector, overlay mode toggle, and download report button.
- Updates: `ModuleWorkbench.tsx` to mount `BrainMRIWorkbench` when `tabId === 'brain-mri'`.

- [ ] **Step 1: Add `ActiveBrainMRICase`, `BrainMRIStatusResponse`, `BrainMRISegmentResponse` to `frontend/src/types/index.ts`**
- [ ] **Step 2: Add Brain MRI API calls to `frontend/src/services/api.ts`**
- [ ] **Step 3: Update `AnalysisContextDefinition.ts` and `AnalysisContext.tsx` with `activeBrainMRICase` state & reopen support**
- [ ] **Step 4: Create `BrainMRIWorkbench.tsx` with interactive slice viewer & volumetric cards**
- [ ] **Step 5: Mount `BrainMRIWorkbench` in `ModuleWorkbench.tsx`**
- [ ] **Step 6: Run frontend lint & build**
  Run: `npm run lint` & `npm run build`
  Expected: 0 errors, 0 warnings, build exit code 0.

---

### Task 7: Full Test Suite, End-to-End Verification & Instructions

**Files:**
- Consolidate: `backend/tests/test_brain_mri.py`
- Create: `scratch/verify_brain_mri_all.py`

- [ ] **Step 1: Consolidate comprehensive Brain MRI tests in `backend/tests/test_brain_mri.py`**
- [ ] **Step 2: Run all backend tests across all modules (Chest X-ray, Grad-CAM, Clinical ML, Brain MRI)**
  Run: `.\backend\.venv\Scripts\python.exe -m pytest backend/tests -v`
  Expected: 100% pass rate.
- [ ] **Step 3: Run frontend lint & build**
  Run: `npm run lint` & `npm run build`
  Expected: Clean build.
- [ ] **Step 4: Run end-to-end live verification script `scratch/verify_brain_mri_all.py`**
- [ ] **Step 5: Provide simple step-by-step instructions for the user**
