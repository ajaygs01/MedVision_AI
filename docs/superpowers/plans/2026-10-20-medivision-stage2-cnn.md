# MediVision AI - Stage 2 Implementation Plan: CNN Chest X-ray Classification

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the complete PyTorch ResNet-50 CNN chest X-ray classification pipeline for the RSNA Pneumonia Detection Challenge dataset, including DICOM decoding, patient-level splitting, training script with class weighting, evaluation metrics (Accuracy, Sensitivity, Specificity, F1, AUROC, Confusion Matrix), inference service, FastAPI endpoint `POST /api/imaging/chest-xray/classify`, dynamic model status reporting, and React dashboard workbench.

**Architecture:** 
- ML Core: `ml/chest_xray/cnn/` containing `config.py`, `dataset.py`, `train.py`, `evaluate.py`, and `predict.py`.
- Backend: FastAPI route mounted under `/api/imaging/chest-xray/classify`, reading from `ml.chest_xray.cnn.predict.ChestXRayInferenceService`, with dynamic checkpoint detection in `app/api/endpoints/models.py`.
- Frontend: Dedicated `ChestXRayWorkbench.tsx` with drag-and-drop file upload, DICOM/image preview, Analyze button, prominent **NOT TRAINED** alert, and real inference display when weights are available.
- Documentation: `datasets/README.md` guiding Kaggle RSNA dataset setup.

**Tech Stack:** Python 3.10+, PyTorch, Torchvision, pydicom, NumPy, Pandas, Scikit-learn, Pillow, Matplotlib, FastAPI, React 19, TypeScript, Tailwind CSS, Lucide React.

**Spec:** `docs/superpowers/specs/2026-10-20-medivision-stage2-cnn-design.md`

## Global Constraints
- **Strict Integrity Guard:** If no model checkpoint exists, return clean `status: "not_trained"`. NEVER fabricate mock predictions or pseudo-clinical inferences.
- **Zero Data Leakage:** Ensure train/val/test splits occur strictly by unique `patientId`.
- **Medical Disclaimer:** Always attach academic research prototype disclaimers.
- **Large Files Out of Git:** Add `model_artifacts/` and `datasets/rsna_pneumonia/` to `.gitignore`.

---

### Task 1: Environment Dependencies & Directory Setup
**Files:**
- Modify: `backend/requirements.txt`
- Modify: `.gitignore`
- Modify: `backend/.gitignore`
- Create: `datasets/README.md`

- [ ] **Step 1: Add ML dependencies (`torch`, `torchvision`, `pydicom`, `scikit-learn`, `pandas`, `pillow`, `matplotlib`, `python-multipart`) to `backend/requirements.txt`**
- [ ] **Step 2: Update `.gitignore` to exclude `datasets/` (except README.md) and `model_artifacts/*.pth`**
- [ ] **Step 3: Create `datasets/README.md` detailing the RSNA Pneumonia Detection Challenge download procedure**
- [ ] **Step 4: Install dependencies in `backend/.venv`**

---

### Task 2: CNN Configuration Module
**Files:**
- Create: `ml/__init__.py`
- Create: `ml/chest_xray/__init__.py`
- Create: `ml/chest_xray/cnn/__init__.py`
- Create: `ml/chest_xray/cnn/config.py`

- [ ] **Step 1: Create package init files**
- [ ] **Step 2: Implement `CNNConfig` dataclass with image size, batch size, learning rate, paths, and device auto-detection**

---

### Task 3: Dataset Loader with DICOM Support & Patient Splitting
**Files:**
- Create: `ml/chest_xray/cnn/dataset.py`

- [ ] **Step 1: Implement `RSNAPneumoniaDataset` supporting DICOM decoding, windowing, and RGB transformation**
- [ ] **Step 2: Implement `create_patient_split_loaders()` grouping multiple bounding box rows per patient and performing leak-free train/val/test splits**

---

### Task 4: ResNet-50 Training Loop with Checkpointing
**Files:**
- Create: `ml/chest_xray/cnn/train.py`

- [ ] **Step 1: Implement ResNet-50 transfer learning model architecture with single logit head**
- [ ] **Step 2: Implement training loop with BCEWithLogitsLoss, positive class weighting, validation AUROC tracking, and best model checkpointing**
- [ ] **Step 3: Add CLI interface for configurable epochs, batch size, and learning rate**

---

### Task 5: Comprehensive Test Evaluation Module
**Files:**
- Create: `ml/chest_xray/cnn/evaluate.py`

- [ ] **Step 1: Implement evaluation script computing Accuracy, Sensitivity, Specificity, Precision, F1, ROC-AUC, and Confusion Matrix**
- [ ] **Step 2: Export evaluation results to `model_artifacts/chest_xray_cnn/metrics.json`**

---

### Task 6: Reusable Inference Service
**Files:**
- Create: `ml/chest_xray/cnn/predict.py`

- [ ] **Step 1: Implement `ChestXRayInferenceService` singleton with safe checkpoint loading**
- [ ] **Step 2: Implement preprocessing for both DICOM (.dcm) bytes and standard image formats (.png, .jpg)**
- [ ] **Step 3: Implement clean `status: "not_trained"` fallback when checkpoint is absent**

---

### Task 7: FastAPI Endpoint & Dynamic Model Status
**Files:**
- Create: `backend/app/schemas/imaging.py`
- Create: `backend/app/api/endpoints/imaging.py`
- Modify: `backend/app/api/endpoints/models.py`
- Modify: `backend/app/main.py`

- [ ] **Step 1: Create Pydantic schemas for classification request/response**
- [ ] **Step 2: Implement `POST /api/imaging/chest-xray/classify` endpoint**
- [ ] **Step 3: Update `GET /api/models/status` to dynamically detect `best_model.pth` and report ready/untrained state**
- [ ] **Step 4: Mount imaging router in `backend/app/main.py`**

---

### Task 8: Automated Test Suite
**Files:**
- Create: `backend/tests/test_cnn_pipeline.py`
- Create: `backend/tests/test_imaging_api.py`

- [ ] **Step 1: Write unit tests for dataset loader, DICOM decoding, and preprocessing with synthetic fixtures**
- [ ] **Step 2: Write tests for missing checkpoint handling (verifying clean `not_trained` JSON)**
- [ ] **Step 3: Write tests for `POST /api/imaging/chest-xray/classify` with invalid files and valid synthetic images**
- [ ] **Step 4: Run pytest and verify all tests pass**

---

### Task 9: Frontend Chest X-ray Workbench Component
**Files:**
- Create: `frontend/src/components/workbench/ChestXRayWorkbench.tsx`
- Modify: `frontend/src/services/api.ts`
- Modify: `frontend/src/App.tsx`

- [ ] **Step 1: Add `classifyChestXRay(file: File)` API function in `frontend/src/services/api.ts`**
- [ ] **Step 2: Create `ChestXRayWorkbench.tsx` with upload dropzone, image preview, Analyze action, prominent NOT TRAINED notice, and real inference output**
- [ ] **Step 3: Wire `ChestXRayWorkbench` into `frontend/src/App.tsx` for the `'chest-xray'` tab**

---

### Task 10: Full End-to-End Verification & Instructions
- [ ] **Step 1: Run complete backend pytest suite**
- [ ] **Step 2: Run frontend `npm run build` and `npm run lint`**
- [ ] **Step 3: Verify live endpoint `POST /api/imaging/chest-xray/classify` via PowerShell `Invoke-RestMethod`**
- [ ] **Step 4: Update root `README.md` with Stage 2 commands and Kaggle RSNA instructions**
