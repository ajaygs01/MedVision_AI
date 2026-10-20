# MediVision AI - Stage 2 Design Document: CNN Chest X-ray Classification

**Date:** 2026-10-20  
**Status:** Approved for Implementation  
**Milestone:** Stage 2 - CNN Chest X-ray Classification Pipeline  
**Dataset Target:** RSNA Pneumonia Detection Challenge (Kaggle)

---

## 1. Executive Summary & Objective

Stage 2 develops the complete deep learning pipeline for binary classification of chest radiographs (Normal / No Lung Opacity vs. Lung Opacity / Pneumonia) using transfer learning with a PyTorch `torchvision.models.resnet50` architecture.

### Scope & Guardrails
- **Academic & Research Prototype:** Inferences are research demonstrations and not clinically validated.
- **Integrity Safeguard:** When no trained checkpoint exists (`model_artifacts/chest_xray_cnn/best_model.pth`), the system returns a clear, explicit **`NOT TRAINED`** state (`status: "not_trained"`). It does NOT fabricate predictions, mock confidence scores, or simulate fake diagnoses.
- **Dataset Preconditions:** The dataset is not assumed to be present. If absent, the pipeline stands fully implemented and verified via synthetic fixtures, with a clear note that training execution is blocked until the RSNA files are placed in `datasets/rsna_pneumonia/`.

---

## 2. Directory & Component Architecture

```text
MediVision-AI/
├── datasets/
│   ├── README.md                          # Instructions for Kaggle RSNA dataset setup
│   └── rsna_pneumonia/                    # Target directory for RSNA dataset
│       ├── stage_2_train_labels.csv
│       ├── stage_2_detailed_class_info.csv
│       └── stage_2_train_images/          # DICOM files (.dcm)
├── model_artifacts/
│   └── chest_xray_cnn/                    # Checkpoint directory (git-ignored)
│       ├── best_model.pth                 # Saved weights (when trained)
│       └── metrics.json                   # Test set metrics & metadata
├── ml/
│   └── chest_xray/
│       └── cnn/
│           ├── __init__.py
│           ├── config.py                  # Hyperparameters & path configurations
│           ├── dataset.py                 # Patient-level splitting & DICOM data loader
│           ├── train.py                   # PyTorch ResNet-50 training loop
│           ├── evaluate.py                # Comprehensive test set evaluation script
│           └── predict.py                 # Reusable inference service with missing-checkpoint guard
├── backend/
│   ├── app/
│   │   ├── api/
│   │   │   └── endpoints/
│   │   │       ├── health.py
│   │   │       ├── models.py              # Dynamic checkpoint detection
│   │   │       └── imaging.py             # POST /api/imaging/chest-xray/classify
│   │   ├── schemas/
│   │   │   └── imaging.py                 # Request & response Pydantic models
│   │   └── main.py
│   └── tests/
│       ├── test_cnn_pipeline.py           # Unit tests for dataset, transforms & inference
│       └── test_imaging_api.py            # API endpoint integration tests
└── frontend/
    └── src/
        ├── components/
        │   └── workbench/
        │       └── ChestXRayWorkbench.tsx # Upload, preview, analyze, NOT TRAINED warning
        └── services/
            └── api.ts                     # classifyChestXRay API client method
```

---

## 3. Machine Learning Specifications

### 3.1 Architecture
- **Backbone:** `torchvision.models.resnet50(weights=ResNet50_Weights.DEFAULT)`
- **Classifier Head:** Replaces `model.fc` with:
  ```python
  nn.Sequential(
      nn.Dropout(p=0.3),
      nn.Linear(2048, 1)  # Single logit output for BCEWithLogitsLoss
  )
  ```
- **Loss Formulation:** `torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)`
  - Class imbalance handling: `pos_weight = (num_negative / num_positive)` calculated from training set labels.
- **Optimizer:** `torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)`
- **Learning Rate Scheduler:** `torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)` monitoring validation AUROC.

### 3.2 Dataset Loader & Patient-Level Splitting
- **RSNA CSV Schema:**
  - `stage_2_train_labels.csv`: columns `patientId`, `x`, `y`, `width`, `height`, `Target`.
  - Notice: If a patient has multiple lung opacity bounding boxes, multiple rows exist with the same `patientId`. The loader groups by `patientId`, taking `max(Target)` as the binary label.
- **Patient-Level Split:**
  - Splitting is performed strictly on unique `patientId`s (80% train, 10% val, 10% test).
  - Images from the same patient NEVER cross partitions, guaranteeing zero data leakage.
- **DICOM Decoding:**
  - Uses `pydicom.dcmread()` to extract `pixel_array`.
  - Intensity normalization: windowing / min-max scaling to `[0, 1]`.
  - Channel handling: Single-channel monochrome converted to 3-channel RGB for ResNet-50 input.
  - Standard input size: `224x224`.

---

## 4. API & Inference Contracts

### 4.1 Endpoint: `POST /api/imaging/chest-xray/classify`
- **Request:** `multipart/form-data` with `file: UploadFile` (supports `.dcm`, `.png`, `.jpg`, `.jpeg`).
- **File Validation:**
  - Max file size: 25 MB.
  - Allowed extensions: `.dcm`, `.png`, `.jpg`, `.jpeg`.
  - Integrity check: Validates DICOM header / PIL image decoding.
- **Response Schemas:**

**When Model Checkpoint Does Not Exist (Default Stage 2 initial state):**
```json
{
  "status": "not_trained",
  "model_name": "Chest X-ray ResNet50 Classifier",
  "prediction": null,
  "confidence": null,
  "message": "Model weights checkpoint not found. Model training has not been executed yet.",
  "disclaimer": "Academic research prototype. Results are not clinically validated."
}
```

**When Model Checkpoint Exists:**
```json
{
  "status": "success",
  "model_name": "Chest X-ray ResNet50 Classifier",
  "prediction": "Lung Opacity / Pneumonia" | "Normal / No Lung Opacity",
  "predicted_class_id": 1 | 0,
  "confidence": 0.874,
  "threshold": 0.5,
  "filename": "patient_xray.dcm",
  "file_type": "dicom" | "standard_image",
  "message": "Inference completed successfully using trained ResNet50 checkpoint.",
  "disclaimer": "Academic research prototype. Results are not clinically validated."
}
```

### 4.2 Dynamic Model Status: `GET /api/models/status`
- If `model_artifacts/chest_xray_cnn/best_model.pth` exists:
  - `models["cnn_chest_xray"].status = "ready"`
  - `models["cnn_chest_xray"].trained = true`
  - `models["cnn_chest_xray"].weights_path = "model_artifacts/chest_xray_cnn/best_model.pth"`
  - `active_trained_count` increments.
- If it does not exist:
  - Remains `status: "unconfigured"`, `trained: false`.

---

## 5. Frontend Chest X-ray Workbench UI
- Located at `/chest-xray` navigation tab.
- **Upload Dropzone:** Supports drag-and-drop or click to browse `.dcm`, `.png`, `.jpg` files.
- **Image Preview:** Renders uploaded standard images, or provides DICOM metadata preview.
- **Analyze Button:** Disabled when no image is loaded, animated spinner during API call.
- **NOT TRAINED Warning:** Prominently alerts the user that the CNN weights have not been trained yet.
- **Results Container:** Displays prediction, confidence bar, and disclaimer when inference succeeds.
