# MediVision AI - Stage 1 Architecture Design Document

**Date:** 2026-10-20  
**Status:** Approved for Implementation  
**Project:** MediVision AI - Final Year Major Project  
**Scope:** Stage 1 Foundation Only (Strictly No Model Training / No Fake Inferences)

---

## 1. Executive Summary & Project Goal
MediVision AI is a multi-disease medical AI research platform designed to support multiple diagnostic pipelines:
- Chest X-ray Classification (CNN / DenseNet / ResNet)
- Brain MRI Segmentation (U-Net)
- Skin Lesion Classification & Dermatoscopy (CNN / Vision Transformer)
- Anatomical / Lesion Localization (YOLO)
- Tabular Clinical Risk Prediction (Classical ML / XGBoost)
- Explainability (Grad-CAM, SHAP, Integrated Gradients)
- Medical Research Assistant (RAG with vector embeddings & LLM)

**Stage 1 Scope:** Establish the software engineering, API architecture, frontend interface, and configuration foundation. Stage 1 does not train or claim to train any models, nor does it present fake predictions.

---

## 2. Technology Stack & Directory Structure

### 2.1 Technology Stack
- **Frontend:** React + Vite + TypeScript
- **Styling:** Tailwind CSS (Deep Navy `#0B132B`/`#0F172A`, Medical Teal `#0D9488`/`#14B8A6`, Clean White `#FFFFFF`/`#F8FAFC`)
- **Icons:** Lucide React
- **Data Visualization:** Recharts
- **Backend:** Python FastAPI (Uvicorn, Pydantic v2)
- **Database:** MongoDB Atlas (Motor async driver, graceful offline fallback)
- **Environment Management:** Python venv + Node.js (strictly isolated)

### 2.2 Directory Structure
```
MediVision-AI/
├── frontend/
│   ├── public/
│   ├── src/
│   │   ├── assets/
│   │   ├── components/
│   │   │   ├── layout/
│   │   │   │   ├── Sidebar.tsx
│   │   │   │   └── Navbar.tsx
│   │   │   ├── dashboard/
│   │   │   │   ├── MetricCards.tsx
│   │   │   │   ├── ModelStatusGrid.tsx
│   │   │   │   ├── ActivityFeed.tsx
│   │   │   │   └── SampleCharts.tsx
│   │   │   ├── workbench/
│   │   │   │   └── ModuleWorkbench.tsx
│   │   │   └── common/
│   │   │       ├── StatusBadge.tsx
│   │   │       └── NoticeBanner.tsx
│   │   ├── services/
│   │   │   └── api.ts
│   │   ├── types/
│   │   │   └── index.ts
│   │   ├── App.tsx
│   │   ├── main.tsx
│   │   └── index.css
│   ├── .env.example
│   ├── .gitignore
│   ├── package.json
│   ├── tsconfig.json
│   ├── tailwind.config.js
│   ├── postcss.config.js
│   └── vite.config.ts
├── backend/
│   ├── app/
│   │   ├── api/
│   │   │   ├── __init__.py
│   │   │   └── endpoints/
│   │   │       ├── __init__.py
│   │   │       ├── health.py
│   │   │       └── models.py
│   │   ├── core/
│   │   │   ├── __init__.py
│   │   │   └── config.py
│   │   ├── db/
│   │   │   ├── __init__.py
│   │   │   └── mongodb.py
│   │   ├── schemas/
│   │   │   ├── __init__.py
│   │   │   └── status.py
│   │   └── main.py
│   ├── .env.example
│   ├── .gitignore
│   ├── requirements.txt
│   └── run.py
├── docs/
│   └── superpowers/specs/2026-10-20-medivision-stage1-design.md
└── README.md
```

---

## 3. Design System & UI/UX Guidelines
- **Palette:**
  - Deep Navy background / surfaces: `#0B132B`, `#0F172A`, `#1E293B`
  - Teal accents & branding: `#0D9488`, `#14B8A6`, `#2DD4BF`
  - Crisp cards & contrast: White `#FFFFFF`, Slate `#F8FAFC`, Border `#E2E8F0` / `#334155`
  - Text: Dark `#0F172A` on light cards; `#F8FAFC` on navy panels
- **Navigation (11 items):**
  1. Dashboard
  2. Chest X-ray Analysis
  3. Brain MRI Analysis
  4. Skin Lesion Analysis
  5. YOLO Detection
  6. Clinical ML
  7. Explainable AI
  8. Medical Research Assistant
  9. Model Evaluation
  10. Analysis History
  11. Settings
- **Chart Placeholders:** Clearly marked with badges: `"SAMPLE DATA - PRE-TRAINING BENCHMARK"`.
- **Integrity Rule:** No fake predictions, no mock confidence percentages represented as real AI inferences.

---

## 4. Backend API Specification

### 4.1 `GET /api/health`
- **Response Schema:**
  ```json
  {
    "status": "online",
    "version": "1.0.0-stage1",
    "timestamp": "ISO-8601",
    "database": {
      "status": "connected" | "disconnected",
      "engine": "MongoDB Atlas",
      "details": "Connection message / offline mode"
    },
    "environment": "development"
  }
  ```

### 4.2 `GET /api/models/status`
- **Response Schema:**
  ```json
  {
    "stage": 1,
    "system_status": "foundation_ready",
    "models": {
      "cnn_chest_xray": {
        "name": "Chest X-ray Classifier",
        "modality": "chest_xray",
        "architecture": "DenseNet121 / ResNet50",
        "framework": "PyTorch / Torchvision",
        "status": "unconfigured",
        "trained": false,
        "weights_path": null,
        "next_stage": "Stage 2 Data Ingestion & Preprocessing"
      },
      "unet_brain_mri": {
        "name": "Brain MRI Lesion Segmenter",
        "modality": "brain_mri",
        "architecture": "Attention U-Net",
        "framework": "PyTorch",
        "status": "unconfigured",
        "trained": false,
        "weights_path": null,
        "next_stage": "Stage 2 Segmentation Pipeline"
      },
      "cnn_skin_lesion": {
        "name": "Skin Lesion Classifier",
        "modality": "dermatoscopy",
        "architecture": "EfficientNet-B4 / ViT",
        "framework": "PyTorch",
        "status": "unconfigured",
        "trained": false,
        "weights_path": null,
        "next_stage": "Stage 2 Pretrained Transfer Tuning"
      },
      "yolo_detection": {
        "name": "Anatomical Lesion Detector",
        "modality": "multi_modal_radiology",
        "architecture": "YOLOv8 / YOLOv11 Medical",
        "framework": "Ultralytics PyTorch",
        "status": "unconfigured",
        "trained": false,
        "weights_path": null,
        "next_stage": "Stage 3 Bounding Box Annotation"
      },
      "clinical_ml": {
        "name": "Tabular Clinical Risk Engine",
        "modality": "clinical_tabular",
        "architecture": "XGBoost / LightGBM / Random Forest",
        "framework": "Scikit-Learn / XGBoost",
        "status": "unconfigured",
        "trained": false,
        "weights_path": null,
        "next_stage": "Stage 3 Feature Engineering"
      },
      "xai_engine": {
        "name": "Explainable AI Engine",
        "modality": "interpretability",
        "architecture": "Grad-CAM / SHAP / Captum",
        "framework": "PyTorch Captum / SHAP",
        "status": "unconfigured",
        "trained": false,
        "weights_path": null,
        "next_stage": "Stage 4 Attribution Map Hooks"
      },
      "rag_assistant": {
        "name": "Medical Research Assistant",
        "modality": "research_literature",
        "architecture": "RAG (Vector Store + Embeddings + LLM)",
        "framework": "LangChain / LlamaIndex",
        "status": "unconfigured",
        "trained": false,
        "weights_path": null,
        "next_stage": "Stage 4 PubMed Document Indexing"
      }
    }
  }
  ```

---

## 5. Verification Plan
1. Check Python version and Node.js version.
2. Initialize backend directory, create virtual environment (or install requirements via pip), verify FastAPI server runs and returns 200 on `/api/health` and `/api/models/status`.
3. Initialize frontend with Vite + React + TypeScript, install Tailwind CSS, Lucide React, and Recharts.
4. Verify frontend dev build compiles with zero TypeScript errors.
5. Verify cross-origin connectivity: frontend dynamically pulls and displays health and model statuses from backend.
6. Verify clear visual labeling of sample charts and untrained model notices.
