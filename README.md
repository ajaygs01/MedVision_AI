# MediVision AI • Multi-Disease Medical AI Research Platform

> **Final-Year Major Research Project**  
> **Status:** Stage 1 Foundation Complete  
> **Milestone Notice:** Stage 1 establishes software architecture, API contracts, database configuration, and the research dashboard. In accordance with clinical integrity protocols, **no machine learning models are trained or executing inferences** in this stage.

---

## 📌 Project Overview

MediVision AI is a multi-disease medical AI research platform designed to support seven state-of-the-art diagnostic and research pipelines:

1. **Chest X-ray Classification:** DenseNet-121 / ResNet-50 multi-label pathology detection.
2. **Brain MRI Segmentation:** Attention 3D U-Net for volumetric brain tumor delineation.
3. **Skin Lesion Classification:** EfficientNet-B4 / Vision Transformer (ViT) dermatoscopy classifier.
4. **Anatomical Object Localization:** YOLOv8 / YOLOv11 Medical for spatial lesion detection.
5. **Clinical ML Engine:** XGBoost / LightGBM ensemble for patient tabular risk stratification.
6. **Explainable AI (XAI):** Grad-CAM, Score-CAM, SHAP, and Captum attribution heatmaps.
7. **Medical Research Assistant:** RAG copilot grounded in PubMed literature and clinical guidelines.

---

## 🛠️ Technology Stack

- **Frontend:** React 19 + Vite 8 + TypeScript
- **Styling:** Tailwind CSS v4 (Deep Navy `#0B132B`, Medical Teal `#0D9488`, Slate `#F8FAFC`)
- **Icons:** Lucide React
- **Data Visualization:** Recharts
- **Backend:** Python FastAPI + Uvicorn + Pydantic v2
- **Database:** MongoDB Atlas (Async Motor / PyMongo with graceful offline fallback)
- **Environment Isolation:** Standalone Node.js `node_modules` and Python virtual environment (`.venv`)

---

## 📂 Project Directory Structure

```text
MediVision-AI/
├── frontend/                     # React + Vite + TypeScript Client
│   ├── src/
│   │   ├── components/
│   │   │   ├── common/           # StatusBadge, NoticeBanner
│   │   │   ├── dashboard/        # MetricCards, ModelStatusGrid, SampleCharts, ActivityFeed
│   │   │   ├── layout/           # Sidebar, Navbar
│   │   │   └── workbench/        # Dedicated Module Technical Blueprints
│   │   ├── services/             # Centralized typed API service (api.ts)
│   │   ├── types/                # TypeScript interfaces
│   │   ├── App.tsx               # Main application component
│   │   ├── main.tsx              # React DOM entrypoint
│   │   └── index.css             # Tailwind v4 & Design Tokens
│   ├── .env.example              # Frontend environment template
│   ├── package.json              # Frontend dependencies
│   ├── tailwind.config.js
│   └── vite.config.ts            # Vite config with API dev proxy
├── backend/                      # Python FastAPI Backend
│   ├── app/
│   │   ├── api/
│   │   │   └── endpoints/        # health.py, models.py
│   │   ├── core/                 # config.py (Pydantic BaseSettings)
│   │   ├── db/                   # mongodb.py (Async Atlas Client & Fallback)
│   │   ├── schemas/              # status.py (Pydantic Response Models)
│   │   └── main.py               # FastAPI application factory & CORS
│   ├── tests/                    # Pytest endpoint test suite
│   ├── .env.example              # Backend environment template
│   ├── requirements.txt          # Python dependencies
│   └── run.py                    # Uvicorn launcher script
├── docs/                         # Architecture specifications and implementation plans
├── .env.example                  # Root environment template
├── .gitignore                    # Root gitignore
└── README.md                     # Documentation & setup instructions
```

---

## 🚀 Getting Started on Windows PowerShell

### Prerequisites
- **Python 3.10+** (verified on Python 3.14)
- **Node.js 18+** (verified on Node.js v24 & npm 11)

---

### Step 1: Start the FastAPI Backend

Open a **Windows PowerShell** terminal:

```powershell
# Navigate to backend directory
cd C:\Users\gsaja\OneDrive\Desktop\MediVision-AI\backend

# Activate the virtual environment
.\.venv\Scripts\Activate.ps1

# (Optional) If running for the first time, install dependencies:
# pip install -r requirements.txt

# Start the FastAPI server
python run.py
```

The backend server will start at:
- **API Base:** `http://localhost:8000`
- **Interactive Swagger Docs:** `http://localhost:8000/docs`
- **Health Endpoint:** `http://localhost:8000/api/health`
- **Model Status Endpoint:** `http://localhost:8000/api/models/status`

#### Run Automated Backend Tests:
```powershell
.\.venv\Scripts\pytest.exe -v
```

---

### Step 2: Start the React Frontend

Open a **second Windows PowerShell** terminal:

```powershell
# Navigate to frontend directory
cd C:\Users\gsaja\OneDrive\Desktop\MediVision-AI\frontend

# (Optional) If running for the first time, install dependencies:
# npm install

# Start the Vite development server
npm run dev
```

The frontend dashboard will be available at:
- **Web UI:** `http://localhost:5173`

---

## ⚙️ Environment Variables & MongoDB Atlas

Both backend and frontend support configurable environment variables:

### Backend `.env` (`backend/.env` or root `.env`)
```ini
PROJECT_NAME="MediVision AI Research Platform"
API_V1_PREFIX="/api"
PORT=8000
HOST="0.0.0.0"
ENVIRONMENT="development"
CORS_ORIGINS=["http://localhost:5173","http://127.0.0.1:5173","http://localhost:3000"]

# MongoDB Atlas Connection URI (Optional for Stage 1)
# Leave empty to run in Graceful Offline Fallback mode:
MONGODB_URI=mongodb+srv://<username>:<password>@cluster0.example.mongodb.net/?retryWrites=true&w=majority
MONGODB_DB_NAME=medivision_ai
MONGODB_CONNECT_TIMEOUT_MS=2500
```

> **Note on Database Resiliency:** If `MONGODB_URI` is omitted or network access is blocked, the backend **does not crash**. It automatically transitions to `disconnected / offline fallback` mode and reports this gracefully on `/api/health` and the dashboard navbar badge.

---

## 📡 API Endpoints

### 1. `GET /api/health`
Returns system health, server timestamp, API version, and database connectivity.

```json
{
  "status": "online",
  "version": "1.0.0-stage1",
  "timestamp": "2026-10-20T11:00:00Z",
  "database": {
    "status": "connected",
    "engine": "MongoDB Atlas",
    "details": "Connected to MongoDB Atlas (medivision_ai)"
  },
  "environment": "development"
}
```

### 2. `GET /api/models/status`
Returns configuration states for all 7 research pipelines. In Stage 1, all pipelines have `trained: false` and `status: "unconfigured"`.

```json
{
  "stage": 1,
  "system_status": "foundation_ready",
  "total_models": 7,
  "active_trained_count": 0,
  "models": {
    "cnn_chest_xray": {
      "name": "Chest X-ray Multi-Abnormality Classifier",
      "architecture": "DenseNet-121 / ResNet-50 Pretrained Backbone",
      "framework": "PyTorch 2.x / Torchvision",
      "status": "unconfigured",
      "trained": false,
      "weights_path": null,
      "next_stage": "Stage 2 Data Ingestion & Preprocessing (NIH ChestX-ray14 / CheXpert)"
    }
    // unet_brain_mri, cnn_skin_lesion, yolo_detection, clinical_ml, xai_engine, rag_assistant
  },
  "disclaimer": "Stage 1 foundation only: No clinical prediction models are trained or executing inferences."
}
```

---

## 🗺️ Multi-Stage Project Roadmap

| Stage | Focus Area | Status | Deliverables |
|---|---|---|---|
| **Stage 1** | **Software & System Foundation** | **✅ COMPLETE** | FastAPI API, React+Vite+Tailwind Dashboard, 11 Nav views, MongoDB Atlas config, Health checks, Pytest suite |
| **Stage 2** | **Medical Imaging Pipelines** | ⏳ Planned | NIH ChestX-ray14, BraTS MRI, ISIC Skin Lesion preprocessing and baseline training loops |
| **Stage 3** | **Localization & Clinical ML** | ⏳ Planned | YOLOv8 bounding box detection on radiology scans, XGBoost EHR risk engine |
| **Stage 4** | **Explainability & RAG Assistant** | ⏳ Planned | Grad-CAM saliency heatmaps, SHAP values, PubMed literature RAG copilot |
| **Stage 5** | **Clinical Evaluation & Deployment** | ⏳ Planned | Multi-center validation, ROC/Dice benchmark reports, Docker containerization |

---

## 🛡️ Research Ethics & Academic Integrity

- **No Fake Predictions:** Stage 1 does not use mock inference engines or pretend that un-trained models are producing real diagnoses.
- **Labeled Benchmarks:** All charts and metric visuals in Stage 1 are explicitly watermarked and labeled as sample pre-training references.
- **Data Privacy:** Sensitive credentials, connection strings, and certificates are strictly excluded via `.gitignore`.
