# MediVision AI - Stage 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create the solid software engineering foundation for MediVision AI Stage 1: a multi-disease medical AI research platform featuring a FastAPI backend with working `/api/health` and `/api/models/status` endpoints, MongoDB Atlas configuration, and a responsive React + Vite + TypeScript dashboard in a Deep Navy/Teal/White aesthetic with 11 functional navigation views and sample-labeled charts.

**Architecture:** Monorepo containing strictly isolated `frontend/` and `backend/` packages. Backend exposes asynchronous FastAPI endpoints with Pydantic v2 schemas and graceful MongoDB Atlas ping. Frontend uses Vite + React 18/19 + TypeScript + Tailwind CSS with a dedicated API service layer (`src/services/api.ts`), status polling, responsive sidebar layout, and dedicated module workbenches with explicit Stage 2 Pending notices.

**Tech Stack:** Python 3.10+, FastAPI, Uvicorn, Pydantic v2, Motor/PyMongo, React 18/19, Vite, TypeScript, Tailwind CSS, Lucide React, Recharts.

**Spec:** `docs/superpowers/specs/2026-10-20-medivision-stage1-design.md`

## Global Constraints
- **Strictly No Model Training:** No CNN, YOLO, U-Net, Clinical ML, or RAG model training or weights loading in Stage 1.
- **Strictly No Fake Predictions:** Do not output mock patient diagnoses or pretend models are trained.
- **Dependency Isolation:** Keep frontend `package.json` and backend `requirements.txt` strictly separate.
- **Color Scheme:** Deep Navy (`#0B132B`/`#0F172A`), Teal (`#0D9488`/`#14B8A6`), Clean White/Slate cards (`#FFFFFF`/`#F8FAFC`).
- **Graceful DB Fallback:** Backend must boot and pass health checks even if `MONGODB_URI` is unset or offline.
- **Operating System:** Windows PowerShell commands must be provided and validated.

---

### Task 1: Repository & Backend Foundation Scaffolding
**Files:**
- Create: `backend/requirements.txt`
- Create: `backend/.env.example`
- Create: `backend/.gitignore`
- Create: `.gitignore`
- Create: `.env.example`

**Interfaces:**
- Produces: Backend dependency specifications (`fastapi`, `uvicorn`, `pydantic`, `pydantic-settings`, `motor`, `pymongo`, `pytest`, `httpx`).

- [ ] **Step 1: Create backend dependencies list in `backend/requirements.txt`**
- [ ] **Step 2: Create root and backend `.gitignore` and `.env.example` files**
- [ ] **Step 3: Verify Python virtual environment and install backend dependencies**

---

### Task 2: Backend Core Configuration & Database Connection
**Files:**
- Create: `backend/app/core/config.py`
- Create: `backend/app/db/mongodb.py`

**Interfaces:**
- Produces: `get_settings()` returning `Settings` instance with `api_v1_prefix`, `project_name`, `mongodb_uri`, `mongodb_db_name`, `cors_origins`.
- Produces: `check_mongo_connection()` returning `Tuple[bool, str]`.

- [ ] **Step 1: Implement Pydantic `Settings` in `backend/app/core/config.py`**
- [ ] **Step 2: Implement async MongoDB connection manager in `backend/app/db/mongodb.py` with graceful timeout fallback**
- [ ] **Step 3: Test settings load and offline fallback**

---

### Task 3: Backend Schemas & Endpoints (`/api/health` & `/api/models/status`)
**Files:**
- Create: `backend/app/schemas/status.py`
- Create: `backend/app/api/endpoints/health.py`
- Create: `backend/app/api/endpoints/models.py`
- Create: `backend/app/main.py`
- Create: `backend/run.py`

**Interfaces:**
- Produces: `GET /api/health` returning `HealthResponse`
- Produces: `GET /api/models/status` returning `ModelsStatusResponse`
- Produces: CORS enabled FastAPI application on port 8000

- [ ] **Step 1: Implement Pydantic models for Health and Model status**
- [ ] **Step 2: Implement health endpoint checking system time, uptime, and MongoDB status**
- [ ] **Step 3: Implement model status endpoint explicitly describing CNN, YOLO, U-Net, Clinical ML, XAI, and RAG as unconfigured/untrained**
- [ ] **Step 4: Create `backend/app/main.py` mounting routers and CORS middleware**
- [ ] **Step 5: Create `backend/run.py` launcher script**

---

### Task 4: Backend Automated Tests
**Files:**
- Create: `backend/tests/test_endpoints.py`

**Interfaces:**
- Consumes: FastAPI app from `backend.app.main:app`
- Produces: Pytest test suite asserting 200 OK on `/api/health` and `/api/models/status`

- [ ] **Step 1: Write pytest test cases for `/api/health` and `/api/models/status`**
- [ ] **Step 2: Run pytest to verify all tests pass**

---

### Task 5: Frontend Project Initialization with Vite, React, TypeScript, and Tailwind
**Files:**
- Create: `frontend/package.json`
- Create: `frontend/vite.config.ts`
- Create: `frontend/tsconfig.json`
- Create: `frontend/tailwind.config.js`
- Create: `frontend/postcss.config.js`
- Create: `frontend/index.html`
- Create: `frontend/src/index.css`
- Create: `frontend/.env.example`
- Create: `frontend/.gitignore`

**Interfaces:**
- Produces: Tailwind tokens (navy `#0B132B`, teal `#0D9488`), Lucide React icons, Recharts installed.

- [ ] **Step 1: Initialize Vite React TypeScript project in `frontend/`**
- [ ] **Step 2: Install dependencies (`lucide-react`, `recharts`, `clsx`, `tailwind-merge`)**
- [ ] **Step 3: Configure Tailwind CSS with Deep Navy & Medical Teal palette in `tailwind.config.js` and `index.css`**
- [ ] **Step 4: Verify frontend builds via `npm run build`**

---

### Task 6: Frontend API Service & Type Contracts
**Files:**
- Create: `frontend/src/types/index.ts`
- Create: `frontend/src/services/api.ts`

**Interfaces:**
- Produces: `fetchHealth(): Promise<HealthResponse>`
- Produces: `fetchModelsStatus(): Promise<ModelsStatusResponse>`

- [ ] **Step 1: Define TypeScript interfaces matching backend Pydantic schemas**
- [ ] **Step 2: Implement resilient `api.ts` client with error handling and fallback status**

---

### Task 7: Frontend Layout & Sidebar Navigation
**Files:**
- Create: `frontend/src/components/layout/Sidebar.tsx`
- Create: `frontend/src/components/layout/Navbar.tsx`
- Create: `frontend/src/components/common/StatusBadge.tsx`

**Interfaces:**
- Produces: Responsive sidebar featuring all 11 required navigation tabs with active indicator and badges
- Produces: Top navigation bar displaying live backend connection status and stage indicators

- [ ] **Step 1: Build `StatusBadge` component for online/offline/untrained states**
- [ ] **Step 2: Build `Sidebar` with all 11 navigation items and Lucide icons**
- [ ] **Step 3: Build `Navbar` with system breadcrumb, backend status pill, and refresh button**

---

### Task 8: Frontend Dashboard Views
**Files:**
- Create: `frontend/src/components/dashboard/MetricCards.tsx`
- Create: `frontend/src/components/dashboard/ModelStatusGrid.tsx`
- Create: `frontend/src/components/dashboard/ActivityFeed.tsx`
- Create: `frontend/src/components/dashboard/SampleCharts.tsx`
- Create: `frontend/src/components/dashboard/DashboardView.tsx`

**Interfaces:**
- Produces: Complete overview dashboard combining metrics, live model status grid, audit activity, and clearly labeled sample benchmark charts.

- [ ] **Step 1: Build `MetricCards` with key platform metrics and Stage 1 indicators**
- [ ] **Step 2: Build `ModelStatusGrid` rendering cards for CNN, YOLO, U-Net, Clinical ML, XAI, and RAG based on live backend data**
- [ ] **Step 3: Build `ActivityFeed` showing recent system and deployment audit logs**
- [ ] **Step 4: Build `SampleCharts` using Recharts, with prominent "SAMPLE PRE-TRAINING BENCHMARK" banners**
- [ ] **Step 5: Assemble `DashboardView`**

---

### Task 9: Frontend Module Workbenches
**Files:**
- Create: `frontend/src/components/workbench/ModuleWorkbench.tsx`

**Interfaces:**
- Produces: Interactive view for each non-dashboard tab (Chest X-ray, Brain MRI, Skin Lesion, YOLO, Clinical ML, XAI, RAG Assistant, Model Evaluation, History, Settings) detailing technical specifications, input requirements, architecture diagram, and "Stage 2 Pending" notice.

- [ ] **Step 1: Create module configuration dictionary with technical architecture details for all 10 modules**
- [ ] **Step 2: Implement `ModuleWorkbench` rendering technical specs, inputs, planned pipelines, and untrained status**
- [ ] **Step 3: Wire navigation in `frontend/src/App.tsx`**

---

### Task 10: Root Documentation & Setup Commands
**Files:**
- Create: `README.md`

**Interfaces:**
- Produces: Clear, comprehensive setup guide with exact Windows PowerShell commands for frontend and backend.

- [ ] **Step 1: Create root `README.md` covering architecture, prerequisites, environment variables, startup commands, and roadmap**

---

### Task 11: End-to-End Verification & Validation
- [ ] **Step 1: Start backend server and test `http://localhost:8000/api/health` and `http://localhost:8000/api/models/status`**
- [ ] **Step 2: Build and run frontend dev server, verifying communication with backend**
- [ ] **Step 3: Confirm all 11 navigation tabs, model cards, charts, and graceful DB fallback work seamlessly without errors**
