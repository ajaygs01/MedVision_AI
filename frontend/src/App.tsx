import { useState, useEffect, useCallback } from 'react';
import type { HealthResponse, ModelsStatusResponse, NavTabId } from './types';
import { fetchHealth, fetchModelsStatus } from './services/api';
import { Sidebar } from './components/layout/Sidebar';
import { Navbar } from './components/layout/Navbar';
import { DashboardView } from './components/dashboard/DashboardView';
import { ModuleWorkbench } from './components/workbench/ModuleWorkbench';
import { ChestXRayWorkbench } from './components/workbench/ChestXRayWorkbench';

const TAB_TITLES: Record<NavTabId, string> = {
  dashboard: 'Executive Research Dashboard',
  'chest-xray': 'Chest X-ray Multi-Abnormality Analysis',
  'brain-mri': 'Brain MRI Lesion & Tumor Segmentation',
  'skin-lesion': 'Skin Lesion Dermatoscopy Classifier',
  'yolo-detection': 'Anatomical Lesion YOLO Localization',
  'clinical-ml': 'Clinical Tabular Risk Predictor',
  'explainable-ai': 'Explainable AI (XAI) Attribution Suite',
  'research-assistant': 'Medical Research Assistant (RAG)',
  'model-evaluation': 'Model Evaluation & Benchmarking',
  'analysis-history': 'Analysis History & Audit Trail',
  settings: 'Platform Settings & Config',
};

export function App() {
  const [activeTab, setActiveTab] = useState<NavTabId>('dashboard');
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [modelsStatus, setModelsStatus] = useState<ModelsStatusResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [lastSyncTime, setLastSyncTime] = useState<string>('');

  const loadSystemState = useCallback(async (isManualRefresh = false) => {
    if (isManualRefresh) {
      setLoading(true);
    }
    try {
      const [healthRes, modelsRes] = await Promise.all([
        fetchHealth(),
        fetchModelsStatus(),
      ]);

      if (healthRes.data) {
        setHealth(healthRes.data);
      } else {
        // Fallback offline mock for UI responsiveness when server is paused
        setHealth({
          status: 'offline',
          version: '1.0.0-stage1',
          timestamp: new Date().toISOString(),
          database: {
            status: 'disconnected',
            engine: 'MongoDB Atlas',
            details: healthRes.error || 'Backend not responding',
          },
          environment: 'development',
        });
      }

      if (modelsRes.data) {
        setModelsStatus(modelsRes.data);
      }
      setLastSyncTime(new Date().toLocaleTimeString());
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    let isCurrent = true;

    async function initialSync() {
      const [healthRes, modelsRes] = await Promise.all([
        fetchHealth(),
        fetchModelsStatus(),
      ]);

      if (isCurrent) {
        if (healthRes.data) {
          setHealth(healthRes.data);
        } else {
          setHealth({
            status: 'offline',
            version: '1.0.0-stage1',
            timestamp: new Date().toISOString(),
            database: {
              status: 'disconnected',
              engine: 'MongoDB Atlas',
              details: healthRes.error || 'Backend not responding',
            },
            environment: 'development',
          });
        }

        if (modelsRes.data) {
          setModelsStatus(modelsRes.data);
        }
        setLastSyncTime(new Date().toLocaleTimeString());
        setLoading(false);
      }
    }

    initialSync();

    const interval = setInterval(() => {
      loadSystemState(false);
    }, 20000);

    return () => {
      isCurrent = false;
      clearInterval(interval);
    };
  }, [loadSystemState]);

  const isBackendOnline = health?.status === 'online';
  const dbStatus = health?.database?.status || 'disconnected';

  return (
    <div className="flex min-h-screen bg-[#070D1F] text-slate-100 antialiased selection:bg-teal-500/30 selection:text-teal-200">
      {/* Persistent Left Sidebar Navigation */}
      <Sidebar
        activeTab={activeTab}
        onSelectTab={setActiveTab}
        backendOnline={isBackendOnline}
        dbStatus={dbStatus}
      />

      {/* Main Content Area */}
      <div className="flex-1 flex flex-col min-w-0 bg-[#0B132B]">
        {/* Sticky Top Navigation Bar */}
        <Navbar
          currentTabName={TAB_TITLES[activeTab] || 'Dashboard'}
          health={health}
          loading={loading}
          onRefresh={() => loadSystemState(true)}
        />

        {/* Dynamic Viewport Container */}
        <main className="flex-1 p-6 md:p-8 max-w-7xl w-full mx-auto space-y-6">
          {activeTab === 'dashboard' ? (
            <DashboardView
              health={health}
              modelsStatus={modelsStatus}
              onNavigateToWorkbench={(tab) => setActiveTab(tab)}
            />
          ) : activeTab === 'chest-xray' ? (
            <ChestXRayWorkbench
              onBackToDashboard={() => setActiveTab('dashboard')}
              modelStatus={modelsStatus?.models?.cnn_chest_xray}
            />
          ) : (
            <ModuleWorkbench
              tabId={activeTab}
              onBackToDashboard={() => setActiveTab('dashboard')}
            />
          )}

          {/* Institutional Academic Footer */}
          <footer className="mt-12 pt-6 border-t border-slate-800/80 text-center text-xs text-slate-300 space-y-1">
            <p className="font-medium text-slate-400">
              MediVision AI • Final-Year Major Research Project • Stage 1 Software Architecture Foundation
            </p>
            <p className="text-[11px] text-slate-400">
              Strict Medical Integrity Safeguard: Predictive algorithms and weights are currently unloaded. No actual clinical diagnoses are executed.
            </p>
            {lastSyncTime && (
              <p className="text-[10px] text-slate-400 font-mono">
                Telemetry synchronized at {lastSyncTime}
              </p>
            )}
          </footer>
        </main>
      </div>
    </div>
  );
}

export default App;
