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
    <div className="flex h-screen w-full bg-[#F8F3F0] text-[#1A1512] antialiased selection:bg-[#014BAA]/20 selection:text-[#014BAA] relative overflow-hidden">
      {/* Background Ambient Aura Orbs & Medical Grid */}
      <div className="fixed inset-0 pointer-events-none overflow-hidden z-0">
        <div className="absolute -top-32 -right-32 w-[600px] h-[600px] rounded-full bg-[#014BAA]/12 blur-[130px] animate-float-slow" />
        <div className="absolute top-1/2 -left-32 w-[520px] h-[520px] rounded-full bg-[#1E6BEB]/10 blur-[140px] animate-float-reverse" />
        <div className="absolute -bottom-32 right-1/4 w-[500px] h-[500px] rounded-full bg-[#014BAA]/8 blur-[120px]" />
        <div className="absolute inset-0 bg-medical-grid opacity-60" />
      </div>

      {/* Persistent Left Sidebar Navigation in Deep Royal Blue - Fixed in place */}
      <Sidebar
        activeTab={activeTab}
        onSelectTab={setActiveTab}
        backendOnline={isBackendOnline}
        dbStatus={dbStatus}
      />

      {/* Main Content Area */}
      <div className="flex-1 flex flex-col h-full min-w-0 overflow-hidden relative z-10">
        {/* Sticky/Fixed Top Navigation Bar */}
        <Navbar
          currentTabName={TAB_TITLES[activeTab] || 'Dashboard'}
          health={health}
          loading={loading}
          onRefresh={() => loadSystemState(true)}
        />

        {/* Scrollable Viewport Container - Only this area scrolls! */}
        <div className="flex-1 overflow-y-auto overflow-x-hidden">
          <main className="p-6 md:p-8 max-w-7xl w-full mx-auto space-y-6">
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
            <footer className="mt-12 pt-6 border-t border-[#EAE3DD] text-center text-xs text-[#61554E] space-y-1">
              <p className="font-semibold text-[#014BAA]">
                MediVision AI • Final-Year Major Research Project • Multi-Disease Biomedical Intelligence
              </p>
              <p className="text-[11px] text-[#7C6E66]">
                Strict Medical Integrity Safeguard: Predictive algorithms are research prototypes. No clinical diagnosis is executed without validated human verification.
              </p>
              {lastSyncTime && (
                <p className="text-[10px] text-[#8A7B73] font-mono">
                  Telemetry synchronized at {lastSyncTime}
                </p>
              )}
            </footer>
          </main>
        </div>
      </div>
    </div>
  );
}

export default App;
