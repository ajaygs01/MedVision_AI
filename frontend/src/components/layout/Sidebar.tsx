import React from 'react';
import {
  LayoutDashboard,
  FileHeart,
  Brain,
  Microscope,
  Crosshair,
  TableProperties,
  Eye,
  Bot,
  BarChart3,
  History,
  Settings,
  Activity,
  ChevronRight,
  Database,
} from 'lucide-react';
import type { NavTabId } from '../../types';

interface SidebarProps {
  activeTab: NavTabId;
  onSelectTab: (tab: NavTabId) => void;
  backendOnline: boolean;
  dbStatus: string;
}

interface NavItem {
  id: NavTabId;
  label: string;
  icon: React.ComponentType<{ className?: string }>;
  badge?: string;
  category: string;
}

const NAV_ITEMS: NavItem[] = [
  {
    id: 'dashboard',
    label: 'Dashboard',
    icon: LayoutDashboard,
    category: 'Overview',
  },
  {
    id: 'chest-xray',
    label: 'Chest X-ray Analysis',
    icon: FileHeart,
    badge: 'Stage 2',
    category: 'Imaging Modalities',
  },
  {
    id: 'brain-mri',
    label: 'Brain MRI Analysis',
    icon: Brain,
    badge: 'Stage 2',
    category: 'Imaging Modalities',
  },
  {
    id: 'skin-lesion',
    label: 'Skin Lesion Analysis',
    icon: Microscope,
    badge: 'Stage 2',
    category: 'Imaging Modalities',
  },
  {
    id: 'yolo-detection',
    label: 'YOLO Detection',
    icon: Crosshair,
    badge: 'Stage 3',
    category: 'Imaging Modalities',
  },
  {
    id: 'clinical-ml',
    label: 'Clinical ML',
    icon: TableProperties,
    badge: 'Stage 3',
    category: 'Predictive & Tabular',
  },
  {
    id: 'explainable-ai',
    label: 'Explainable AI',
    icon: Eye,
    badge: 'Stage 4',
    category: 'Interpretability & Copilot',
  },
  {
    id: 'research-assistant',
    label: 'Medical Research Assistant',
    icon: Bot,
    badge: 'Stage 4',
    category: 'Interpretability & Copilot',
  },
  {
    id: 'model-evaluation',
    label: 'Model Evaluation',
    icon: BarChart3,
    badge: 'Stage 2',
    category: 'Research Governance',
  },
  {
    id: 'analysis-history',
    label: 'Analysis History',
    icon: History,
    category: 'Research Governance',
  },
  {
    id: 'settings',
    label: 'Settings',
    icon: Settings,
    category: 'Research Governance',
  },
];

export const Sidebar: React.FC<SidebarProps> = ({
  activeTab,
  onSelectTab,
  backendOnline,
  dbStatus,
}) => {
  const categories = Array.from(new Set(NAV_ITEMS.map((item) => item.category)));

  return (
    <aside className="w-64 bg-[#070D1F] border-r border-slate-800 flex flex-col h-screen shrink-0 sticky top-0 select-none">
      {/* Brand Header */}
      <div className="p-5 border-b border-slate-800 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="h-10 w-10 rounded-xl bg-gradient-to-br from-teal-500 to-teal-700 flex items-center justify-center text-slate-950 font-bold shadow-md shadow-teal-900/30">
            <Activity className="h-6 w-6 text-slate-950" />
          </div>
          <div>
            <h1 className="font-bold text-base tracking-tight text-white flex items-center gap-1.5">
              MediVision <span className="text-teal-400">AI</span>
            </h1>
            <p className="text-[10px] text-teal-300/80 font-medium tracking-wide uppercase">
              Major Project • Stage 1
            </p>
          </div>
        </div>
      </div>

      {/* Navigation List */}
      <nav className="flex-1 overflow-y-auto px-3 py-4 space-y-6">
        {categories.map((category) => (
          <div key={category} className="space-y-1">
            <h3 className="px-3 text-[11px] font-semibold tracking-wider text-slate-400 uppercase">
              {category}
            </h3>
            <div className="mt-1.5 space-y-0.5">
              {NAV_ITEMS.filter((item) => item.category === category).map((item) => {
                const Icon = item.icon;
                const isActive = activeTab === item.id;

                return (
                  <button
                    key={item.id}
                    onClick={() => onSelectTab(item.id)}
                    className={`w-full flex items-center justify-between px-3 py-2 rounded-lg text-xs font-medium transition-all group ${
                      isActive
                        ? 'bg-gradient-to-r from-teal-500/20 to-teal-500/5 text-teal-200 border-l-2 border-teal-400 font-semibold'
                        : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/40'
                    }`}
                  >
                    <div className="flex items-center gap-2.5 truncate">
                      <Icon
                        className={`h-4 w-4 shrink-0 transition-colors ${
                          isActive ? 'text-teal-400' : 'text-slate-400 group-hover:text-slate-200'
                        }`}
                      />
                      <span className="truncate">{item.label}</span>
                    </div>

                    <div className="flex items-center gap-1.5">
                      {item.badge && (
                        <span
                          className={`text-[9px] px-1.5 py-0.5 rounded tracking-wide font-medium ${
                            isActive
                              ? 'bg-teal-500/30 text-teal-200'
                              : 'bg-slate-800 text-slate-400 group-hover:bg-slate-700/60'
                          }`}
                        >
                          {item.badge}
                        </span>
                      )}
                      {isActive && <ChevronRight className="h-3 w-3 text-teal-400" />}
                    </div>
                  </button>
                );
              })}
            </div>
          </div>
        ))}
      </nav>

      {/* System Status Footer */}
      <div className="p-3 border-t border-slate-800/80 bg-[#060A18]/60 space-y-2">
        <div className="flex items-center justify-between text-[11px] px-2 py-1 rounded bg-slate-900/60 border border-slate-800">
          <span className="text-slate-400 flex items-center gap-1.5">
            <span
              className={`w-2 h-2 rounded-full ${
                backendOnline ? 'bg-emerald-400 animate-status-pulse' : 'bg-rose-500'
              }`}
            />
            FastAPI Server
          </span>
          <span className={backendOnline ? 'text-emerald-300 font-mono text-[10px]' : 'text-rose-400 font-mono text-[10px]'}>
            {backendOnline ? ':8000 Online' : 'Offline'}
          </span>
        </div>

        <div className="flex items-center justify-between text-[11px] px-2 py-1 rounded bg-slate-900/60 border border-slate-800">
          <span className="text-slate-400 flex items-center gap-1.5">
            <Database className="h-3 w-3 text-cyan-400" />
            MongoDB
          </span>
          <span className="text-slate-300 font-mono text-[10px] truncate max-w-[100px]">
            {dbStatus === 'connected' ? 'Atlas Connected' : 'Unconfigured'}
          </span>
        </div>
      </div>
    </aside>
  );
};
