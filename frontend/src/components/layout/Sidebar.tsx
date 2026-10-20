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
    <aside className="w-64 bg-gradient-to-b from-[#890149] via-[#780140] to-[#59002E] text-[#F3F3E7] flex flex-col h-screen shrink-0 sticky top-0 select-none shadow-xl border-r border-[#6E013A] z-20">
      {/* Brand Header */}
      <div className="p-5 border-b border-white/10 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="h-10 w-10 rounded-xl bg-[#F3F3E7] flex items-center justify-center text-[#890149] font-bold shadow-md shadow-black/20">
            <Activity className="h-6 w-6 text-[#890149]" />
          </div>
          <div>
            <h1 className="font-bold text-base tracking-tight text-[#F3F3E7] flex items-center gap-1.5">
              MediVision <span className="text-[#F3F3E7]/90 font-light">AI</span>
            </h1>
            <p className="text-[10px] text-[#F3F3E7]/70 font-semibold tracking-wider uppercase">
              Clinical Research Platform
            </p>
          </div>
        </div>
      </div>

      {/* Navigation List */}
      <nav className="flex-1 overflow-y-auto px-3 py-4 space-y-5">
        {categories.map((category) => (
          <div key={category} className="space-y-1">
            <h3 className="px-3 text-[10px] font-bold tracking-wider text-[#F3F3E7]/60 uppercase">
              {category}
            </h3>
            <div className="mt-1 space-y-0.5">
              {NAV_ITEMS.filter((item) => item.category === category).map((item) => {
                const Icon = item.icon;
                const isActive = activeTab === item.id;

                return (
                  <button
                    key={item.id}
                    onClick={() => onSelectTab(item.id)}
                    className={`w-full flex items-center justify-between px-3 py-2 rounded-xl text-xs transition-all duration-150 group ${
                      isActive
                        ? 'bg-[#F3F3E7] text-[#890149] font-bold shadow-md shadow-black/15 translate-x-1'
                        : 'text-[#F3F3E7]/85 hover:text-white hover:bg-white/10 font-medium'
                    }`}
                  >
                    <div className="flex items-center gap-2.5 truncate">
                      <Icon
                        className={`h-4 w-4 shrink-0 transition-colors ${
                          isActive ? 'text-[#890149]' : 'text-[#F3F3E7]/75 group-hover:text-white'
                        }`}
                      />
                      <span className="truncate">{item.label}</span>
                    </div>

                    <div className="flex items-center gap-1.5">
                      {item.badge && (
                        <span
                          className={`text-[9px] px-1.5 py-0.5 rounded tracking-wide font-semibold ${
                            isActive
                              ? 'bg-[#890149]/15 text-[#890149]'
                              : 'bg-white/15 text-[#F3F3E7]/90 group-hover:bg-white/20'
                          }`}
                        >
                          {item.badge}
                        </span>
                      )}
                      {isActive && <ChevronRight className="h-3 w-3 text-[#890149]" />}
                    </div>
                  </button>
                );
              })}
            </div>
          </div>
        ))}
      </nav>

      {/* System Status Footer */}
      <div className="p-3 border-t border-white/10 bg-black/20 space-y-2">
        <div className="flex items-center justify-between text-[11px] px-2.5 py-1.5 rounded-lg bg-black/20 border border-white/10">
          <span className="text-[#F3F3E7]/90 flex items-center gap-1.5">
            <span
              className={`w-2 h-2 rounded-full ${
                backendOnline ? 'bg-emerald-400 animate-status-pulse' : 'bg-rose-400'
              }`}
            />
            FastAPI Server
          </span>
          <span className={backendOnline ? 'text-emerald-300 font-mono text-[10px] font-semibold' : 'text-rose-300 font-mono text-[10px]'}>
            {backendOnline ? 'Online' : 'Offline'}
          </span>
        </div>

        <div className="flex items-center justify-between text-[11px] px-2.5 py-1.5 rounded-lg bg-black/20 border border-white/10">
          <span className="text-[#F3F3E7]/90 flex items-center gap-1.5">
            <Database className="h-3 w-3 text-[#F3F3E7]/70" />
            MongoDB
          </span>
          <span className="text-[#F3F3E7]/75 font-mono text-[10px] truncate max-w-[100px]">
            {dbStatus === 'connected' ? 'Atlas Connected' : 'Unconfigured'}
          </span>
        </div>
      </div>
    </aside>
  );
};
