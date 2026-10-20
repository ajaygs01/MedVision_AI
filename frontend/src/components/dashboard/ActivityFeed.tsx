import React from 'react';
import { History, CheckCircle2, ShieldCheck, Database, Radio, Cpu } from 'lucide-react';

interface ActivityItem {
  id: string;
  timestamp: string;
  title: string;
  description: string;
  icon: React.ComponentType<{ className?: string }>;
  tag: string;
  tagColor: string;
}

const ACTIVITIES: ActivityItem[] = [
  {
    id: 'act-1',
    timestamp: 'Just now',
    title: 'Stage 1 Architecture Initialized',
    description: 'Vite React TypeScript dashboard and FastAPI backend scaffolded with clean dependency isolation.',
    icon: ShieldCheck,
    tag: 'Platform',
    tagColor: 'text-teal-300 bg-teal-500/10 border-teal-500/30',
  },
  {
    id: 'act-2',
    timestamp: '1 min ago',
    title: 'Model Status Router Online',
    description: 'Synchronized GET /api/models/status covering 7 research pipelines (CNN, YOLO, U-Net, Tabular, XAI, RAG).',
    icon: Cpu,
    tag: 'Backend API',
    tagColor: 'text-cyan-300 bg-cyan-500/10 border-cyan-500/30',
  },
  {
    id: 'act-3',
    timestamp: '2 mins ago',
    title: 'MongoDB Atlas Fallback Configured',
    description: 'Configurable database client instantiated. Soft-fallback enabled for offline/development environments.',
    icon: Database,
    tag: 'Database',
    tagColor: 'text-amber-300 bg-amber-500/10 border-amber-500/30',
  },
  {
    id: 'act-4',
    timestamp: '3 mins ago',
    title: 'Verification Engine Active',
    description: 'System-wide guardrail active: all predictive models locked to unconfigured state until Stage 2 datasets are ingested.',
    icon: Radio,
    tag: 'Governance',
    tagColor: 'text-emerald-300 bg-emerald-500/10 border-emerald-500/30',
  },
];

export const ActivityFeed: React.FC = () => {
  return (
    <div className="rounded-xl border border-slate-800 bg-[#0C1430]/70 p-5 flex flex-col justify-between">
      <div>
        <div className="flex items-center justify-between pb-3 border-b border-slate-800">
          <div className="flex items-center gap-2">
            <History className="h-4 w-4 text-teal-400" />
            <h3 className="text-sm font-semibold text-white tracking-tight">
              Recent System & Audit Activity
            </h3>
          </div>
          <span className="text-[11px] text-slate-400 font-mono">Stage 1 Event Log</span>
        </div>

        <div className="mt-4 space-y-4">
          {ACTIVITIES.map((item) => {
            const Icon = item.icon;
            return (
              <div key={item.id} className="flex items-start gap-3 group">
                <div className="p-1.5 rounded-md bg-slate-900 border border-slate-800 text-teal-400 shrink-0 mt-0.5 group-hover:border-teal-500/30 transition-colors">
                  <Icon className="h-3.5 w-3.5" />
                </div>
                <div className="flex-1 min-w-0">
                  <div className="flex items-center justify-between gap-2">
                    <h4 className="text-xs font-semibold text-slate-200 truncate">
                      {item.title}
                    </h4>
                    <span className="text-[10px] text-slate-400 shrink-0 font-mono">
                      {item.timestamp}
                    </span>
                  </div>
                  <p className="mt-0.5 text-[11px] text-slate-300 leading-relaxed">
                    {item.description}
                  </p>
                  <div className="mt-1.5">
                    <span
                      className={`inline-block px-1.5 py-0.5 rounded text-[9px] font-semibold border ${item.tagColor}`}
                    >
                      {item.tag}
                    </span>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      <div className="mt-4 pt-3 border-t border-slate-800/80 flex items-center justify-between text-[11px] text-slate-400">
        <span className="flex items-center gap-1.5">
          <CheckCircle2 className="h-3.5 w-3.5 text-teal-400" />
          Audit trail active
        </span>
        <span className="font-mono text-[10px] text-slate-400">Stage 1 Compliant</span>
      </div>
    </div>
  );
};
