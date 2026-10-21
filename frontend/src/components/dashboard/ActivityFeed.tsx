import React from 'react';
import { History, ShieldCheck, Database, Radio, Cpu } from 'lucide-react';

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
    title: 'RSNA Pneumonia Dataset Verified',
    description: '29,684 DICOM radiographs indexed in datasets/rsna_pneumonia. Patient-level stratification verified with 0 leakage.',
    icon: ShieldCheck,
    tag: 'Dataset',
    tagColor: 'text-[#014BAA] bg-[#014BAA]/10 border-[#014BAA]/25',
  },
  {
    id: 'act-2',
    timestamp: '2 mins ago',
    title: 'ResNet-50 CNN Pipeline Active',
    description: 'PyTorch transfer learning architecture, BCEWithLogitsLoss class balancing, and inference service compiled.',
    icon: Cpu,
    tag: 'Stage 2 CNN',
    tagColor: 'text-[#014BAA] bg-[#014BAA]/10 border-[#014BAA]/25',
  },
  {
    id: 'act-3',
    timestamp: '5 mins ago',
    title: 'Model Status Endpoint Synced',
    description: 'POST /api/imaging/chest-xray/classify operational with strict NOT TRAINED safeguard for unloaded weights.',
    icon: Radio,
    tag: 'FastAPI',
    tagColor: 'text-emerald-800 bg-emerald-50 border-emerald-300',
  },
  {
    id: 'act-4',
    timestamp: '10 mins ago',
    title: 'Database Fallback Active',
    description: 'MongoDB Atlas client configured with graceful degradation when credentials are not supplied.',
    icon: Database,
    tag: 'Database',
    tagColor: 'text-amber-800 bg-amber-50 border-amber-300',
  },
];

export const ActivityFeed: React.FC = () => {
  return (
    <div className="rounded-2xl border border-[#EAE3DD] bg-white/95 p-5 shadow-sm">
      <div className="flex items-center justify-between pb-3 border-b border-[#EAE3DD]">
        <div className="flex items-center gap-2">
          <History className="h-4 w-4 text-[#014BAA]" />
          <h3 className="text-sm font-bold text-[#1A1512] tracking-tight">
            Recent System & Audit Activity
          </h3>
        </div>
        <span className="text-[11px] text-[#61554E] font-mono">Stage 2 Event Log</span>
      </div>

      <div className="mt-4 space-y-4">
        {ACTIVITIES.map((item) => {
          const Icon = item.icon;
          return (
            <div key={item.id} className="flex items-start gap-3 group">
              <div className="p-2 rounded-xl bg-[#014BAA]/10 text-[#014BAA] border border-[#014BAA]/20 shrink-0 mt-0.5 group-hover:bg-[#014BAA] group-hover:text-[#F8F3F0] transition-all">
                <Icon className="h-3.5 w-3.5" />
              </div>
              <div className="flex-1 min-w-0">
                <div className="flex items-center justify-between gap-2">
                  <h4 className="text-xs font-bold text-[#1A1512] truncate group-hover:text-[#014BAA] transition-colors">
                    {item.title}
                  </h4>
                  <span className="text-[10px] text-[#61554E] shrink-0 font-mono">
                    {item.timestamp}
                  </span>
                </div>
                <p className="mt-0.5 text-[11px] text-[#61554E] leading-relaxed">
                  {item.description}
                </p>
                <div className="mt-1.5">
                  <span
                    className={`inline-block px-2 py-0.5 rounded-full text-[9px] font-bold border ${item.tagColor}`}
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
  );
};
