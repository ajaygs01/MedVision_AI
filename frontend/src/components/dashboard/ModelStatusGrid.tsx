import React from 'react';
import {
  FileHeart,
  Brain,
  Microscope,
  Crosshair,
  TableProperties,
  Eye,
  Bot,
  Layers,
  ArrowRight,
  ShieldAlert,
} from 'lucide-react';
import type { ModelsStatusResponse, NavTabId } from '../../types';
import { StatusBadge } from '../common/StatusBadge';

interface ModelStatusGridProps {
  modelsStatus: ModelsStatusResponse | null;
  onNavigateToWorkbench: (tab: NavTabId) => void;
}

const MODEL_ICON_MAP: Record<string, { icon: React.ComponentType<{ className?: string }>; tab: NavTabId }> = {
  cnn_chest_xray: { icon: FileHeart, tab: 'chest-xray' },
  unet_brain_mri: { icon: Brain, tab: 'brain-mri' },
  cnn_skin_lesion: { icon: Microscope, tab: 'skin-lesion' },
  yolo_detection: { icon: Crosshair, tab: 'yolo-detection' },
  clinical_ml: { icon: TableProperties, tab: 'clinical-ml' },
  xai_engine: { icon: Eye, tab: 'explainable-ai' },
  rag_assistant: { icon: Bot, tab: 'research-assistant' },
};

export const ModelStatusGrid: React.FC<ModelStatusGridProps> = ({
  modelsStatus,
  onNavigateToWorkbench,
}) => {
  const models = modelsStatus?.models || {};

  return (
    <div className="space-y-4">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
        <div>
          <h3 className="text-base font-bold text-[#1A1512] tracking-tight flex items-center gap-2">
            <Layers className="h-5 w-5 text-[#014BAA]" />
            AI Research Pipelines & Architecture Status
          </h3>
          <p className="text-xs text-[#61554E]">
            Real-time status synced with FastAPI endpoint <code className="text-[#014BAA] font-semibold">/api/models/status</code>
          </p>
        </div>

        <div className="flex items-center gap-2">
          <span className="text-[11px] text-[#014BAA] bg-[#014BAA]/10 border border-[#014BAA]/25 px-3 py-1 rounded-full flex items-center gap-1.5 font-bold">
            <ShieldAlert className="h-3.5 w-3.5 text-[#014BAA]" />
            Weights Integrity Guard Active
          </span>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
        {Object.entries(models).map(([key, model]) => {
          const mapping = MODEL_ICON_MAP[key] || { icon: Layers, tab: 'dashboard' as NavTabId };
          const IconComponent = mapping.icon;

          return (
            <div
              key={key}
              className="rounded-2xl border border-[#EAE3DD] bg-white/95 p-5 flex flex-col justify-between hover:border-[#014BAA]/50 hover:shadow-md transition-all group"
            >
              <div>
                {/* Header: Icon, Name, and Status */}
                <div className="flex items-start justify-between gap-2">
                  <div className="flex items-center gap-3">
                    <div className="p-2.5 rounded-xl bg-[#014BAA]/10 text-[#014BAA] border border-[#014BAA]/20 group-hover:bg-[#014BAA] group-hover:text-[#F8F3F0] transition-all">
                      <IconComponent className="h-5 w-5" />
                    </div>
                    <div>
                      <h4 className="text-sm font-bold text-[#1A1512] group-hover:text-[#014BAA] transition-colors">
                        {model.name}
                      </h4>
                      <p className="text-[11px] text-[#014BAA] font-medium">{model.modality}</p>
                    </div>
                  </div>

                  <StatusBadge status="untrained" label="NOT TRAINED" />
                </div>

                {/* Description */}
                <p className="mt-3 text-xs text-[#61554E] leading-relaxed line-clamp-2">
                  {model.description}
                </p>

                {/* Technical Specs Tags */}
                <div className="mt-4 space-y-1.5 border-t border-[#F0EAE5] pt-3 text-[11px]">
                  <div className="flex items-center justify-between">
                    <span className="text-[#7C6E66]">Training State:</span>
                    <span className="text-amber-700 font-bold font-mono text-[10px]">
                      NOT TRAINED (Weights Pending)
                    </span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-[#7C6E66]">Architecture:</span>
                    <span className="text-[#1A1512] font-mono text-[10px] truncate max-w-[170px]" title={model.architecture}>
                      {model.architecture}
                    </span>
                  </div>

                  <div className="flex items-center justify-between">
                    <span className="text-[#7C6E66]">Framework:</span>
                    <span className="text-[#014BAA] font-mono text-[10px] font-semibold">{model.framework}</span>
                  </div>

                  <div className="flex items-center justify-between">
                    <span className="text-[#7C6E66]">Next Milestone:</span>
                    <span className="text-[#61554E] text-[10px] truncate max-w-[170px]" title={model.next_stage}>
                      {model.next_stage}
                    </span>
                  </div>
                </div>
              </div>

              {/* Action Button */}
              <button
                onClick={() => onNavigateToWorkbench(mapping.tab)}
                className="mt-4 w-full py-2 px-3 rounded-xl bg-[#F8F3F0] hover:bg-[#014BAA] hover:text-[#F8F3F0] border border-[#EAE3DD] hover:border-[#014BAA] text-xs font-bold text-[#1A1512] flex items-center justify-center gap-1.5 transition-all shadow-xs"
              >
                <span>Open Research Workbench</span>
                <ArrowRight className="h-3.5 w-3.5 group-hover:translate-x-0.5 transition-transform" />
              </button>
            </div>
          );
        })}
      </div>
    </div>
  );
};
