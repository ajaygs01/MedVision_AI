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
          <h3 className="text-base font-bold text-[#1E161C] tracking-tight flex items-center gap-2">
            <Layers className="h-5 w-5 text-[#890149]" />
            AI Research Pipelines & Architecture Status
          </h3>
          <p className="text-xs text-[#5C4E57]">
            Real-time status synced with FastAPI endpoint <code className="text-[#890149] font-semibold">/api/models/status</code>
          </p>
        </div>

        <div className="flex items-center gap-2">
          <span className="text-[11px] text-[#890149] bg-[#890149]/10 border border-[#890149]/25 px-3 py-1 rounded-full flex items-center gap-1.5 font-bold">
            <ShieldAlert className="h-3.5 w-3.5 text-[#890149]" />
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
              className="rounded-2xl border border-[#E5E5D8] bg-white/95 p-5 flex flex-col justify-between hover:border-[#890149]/50 hover:shadow-md transition-all group"
            >
              <div>
                {/* Header: Icon, Name, and Status */}
                <div className="flex items-start justify-between gap-2">
                  <div className="flex items-center gap-3">
                    <div className="p-2.5 rounded-xl bg-[#890149]/10 text-[#890149] border border-[#890149]/20 group-hover:bg-[#890149] group-hover:text-[#F3F3E7] transition-all">
                      <IconComponent className="h-5 w-5" />
                    </div>
                    <div>
                      <h4 className="text-sm font-bold text-[#1E161C] group-hover:text-[#890149] transition-colors">
                        {model.name}
                      </h4>
                      <p className="text-[11px] text-[#890149] font-medium">{model.modality}</p>
                    </div>
                  </div>

                  <StatusBadge status="untrained" label="NOT TRAINED" />
                </div>

                {/* Description */}
                <p className="mt-3 text-xs text-[#5C4E57] leading-relaxed line-clamp-2">
                  {model.description}
                </p>

                {/* Technical Specs Tags */}
                <div className="mt-4 space-y-1.5 border-t border-[#F0F0E4] pt-3 text-[11px]">
                  <div className="flex items-center justify-between">
                    <span className="text-[#7A6B74]">Training State:</span>
                    <span className="text-amber-700 font-bold font-mono text-[10px]">
                      NOT TRAINED (Weights Pending)
                    </span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-[#7A6B74]">Architecture:</span>
                    <span className="text-[#1E161C] font-mono text-[10px] truncate max-w-[170px]" title={model.architecture}>
                      {model.architecture}
                    </span>
                  </div>

                  <div className="flex items-center justify-between">
                    <span className="text-[#7A6B74]">Framework:</span>
                    <span className="text-[#890149] font-mono text-[10px] font-semibold">{model.framework}</span>
                  </div>

                  <div className="flex items-center justify-between">
                    <span className="text-[#7A6B74]">Next Milestone:</span>
                    <span className="text-[#5C4E57] text-[10px] truncate max-w-[170px]" title={model.next_stage}>
                      {model.next_stage}
                    </span>
                  </div>
                </div>
              </div>

              {/* Action Button */}
              <button
                onClick={() => onNavigateToWorkbench(mapping.tab)}
                className="mt-4 w-full py-2 px-3 rounded-xl bg-[#F3F3E7] hover:bg-[#890149] hover:text-[#F3F3E7] border border-[#E5E5D8] hover:border-[#890149] text-xs font-bold text-[#1E161C] flex items-center justify-center gap-1.5 transition-all shadow-xs"
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
