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
          <h3 className="text-base font-semibold text-white tracking-tight flex items-center gap-2">
            <Layers className="h-4 w-4 text-teal-400" />
            AI Research Pipelines & Model Configurations
          </h3>
          <p className="text-xs text-slate-400">
            Real-time synchronization with FastAPI endpoint <code className="text-teal-300">/api/models/status</code>
          </p>
        </div>

        <div className="flex items-center gap-2">
          <span className="text-[11px] text-amber-300 bg-amber-950/60 border border-amber-500/30 px-2.5 py-1 rounded-md flex items-center gap-1.5 font-medium">
            <ShieldAlert className="h-3.5 w-3.5 text-amber-400" />
            All Weights Unloaded (Stage 1)
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
              className="rounded-xl border border-slate-800 bg-[#0C1430]/70 p-4 flex flex-col justify-between hover:border-teal-500/40 transition-all hover:bg-[#0F1A3D]/70 group"
            >
              <div>
                {/* Header: Icon, Name, and Status */}
                <div className="flex items-start justify-between gap-2">
                  <div className="flex items-center gap-2.5">
                    <div className="p-2 rounded-lg bg-teal-500/10 text-teal-400 border border-teal-500/20 group-hover:bg-teal-500/20 transition-colors">
                      <IconComponent className="h-4 w-4" />
                    </div>
                    <div>
                      <h4 className="text-sm font-semibold text-slate-100 group-hover:text-teal-200 transition-colors">
                        {model.name}
                      </h4>
                      <p className="text-[11px] text-teal-300/80 font-medium">{model.modality}</p>
                    </div>
                  </div>

                  <StatusBadge status="untrained" label="NOT TRAINED" />
                </div>

                {/* Description */}
                <p className="mt-3 text-xs text-slate-300 leading-relaxed line-clamp-2">
                  {model.description}
                </p>

                {/* Technical Specs Tags */}
                <div className="mt-3.5 space-y-1.5 border-t border-slate-800/80 pt-3 text-[11px]">
                  <div className="flex items-center justify-between text-slate-400">
                    <span className="text-slate-400">Training State:</span>
                    <span className="text-amber-400 font-bold font-mono text-[10px]">
                      NOT TRAINED (Weights Unloaded)
                    </span>
                  </div>
                  <div className="flex items-center justify-between text-slate-400">
                    <span className="text-slate-400">Architecture:</span>
                    <span className="text-slate-200 font-mono text-[10px] truncate max-w-[170px]" title={model.architecture}>
                      {model.architecture}
                    </span>
                  </div>

                  <div className="flex items-center justify-between text-slate-400">
                    <span className="text-slate-400">Framework:</span>
                    <span className="text-teal-300 font-mono text-[10px]">{model.framework}</span>
                  </div>

                  <div className="flex items-center justify-between text-slate-400">
                    <span className="text-slate-400">Next Milestone:</span>
                    <span className="text-amber-300 text-[10px] truncate max-w-[170px]" title={model.next_stage}>
                      {model.next_stage}
                    </span>
                  </div>
                </div>
              </div>

              {/* Action Button */}
              <button
                onClick={() => onNavigateToWorkbench(mapping.tab)}
                className="mt-4 w-full py-2 px-3 rounded-lg bg-slate-900 hover:bg-teal-600/20 border border-slate-800 hover:border-teal-500/40 text-xs font-medium text-slate-300 hover:text-teal-200 flex items-center justify-center gap-1.5 transition-all"
              >
                <span>View Stage 2 Blueprint</span>
                <ArrowRight className="h-3.5 w-3.5 group-hover:translate-x-0.5 transition-transform" />
              </button>
            </div>
          );
        })}
      </div>
    </div>
  );
};
