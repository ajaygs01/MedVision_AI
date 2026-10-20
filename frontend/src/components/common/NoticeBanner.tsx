import React from 'react';
import { AlertTriangle, CheckCircle2, Info } from 'lucide-react';

interface NoticeBannerProps {
  type?: 'stage1' | 'warning' | 'info';
  title?: string;
  message?: string;
}

export const NoticeBanner: React.FC<NoticeBannerProps> = ({
  type = 'stage1',
  title,
  message,
}) => {
  if (type === 'stage1') {
    return (
      <div className="relative overflow-hidden rounded-xl border border-teal-500/30 bg-gradient-to-r from-teal-950/40 via-slate-900/60 to-slate-900/40 p-4 shadow-lg backdrop-blur-sm">
        <div className="flex items-start gap-3">
          <div className="mt-0.5 rounded-lg bg-teal-500/20 p-1.5 text-teal-400">
            <CheckCircle2 className="h-5 w-5" />
          </div>
          <div className="flex-1">
            <div className="flex items-center gap-2">
              <h4 className="text-sm font-semibold tracking-wide text-teal-200">
                {title || 'Stage 1 Architectural Foundation Active'}
              </h4>
              <span className="rounded bg-teal-500/20 px-2 py-0.5 text-[10px] font-semibold text-teal-300 uppercase tracking-wider">
                Integrity Safeguard
              </span>
            </div>
            <p className="mt-1 text-xs leading-relaxed text-slate-300">
              {message ||
                'This platform is configured for multi-modal medical AI research. In accordance with Stage 1 constraints, all neural networks and prediction pipelines remain unconfigured and untrained. No mock or pseudo-clinical predictions are executed.'}
            </p>
          </div>
        </div>
      </div>
    );
  }

  if (type === 'warning') {
    return (
      <div className="rounded-xl border border-amber-500/30 bg-amber-950/30 p-4 text-amber-200">
        <div className="flex items-start gap-3">
          <AlertTriangle className="h-5 w-5 text-amber-400 shrink-0 mt-0.5" />
          <div>
            <h4 className="text-sm font-semibold">{title || 'Attention Required'}</h4>
            <p className="mt-1 text-xs text-amber-300/80">{message}</p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="rounded-xl border border-cyan-500/30 bg-cyan-950/30 p-4 text-cyan-200">
      <div className="flex items-start gap-3">
        <Info className="h-5 w-5 text-cyan-400 shrink-0 mt-0.5" />
        <div>
          <h4 className="text-sm font-semibold">{title}</h4>
          <p className="mt-1 text-xs text-cyan-300/80">{message}</p>
        </div>
      </div>
    </div>
  );
};
