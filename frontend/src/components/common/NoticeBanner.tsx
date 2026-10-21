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
      <div className="relative overflow-hidden rounded-xl border border-[#014BAA]/20 bg-white/90 p-4 shadow-sm backdrop-blur-sm">
        <div className="flex items-start gap-3">
          <div className="mt-0.5 rounded-lg bg-[#014BAA]/10 p-1.5 text-[#014BAA]">
            <CheckCircle2 className="h-5 w-5" />
          </div>
          <div className="flex-1">
            <div className="flex items-center gap-2">
              <h4 className="text-sm font-bold tracking-tight text-[#014BAA]">
                {title || 'MediVision AI • Research Intelligence Operational'}
              </h4>
              <span className="rounded-full bg-[#014BAA]/10 px-2 py-0.5 text-[10px] font-bold text-[#014BAA] uppercase tracking-wider">
                Clinical Integrity
              </span>
            </div>
            <p className="mt-1 text-xs leading-relaxed text-[#61554E]">
              {message ||
                'This platform is configured for multi-modal medical AI research. In accordance with clinical protocols, untrained models strictly report NOT TRAINED without generating fabricated predictions.'}
            </p>
          </div>
        </div>
      </div>
    );
  }

  if (type === 'warning') {
    return (
      <div className="rounded-xl border border-amber-500/30 bg-amber-50/80 p-4 text-amber-900 shadow-xs">
        <div className="flex items-start gap-3">
          <AlertTriangle className="h-5 w-5 text-amber-600 shrink-0 mt-0.5" />
          <div>
            <h4 className="text-sm font-bold text-amber-900">{title || 'Attention Required'}</h4>
            <p className="mt-1 text-xs text-amber-800">{message}</p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="rounded-xl border border-[#014BAA]/20 bg-white/80 p-4 text-[#1A1512] shadow-xs">
      <div className="flex items-start gap-3">
        <Info className="h-5 w-5 text-[#014BAA] shrink-0 mt-0.5" />
        <div>
          <h4 className="text-sm font-bold text-[#014BAA]">{title}</h4>
          <p className="mt-1 text-xs text-[#61554E]">{message}</p>
        </div>
      </div>
    </div>
  );
};
