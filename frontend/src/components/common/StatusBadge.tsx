import React from 'react';

export interface StatusBadgeProps {
  status: 'online' | 'offline' | 'pending' | 'untrained' | 'connected' | 'disconnected' | 'info';
  label?: string;
  pulse?: boolean;
}

export const StatusBadge: React.FC<StatusBadgeProps> = ({ status, label, pulse = false }) => {
  const getBadgeStyle = () => {
    switch (status) {
      case 'online':
      case 'connected':
        return {
          bg: 'bg-emerald-950/60 border-emerald-500/40 text-emerald-300',
          dot: 'bg-emerald-400',
          defaultText: status === 'online' ? 'Online' : 'Connected',
        };
      case 'untrained':
      case 'pending':
        return {
          bg: 'bg-amber-950/70 border-amber-500/50 text-amber-300 font-bold',
          dot: 'bg-amber-400',
          defaultText: status === 'untrained' ? 'NOT TRAINED' : 'Pending',
        };
      case 'offline':
      case 'disconnected':
        return {
          bg: 'bg-rose-950/60 border-rose-500/40 text-rose-300',
          dot: 'bg-rose-400',
          defaultText: status === 'offline' ? 'Offline' : 'Disconnected',
        };
      case 'info':
      default:
        return {
          bg: 'bg-cyan-950/60 border-cyan-500/40 text-cyan-300',
          dot: 'bg-cyan-400',
          defaultText: 'Information',
        };
    }
  };

  const style = getBadgeStyle();
  const displayText = label || style.defaultText;

  return (
    <span
      className={`inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-xs font-medium border tracking-wide transition-all ${style.bg}`}
    >
      <span
        className={`w-1.5 h-1.5 rounded-full ${style.dot} ${pulse ? 'animate-status-pulse' : ''}`}
      />
      {displayText}
    </span>
  );
};
