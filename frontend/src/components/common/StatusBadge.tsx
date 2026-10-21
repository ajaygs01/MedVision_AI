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
          bg: 'bg-emerald-50 border-emerald-300 text-emerald-800',
          dot: 'bg-emerald-600',
          defaultText: status === 'online' ? 'Online' : 'Connected',
        };
      case 'untrained':
      case 'pending':
        return {
          bg: 'bg-amber-50 border-amber-300 text-amber-900 font-bold',
          dot: 'bg-amber-500',
          defaultText: status === 'untrained' ? 'NOT TRAINED' : 'Pending',
        };
      case 'offline':
      case 'disconnected':
        return {
          bg: 'bg-rose-50 border-rose-300 text-rose-800',
          dot: 'bg-rose-500',
          defaultText: status === 'offline' ? 'Offline' : 'Disconnected',
        };
      case 'info':
      default:
        return {
          bg: 'bg-[#014BAA]/10 border-[#014BAA]/25 text-[#014BAA] font-semibold',
          dot: 'bg-[#014BAA]',
          defaultText: 'Information',
        };
    }
  };

  const style = getBadgeStyle();
  const displayText = label || style.defaultText;

  return (
    <span
      className={`inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-xs font-semibold border tracking-wide transition-all shadow-2xs ${style.bg}`}
    >
      <span
        className={`w-1.5 h-1.5 rounded-full ${style.dot} ${pulse ? 'animate-status-pulse' : ''}`}
      />
      {displayText}
    </span>
  );
};
