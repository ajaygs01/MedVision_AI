import React from 'react';
import { RefreshCw, Server, ShieldCheck, Database } from 'lucide-react';
import type { HealthResponse } from '../../types';

interface NavbarProps {
  currentTabName: string;
  health: HealthResponse | null;
  loading: boolean;
  onRefresh: () => void;
}

export const Navbar: React.FC<NavbarProps> = ({
  currentTabName,
  health,
  loading,
  onRefresh,
}) => {
  const isOnline = health?.status === 'online';
  const dbConnected = health?.database?.status === 'connected';

  return (
    <header className="h-16 bg-[#F8F3F0]/90 backdrop-blur-md border-b border-[#EAE3DD] px-6 flex items-center justify-between sticky top-0 z-30 shadow-xs">
      {/* Left: View Breadcrumb & Stage Badge */}
      <div className="flex items-center gap-3">
        <h2 className="text-base font-bold text-[#1A1512] tracking-tight">
          {currentTabName}
        </h2>
        <span className="rounded-full bg-[#014BAA]/10 border border-[#014BAA]/25 px-2.5 py-0.5 text-[11px] font-bold text-[#014BAA]">
          Stage 2: CNN Pipeline
        </span>
      </div>

      {/* Right: Operational Status Indicators & Controls */}
      <div className="flex items-center gap-2.5">
        {/* Academic Project Badge */}
        <div className="hidden lg:flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-white/80 border border-[#EAE3DD] text-[11px] text-[#61554E] font-medium shadow-xs">
          <ShieldCheck className="h-3.5 w-3.5 text-[#014BAA]" />
          <span>Major Research Platform</span>
        </div>

        {/* Database Status Indicator */}
        <div
          title={health?.database?.details || 'Database connection status'}
          className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg bg-white/80 border border-[#EAE3DD] text-[11px] shadow-xs"
        >
          <Database className={`h-3.5 w-3.5 ${dbConnected ? 'text-emerald-600' : 'text-amber-600'}`} />
          <span className="text-[#61554E] hidden sm:inline">MongoDB:</span>
          <span className={dbConnected ? 'text-emerald-700 font-semibold' : 'text-amber-700 font-semibold'}>
            {dbConnected ? 'Atlas Connected' : 'Unconfigured'}
          </span>
        </div>

        {/* Backend Server Status */}
        <div className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg bg-white/80 border border-[#EAE3DD] text-[11px] shadow-xs">
          <Server className={`h-3.5 w-3.5 ${isOnline ? 'text-emerald-600' : 'text-rose-600'}`} />
          <span className="text-[#61554E] hidden sm:inline">FastAPI:</span>
          <span className={isOnline ? 'text-emerald-700 font-semibold' : 'text-rose-700 font-semibold'}>
            {isOnline ? 'Online' : 'Disconnected'}
          </span>
        </div>

        {/* Refresh Button */}
        <button
          onClick={onRefresh}
          disabled={loading}
          title="Refresh Backend Health & Status"
          className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-[#014BAA] hover:bg-[#01357A] text-[#F8F3F0] text-xs font-semibold transition-all active:scale-95 shadow-sm hover:shadow disabled:opacity-50"
        >
          <RefreshCw className={`h-3.5 w-3.5 ${loading ? 'animate-spin' : ''}`} />
          <span className="hidden md:inline">Sync</span>
        </button>
      </div>
    </header>
  );
};
