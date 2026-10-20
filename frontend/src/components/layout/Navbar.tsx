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
    <header className="h-16 bg-[#090F24]/90 backdrop-blur border-b border-slate-800 px-6 flex items-center justify-between sticky top-0 z-30">
      {/* Left: View Breadcrumb & Stage Badge */}
      <div className="flex items-center gap-3">
        <h2 className="text-base font-semibold text-white tracking-tight">
          {currentTabName}
        </h2>
        <span className="rounded-md bg-teal-500/10 border border-teal-500/30 px-2 py-0.5 text-[11px] font-semibold text-teal-300">
          Stage 1: Foundation
        </span>
      </div>

      {/* Right: Operational Status Indicators & Controls */}
      <div className="flex items-center gap-3">
        {/* Academic Project Badge */}
        <div className="hidden lg:flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-slate-900 border border-slate-800 text-[11px] text-slate-300">
          <ShieldCheck className="h-3.5 w-3.5 text-teal-400" />
          <span>Major Research Project</span>
        </div>

        {/* Database Status Indicator */}
        <div
          title={health?.database?.details || 'Database connection status'}
          className="flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-slate-900 border border-slate-800 text-[11px]"
        >
          <Database className={`h-3.5 w-3.5 ${dbConnected ? 'text-teal-400' : 'text-amber-400'}`} />
          <span className="text-slate-300 hidden sm:inline">MongoDB:</span>
          <span className={dbConnected ? 'text-teal-300 font-medium' : 'text-amber-300 font-medium'}>
            {dbConnected ? 'Atlas Connected' : 'Disconnected (Unconfigured)'}
          </span>
        </div>

        {/* Backend Server Status */}
        <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-slate-900 border border-slate-800 text-[11px]">
          <Server className={`h-3.5 w-3.5 ${isOnline ? 'text-emerald-400' : 'text-rose-400'}`} />
          <span className="text-slate-300 hidden sm:inline">FastAPI:</span>
          <span className={isOnline ? 'text-emerald-300 font-medium' : 'text-rose-400 font-medium'}>
            {isOnline ? 'Online (v1.0.0)' : 'Disconnected'}
          </span>
        </div>

        {/* Refresh Button */}
        <button
          onClick={onRefresh}
          disabled={loading}
          title="Refresh Backend Health & Status"
          className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-teal-600/20 hover:bg-teal-600/30 text-teal-300 hover:text-teal-200 border border-teal-500/30 text-xs font-medium transition-all active:scale-95 disabled:opacity-50"
        >
          <RefreshCw className={`h-3.5 w-3.5 ${loading ? 'animate-spin' : ''}`} />
          <span className="hidden md:inline">Sync Status</span>
        </button>
      </div>
    </header>
  );
};
