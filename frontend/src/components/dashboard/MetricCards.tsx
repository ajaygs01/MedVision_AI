import React from 'react';
import { Layers, Cpu, Server, Database, AlertCircle } from 'lucide-react';
import type { HealthResponse, ModelsStatusResponse } from '../../types';

interface MetricCardsProps {
  health: HealthResponse | null;
  modelsStatus: ModelsStatusResponse | null;
}

export const MetricCards: React.FC<MetricCardsProps> = ({ health, modelsStatus }) => {
  const isOnline = health?.status === 'online';
  const dbConnected = health?.database?.status === 'connected';
  const totalModels = modelsStatus?.total_models ?? 7;
  const trainedCount = modelsStatus?.active_trained_count ?? 0;

  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
      {/* Metric 1: Architectures Planned */}
      <div className="rounded-xl border border-slate-800 bg-[#0E1738]/60 p-4 relative overflow-hidden backdrop-blur-sm group hover:border-slate-700 transition-all">
        <div className="flex items-center justify-between">
          <span className="text-xs font-medium text-slate-400">Planned AI Pipelines</span>
          <div className="p-2 rounded-lg bg-teal-500/10 text-teal-400 border border-teal-500/20">
            <Layers className="h-4 w-4" />
          </div>
        </div>
        <div className="mt-3">
          <div className="text-2xl font-bold text-white tracking-tight">{totalModels} Pipelines</div>
          <p className="mt-1 text-[11px] text-teal-300 flex items-center gap-1">
            CNN, YOLO, U-Net, Tabular, XAI, RAG
          </p>
        </div>
        <div className="mt-3 pt-2.5 border-t border-slate-800/80 flex items-center justify-between text-[11px] text-slate-400">
          <span>Stage 1 Status</span>
          <span className="font-semibold text-teal-300">Architecture Ready</span>
        </div>
      </div>

      {/* Metric 2: Active Trained Models */}
      <div className="rounded-xl border border-slate-800 bg-[#0E1738]/60 p-4 relative overflow-hidden backdrop-blur-sm group hover:border-slate-700 transition-all">
        <div className="flex items-center justify-between">
          <span className="text-xs font-medium text-slate-400">Trained Models</span>
          <div className="p-2 rounded-lg bg-amber-500/10 text-amber-400 border border-amber-500/20">
            <Cpu className="h-4 w-4" />
          </div>
        </div>
        <div className="mt-3">
          <div className="text-2xl font-bold text-amber-300 tracking-tight">{trainedCount} Active</div>
          <p className="mt-1 text-[11px] text-amber-400/90 flex items-center gap-1">
            <AlertCircle className="h-3 w-3 inline" /> Strict Rule: Training Pending
          </p>
        </div>
        <div className="mt-3 pt-2.5 border-t border-slate-800/80 flex items-center justify-between text-[11px] text-slate-400">
          <span>Next Milestone</span>
          <span className="font-semibold text-amber-300">Stage 2 Ingestion</span>
        </div>
      </div>

      {/* Metric 3: FastAPI Backend */}
      <div className="rounded-xl border border-slate-800 bg-[#0E1738]/60 p-4 relative overflow-hidden backdrop-blur-sm group hover:border-slate-700 transition-all">
        <div className="flex items-center justify-between">
          <span className="text-xs font-medium text-slate-400">FastAPI Service</span>
          <div className={`p-2 rounded-lg border ${isOnline ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20' : 'bg-rose-500/10 text-rose-400 border-rose-500/20'}`}>
            <Server className="h-4 w-4" />
          </div>
        </div>
        <div className="mt-3">
          <div className="text-2xl font-bold text-white tracking-tight">
            {isOnline ? 'Healthy' : 'Connecting...'}
          </div>
          <p className="mt-1 text-[11px] text-slate-400">
            {health?.version ? `Build ${health.version}` : 'FastAPI Uvicorn Port 8000'}
          </p>
        </div>
        <div className="mt-3 pt-2.5 border-t border-slate-800/80 flex items-center justify-between text-[11px] text-slate-400">
          <span>Health Check</span>
          <span className={isOnline ? 'text-emerald-300 font-semibold' : 'text-rose-400'}>
            {isOnline ? 'GET /api/health OK' : 'Checking'}
          </span>
        </div>
      </div>

      {/* Metric 4: MongoDB Atlas */}
      <div className="rounded-xl border border-slate-800 bg-[#0E1738]/60 p-4 relative overflow-hidden backdrop-blur-sm group hover:border-slate-700 transition-all">
        <div className="flex items-center justify-between">
          <span className="text-xs font-medium text-slate-400">Database Layer</span>
          <div className={`p-2 rounded-lg border ${dbConnected ? 'bg-teal-500/10 text-teal-400 border-teal-500/20' : 'bg-cyan-500/10 text-cyan-400 border-cyan-500/20'}`}>
            <Database className="h-4 w-4" />
          </div>
        </div>
        <div className="mt-3">
          <div className="text-2xl font-bold text-white tracking-tight">
            {dbConnected ? 'Atlas Cluster' : 'Config Pending'}
          </div>
          <p className="mt-1 text-[11px] text-slate-400 truncate" title={health?.database?.details}>
            {dbConnected ? 'Cloud replica set connected' : 'MONGODB_URI pending in .env'}
          </p>
        </div>
        <div className="mt-3 pt-2.5 border-t border-slate-800/80 flex items-center justify-between text-[11px] text-slate-400">
          <span>Storage Engine</span>
          <span className="text-cyan-300 font-medium font-mono text-[10px]">MongoDB Atlas</span>
        </div>
      </div>
    </div>
  );
};
