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
      <div className="rounded-2xl border border-[#EAE3DD] bg-white/95 p-5 relative overflow-hidden backdrop-blur-sm shadow-sm hover:shadow-md hover:border-[#014BAA]/40 transition-all group">
        <div className="flex items-center justify-between">
          <span className="text-xs font-semibold text-[#61554E]">Planned AI Pipelines</span>
          <div className="p-2 rounded-xl bg-[#014BAA]/10 text-[#014BAA] border border-[#014BAA]/20">
            <Layers className="h-4 w-4" />
          </div>
        </div>
        <div className="mt-3">
          <div className="text-2xl font-black text-[#1A1512] tracking-tight">{totalModels} Pipelines</div>
          <p className="mt-1 text-[11px] text-[#014BAA] font-medium flex items-center gap-1">
            CNN, YOLO, U-Net, Tabular, XAI, RAG
          </p>
        </div>
        <div className="mt-3.5 pt-2.5 border-t border-[#F0EAE5] flex items-center justify-between text-[11px] text-[#61554E]">
          <span>Pipeline Status</span>
          <span className="font-bold text-[#014BAA]">Stage 2 Ready</span>
        </div>
      </div>

      {/* Metric 2: Active Trained Models */}
      <div className="rounded-2xl border border-[#EAE3DD] bg-white/95 p-5 relative overflow-hidden backdrop-blur-sm shadow-sm hover:shadow-md hover:border-[#014BAA]/40 transition-all group">
        <div className="flex items-center justify-between">
          <span className="text-xs font-semibold text-[#61554E]">Trained Models</span>
          <div className={`p-2 rounded-xl border ${trainedCount > 0 ? 'bg-emerald-500/10 text-emerald-700 border-emerald-500/20' : 'bg-amber-500/10 text-amber-700 border-amber-500/20'}`}>
            <Cpu className="h-4 w-4" />
          </div>
        </div>
        <div className="mt-3">
          <div className="text-2xl font-black text-[#1A1512] tracking-tight">{trainedCount} Active</div>
          {trainedCount > 0 ? (
            <p className="mt-1 text-[11px] text-emerald-700 font-medium flex items-center gap-1">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-600 inline-block" /> ResNet-50 Checkpoint Active
            </p>
          ) : (
            <p className="mt-1 text-[11px] text-amber-700 font-medium flex items-center gap-1">
              <AlertCircle className="h-3 w-3 inline" /> Checkpoint Pending
            </p>
          )}
        </div>
        <div className="mt-3.5 pt-2.5 border-t border-[#F0EAE5] flex items-center justify-between text-[11px] text-[#61554E]">
          <span>Dataset Status</span>
          <span className="font-bold text-emerald-700">RSNA Verified</span>
        </div>
      </div>

      {/* Metric 3: FastAPI Backend */}
      <div className="rounded-2xl border border-[#EAE3DD] bg-white/95 p-5 relative overflow-hidden backdrop-blur-sm shadow-sm hover:shadow-md hover:border-[#014BAA]/40 transition-all group">
        <div className="flex items-center justify-between">
          <span className="text-xs font-semibold text-[#61554E]">FastAPI Service</span>
          <div className={`p-2 rounded-xl border ${isOnline ? 'bg-emerald-500/10 text-emerald-700 border-emerald-500/20' : 'bg-rose-500/10 text-rose-700 border-rose-500/20'}`}>
            <Server className="h-4 w-4" />
          </div>
        </div>
        <div className="mt-3">
          <div className="text-2xl font-black text-[#1A1512] tracking-tight">
            {isOnline ? 'Healthy' : 'Connecting...'}
          </div>
          <p className="mt-1 text-[11px] text-[#61554E]">
            {health?.version ? `Build ${health.version}` : 'Uvicorn Port 8000'}
          </p>
        </div>
        <div className="mt-3.5 pt-2.5 border-t border-[#F0EAE5] flex items-center justify-between text-[11px] text-[#61554E]">
          <span>Health Telemetry</span>
          <span className={isOnline ? 'text-emerald-700 font-bold' : 'text-rose-600'}>
            {isOnline ? 'Online (v1.0.0)' : 'Offline'}
          </span>
        </div>
      </div>

      {/* Metric 4: MongoDB Atlas */}
      <div className="rounded-2xl border border-[#EAE3DD] bg-white/95 p-5 relative overflow-hidden backdrop-blur-sm shadow-sm hover:shadow-md hover:border-[#014BAA]/40 transition-all group">
        <div className="flex items-center justify-between">
          <span className="text-xs font-semibold text-[#61554E]">Database Layer</span>
          <div className={`p-2 rounded-xl border ${dbConnected ? 'bg-emerald-500/10 text-emerald-700 border-emerald-500/20' : 'bg-[#014BAA]/10 text-[#014BAA] border-[#014BAA]/20'}`}>
            <Database className="h-4 w-4" />
          </div>
        </div>
        <div className="mt-3">
          <div className="text-2xl font-black text-[#1A1512] tracking-tight">
            {dbConnected ? 'Atlas Cluster' : 'Config Pending'}
          </div>
          <p className="mt-1 text-[11px] text-[#61554E] truncate" title={health?.database?.details}>
            {dbConnected ? 'Cloud replica connected' : 'MONGODB_URI pending in .env'}
          </p>
        </div>
        <div className="mt-3.5 pt-2.5 border-t border-[#F0EAE5] flex items-center justify-between text-[11px] text-[#61554E]">
          <span>Storage Engine</span>
          <span className="text-[#014BAA] font-bold font-mono text-[10px]">MongoDB Atlas</span>
        </div>
      </div>
    </div>
  );
};
