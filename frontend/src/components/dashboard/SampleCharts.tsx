import React from 'react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
  Cell,
  LineChart,
  Line,
} from 'recharts';
import { AlertCircle, BarChart3, Database } from 'lucide-react';

const DATASET_DISTRIBUTION = [
  { name: 'Chest X-ray', targetSamples: 112000, currentIngested: 0, color: '#0D9488' },
  { name: 'Brain MRI', targetSamples: 48000, currentIngested: 0, color: '#14B8A6' },
  { name: 'Skin Lesion', targetSamples: 33000, currentIngested: 0, color: '#2DD4BF' },
  { name: 'YOLO Lesions', targetSamples: 25000, currentIngested: 0, color: '#38BDF8' },
  { name: 'Clinical EHR', targetSamples: 60000, currentIngested: 0, color: '#818CF8' },
];

const TARGET_BENCHMARK_MILESTONES = [
  { epoch: 'Baseline (Untrained)', targetAUROC: 50.0, diceTarget: 0.0, loss: 2.8 },
  { epoch: 'Stage 2 Ingestion', targetAUROC: 72.5, diceTarget: 0.45, loss: 1.6 },
  { epoch: 'Stage 3 Tuning', targetAUROC: 86.0, diceTarget: 0.72, loss: 0.8 },
  { epoch: 'Stage 4 Ensemble Target', targetAUROC: 94.2, diceTarget: 0.89, loss: 0.25 },
];

export const SampleCharts: React.FC = () => {
  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
      {/* Chart 1: Projected Dataset Volume */}
      <div className="rounded-xl border border-slate-800 bg-[#0C1430]/70 p-5 relative overflow-hidden backdrop-blur-sm">
        <div className="flex items-start justify-between gap-2 mb-3">
          <div>
            <div className="flex items-center gap-2">
              <Database className="h-4 w-4 text-teal-400" />
              <h3 className="text-sm font-semibold text-white tracking-tight">
                Target Dataset Distribution (Planned)
              </h3>
            </div>
            <p className="text-xs text-slate-400 mt-0.5">
              Target sample counts across NIH, BraTS, ISIC, and MIMIC cohorts
            </p>
          </div>

          <span className="shrink-0 px-2.5 py-0.5 rounded-full text-[10px] font-semibold bg-amber-950/70 border border-amber-500/40 text-amber-300">
            Sample Data
          </span>
        </div>

        {/* Notice tag inside the chart card */}
        <div className="mb-4 p-2 rounded-lg bg-slate-900/80 border border-slate-800 text-[11px] text-slate-300 flex items-center gap-2">
          <AlertCircle className="h-3.5 w-3.5 text-amber-400 shrink-0" />
          <span>Stage 1 Status: Datasets pending download & preprocessing in Stage 2.</span>
        </div>

        <div className="h-64 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={DATASET_DISTRIBUTION} margin={{ top: 10, right: 10, left: -20, bottom: 20 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1E293B" vertical={false} />
              <XAxis
                dataKey="name"
                stroke="#64748B"
                fontSize={11}
                tickLine={false}
                axisLine={{ stroke: '#334155' }}
              />
              <YAxis
                stroke="#64748B"
                fontSize={11}
                tickLine={false}
                axisLine={{ stroke: '#334155' }}
                tickFormatter={(val) => `${val / 1000}k`}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#070D1F',
                  borderColor: '#334155',
                  borderRadius: '0.5rem',
                  fontSize: '12px',
                  color: '#F8FAFC',
                }}
                formatter={(value: any) => [`${Number(value).toLocaleString()} samples`, 'Planned Volume']}
              />
              <Bar dataKey="targetSamples" radius={[4, 4, 0, 0]}>
                {DATASET_DISTRIBUTION.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.color} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>

        <p className="mt-2 text-[10px] text-slate-400 text-center font-mono">
          * Architectural reference capacity — 0 images ingested in Stage 1
        </p>
      </div>

      {/* Chart 2: Milestone Convergence Plan */}
      <div className="rounded-xl border border-slate-800 bg-[#0C1430]/70 p-5 relative overflow-hidden backdrop-blur-sm">
        <div className="flex items-start justify-between gap-2 mb-3">
          <div>
            <div className="flex items-center gap-2">
              <BarChart3 className="h-4 w-4 text-teal-400" />
              <h3 className="text-sm font-semibold text-white tracking-tight">
                Architectural Validation Targets
              </h3>
            </div>
            <p className="text-xs text-slate-400 mt-0.5">
              Projected convergence trajectory across research phases
            </p>
          </div>

          <span className="shrink-0 px-2.5 py-0.5 rounded-full text-[10px] font-semibold bg-cyan-950/70 border border-cyan-500/40 text-cyan-300">
            Roadmap Target
          </span>
        </div>

        {/* Notice tag inside the chart card */}
        <div className="mb-4 p-2 rounded-lg bg-slate-900/80 border border-slate-800 text-[11px] text-slate-300 flex items-center gap-2">
          <AlertCircle className="h-3.5 w-3.5 text-cyan-400 shrink-0" />
          <span>Stage 1 Status: Training loops and metric calculations will be wired in Stage 2.</span>
        </div>

        <div className="h-64 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart
              data={TARGET_BENCHMARK_MILESTONES}
              margin={{ top: 10, right: 10, left: -20, bottom: 20 }}
            >
              <CartesianGrid strokeDasharray="3 3" stroke="#1E293B" vertical={false} />
              <XAxis
                dataKey="epoch"
                stroke="#64748B"
                fontSize={10}
                tickLine={false}
                axisLine={{ stroke: '#334155' }}
              />
              <YAxis
                stroke="#64748B"
                fontSize={11}
                tickLine={false}
                axisLine={{ stroke: '#334155' }}
                domain={[0, 100]}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#070D1F',
                  borderColor: '#334155',
                  borderRadius: '0.5rem',
                  fontSize: '12px',
                  color: '#F8FAFC',
                }}
              />
              <Line
                type="monotone"
                dataKey="targetAUROC"
                name="Projected AUROC %"
                stroke="#14B8A6"
                strokeWidth={2}
                dot={{ r: 4, fill: '#14B8A6' }}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>

        <p className="mt-2 text-[10px] text-slate-400 text-center font-mono">
          * Target milestones only — No models have been trained or evaluated yet
        </p>
      </div>
    </div>
  );
};
