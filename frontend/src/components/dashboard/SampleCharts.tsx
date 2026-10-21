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
  { name: 'Chest X-ray', targetSamples: 29684, currentIngested: 29684, color: '#014BAA' },
  { name: 'Brain MRI', targetSamples: 48000, currentIngested: 0, color: '#0E5CC7' },
  { name: 'Skin Lesion', targetSamples: 33000, currentIngested: 0, color: '#2574DF' },
  { name: 'YOLO Lesions', targetSamples: 25000, currentIngested: 0, color: '#4B8DF0' },
  { name: 'Clinical EHR', targetSamples: 60000, currentIngested: 0, color: '#7DAEF7' },
];

const TARGET_BENCHMARK_MILESTONES = [
  { epoch: 'Baseline', targetAUROC: 50.0, diceTarget: 0.0, loss: 2.8 },
  { epoch: 'Stage 2 Ingestion', targetAUROC: 72.5, diceTarget: 0.45, loss: 1.6 },
  { epoch: 'Stage 3 Tuning', targetAUROC: 86.0, diceTarget: 0.72, loss: 0.8 },
  { epoch: 'Stage 4 Ensemble', targetAUROC: 94.2, diceTarget: 0.89, loss: 0.25 },
];

export const SampleCharts: React.FC = () => {
  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
      {/* Chart 1: Projected Dataset Volume */}
      <div className="rounded-2xl border border-[#EAE3DD] bg-white/95 p-5 relative overflow-hidden backdrop-blur-sm shadow-sm">
        <div className="flex items-start justify-between gap-2 mb-3">
          <div>
            <div className="flex items-center gap-2">
              <Database className="h-4 w-4 text-[#014BAA]" />
              <h3 className="text-sm font-bold text-[#1A1512] tracking-tight">
                Dataset Capacity & Intake Targets
              </h3>
            </div>
            <p className="text-xs text-[#61554E] mt-0.5">
              RSNA Pneumonia verified (29,684 DICOM images ready)
            </p>
          </div>

          <span className="shrink-0 px-2.5 py-0.5 rounded-full text-[10px] font-bold bg-[#014BAA]/10 border border-[#014BAA]/25 text-[#014BAA]">
            Stage 2
          </span>
        </div>

        {/* Notice tag inside the chart card */}
        <div className="mb-4 p-2.5 rounded-xl bg-[#F8F3F0] border border-[#EAE3DD] text-[11px] text-[#61554E] flex items-center gap-2">
          <AlertCircle className="h-3.5 w-3.5 text-[#014BAA] shrink-0" />
          <span>RSNA dataset verified in <code className="font-mono text-[#014BAA]">datasets/rsna_pneumonia</code>.</span>
        </div>

        <div className="h-64 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={DATASET_DISTRIBUTION} margin={{ top: 10, right: 10, left: -20, bottom: 20 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#EFEFE6" vertical={false} />
              <XAxis
                dataKey="name"
                stroke="#7C6E66"
                fontSize={11}
                tickLine={false}
                axisLine={{ stroke: '#EAE3DD' }}
              />
              <YAxis
                stroke="#7C6E66"
                fontSize={11}
                tickLine={false}
                axisLine={{ stroke: '#EAE3DD' }}
                tickFormatter={(val) => `${val / 1000}k`}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#FFFFFF',
                  borderColor: '#EAE3DD',
                  borderRadius: '0.75rem',
                  fontSize: '12px',
                  color: '#1A1512',
                  boxShadow: '0 4px 12px rgba(0,0,0,0.06)',
                }}
                formatter={(value: any) => [`${Number(value).toLocaleString()} samples`, 'Volume']}
              />
              <Bar dataKey="targetSamples" radius={[6, 6, 0, 0]}>
                {DATASET_DISTRIBUTION.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.color} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>

        <p className="mt-2 text-[10px] text-[#7C6E66] text-center font-mono">
          * Architectural reference capacity across clinical cohorts
        </p>
      </div>

      {/* Chart 2: Target Performance Roadmap */}
      <div className="rounded-2xl border border-[#EAE3DD] bg-white/95 p-5 relative overflow-hidden backdrop-blur-sm shadow-sm">
        <div className="flex items-start justify-between gap-2 mb-3">
          <div>
            <div className="flex items-center gap-2">
              <BarChart3 className="h-4 w-4 text-[#014BAA]" />
              <h3 className="text-sm font-bold text-[#1A1512] tracking-tight">
                Model Evaluation AUROC Trajectory
              </h3>
            </div>
            <p className="text-xs text-[#61554E] mt-0.5">
              Target validation AUROC progression across development phases
            </p>
          </div>

          <span className="shrink-0 px-2.5 py-0.5 rounded-full text-[10px] font-bold bg-[#014BAA]/10 border border-[#014BAA]/25 text-[#014BAA]">
            Benchmark
          </span>
        </div>

        {/* Notice tag */}
        <div className="mb-4 p-2.5 rounded-xl bg-[#F8F3F0] border border-[#EAE3DD] text-[11px] text-[#61554E] flex items-center gap-2">
          <AlertCircle className="h-3.5 w-3.5 text-[#014BAA] shrink-0" />
          <span>Clinical integrity rule: Actual measured metrics will populate upon training completion.</span>
        </div>

        <div className="h-64 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={TARGET_BENCHMARK_MILESTONES} margin={{ top: 10, right: 10, left: -20, bottom: 20 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#EFEFE6" vertical={false} />
              <XAxis
                dataKey="epoch"
                stroke="#7C6E66"
                fontSize={11}
                tickLine={false}
                axisLine={{ stroke: '#EAE3DD' }}
              />
              <YAxis
                stroke="#7C6E66"
                fontSize={11}
                tickLine={false}
                axisLine={{ stroke: '#EAE3DD' }}
                domain={[40, 100]}
                tickFormatter={(val) => `${val}%`}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#FFFFFF',
                  borderColor: '#EAE3DD',
                  borderRadius: '0.75rem',
                  fontSize: '12px',
                  color: '#1A1512',
                  boxShadow: '0 4px 12px rgba(0,0,0,0.06)',
                }}
                formatter={(value: any) => [`${value}% AUROC`, 'Target Metric']}
              />
              <Line
                type="monotone"
                dataKey="targetAUROC"
                stroke="#014BAA"
                strokeWidth={3}
                dot={{ fill: '#014BAA', r: 4 }}
                activeDot={{ r: 6, fill: '#01357A' }}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>

        <p className="mt-2 text-[10px] text-[#7C6E66] text-center font-mono">
          * Target milestones only — zero fabricated values displayed
        </p>
      </div>
    </div>
  );
};
