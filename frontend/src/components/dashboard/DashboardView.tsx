import React from 'react';
import type { HealthResponse, ModelsStatusResponse, NavTabId } from '../../types';
import { NoticeBanner } from '../common/NoticeBanner';
import { MetricCards } from './MetricCards';
import { ModelStatusGrid } from './ModelStatusGrid';
import { SampleCharts } from './SampleCharts';
import { ActivityFeed } from './ActivityFeed';

interface DashboardViewProps {
  health: HealthResponse | null;
  modelsStatus: ModelsStatusResponse | null;
  onNavigateToWorkbench: (tab: NavTabId) => void;
}

export const DashboardView: React.FC<DashboardViewProps> = ({
  health,
  modelsStatus,
  onNavigateToWorkbench,
}) => {
  return (
    <div className="space-y-6">
      {/* Platform Stage 1 Safeguard Banner */}
      <NoticeBanner
        type="stage1"
        title="MediVision AI • Stage 1 Foundation Initialized"
        message="Software engineering foundation, FastAPI service layer, and responsive UI operational. Model training pipelines, weights, and inference pipelines are deliberately not initialized in Stage 1."
      />

      {/* Summary Metric Cards */}
      <MetricCards health={health} modelsStatus={modelsStatus} />

      {/* Live Model Status & Configuration Grid */}
      <ModelStatusGrid
        modelsStatus={modelsStatus}
        onNavigateToWorkbench={onNavigateToWorkbench}
      />

      {/* Two Column Section: Benchmark Charts & Activity Feed */}
      <div className="space-y-6">
        <div>
          <h3 className="text-base font-semibold text-white tracking-tight mb-3">
            Target Pre-Training Benchmarks & Roadmaps
          </h3>
          <SampleCharts />
        </div>

        <ActivityFeed />
      </div>
    </div>
  );
};
