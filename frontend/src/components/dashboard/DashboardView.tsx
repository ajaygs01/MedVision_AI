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
      {/* Platform Stage Notice Banner */}
      <NoticeBanner
        type="stage1"
        title={
          modelsStatus?.models?.cnn_chest_xray?.trained
            ? 'MediVision AI • Stage 2 CNN Inference Pipeline Active'
            : 'MediVision AI • Medical AI Research Platform'
        }
        message={
          modelsStatus?.models?.cnn_chest_xray?.trained
            ? 'ResNet-50 Chest X-ray classification engine is loaded with verified weights and ready for radiograph inference. Non-imaging and secondary pipelines remain in development.'
            : 'Software engineering foundation and FastAPI service layer operational. Awaiting model checkpoint loading.'
        }
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
          <h3 className="text-base font-bold text-[#1A1512] tracking-tight mb-3">
            Target Pre-Training Benchmarks & Roadmaps
          </h3>
          <SampleCharts />
        </div>

        <ActivityFeed />
      </div>
    </div>
  );
};
