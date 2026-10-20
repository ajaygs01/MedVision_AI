export interface DatabaseStatus {
  status: 'connected' | 'disconnected';
  engine: string;
  details: string;
}

export interface HealthResponse {
  status: string;
  version: string;
  timestamp: string;
  database: DatabaseStatus;
  environment: string;
}

export interface ModelItemStatus {
  name: string;
  modality: string;
  architecture: string;
  framework: string;
  status: 'unconfigured' | 'training' | 'ready';
  trained: boolean;
  weights_path: string | null;
  next_stage: string;
  description: string;
}

export interface ModelsStatusResponse {
  stage: number;
  system_status: string;
  total_models: number;
  active_trained_count: number;
  models: Record<string, ModelItemStatus>;
  disclaimer: string;
}

export type NavTabId =
  | 'dashboard'
  | 'chest-xray'
  | 'brain-mri'
  | 'skin-lesion'
  | 'yolo-detection'
  | 'clinical-ml'
  | 'explainable-ai'
  | 'research-assistant'
  | 'model-evaluation'
  | 'analysis-history'
  | 'settings';

export interface NavItemConfig {
  id: NavTabId;
  label: string;
  iconName: string;
  category: 'core' | 'imaging' | 'ml_and_xai' | 'research' | 'management';
  badge?: string;
  stageNote?: string;
}

export interface ModuleTechnicalDetails {
  id: NavTabId;
  title: string;
  subtitle: string;
  modality: string;
  modelKey?: string;
  plannedBackbone: string;
  framework: string;
  targetDataset: string;
  inputSpecification: string;
  outputSpecification: string;
  pipelineSteps: string[];
  plannedEvaluationMetrics: string[];
}
