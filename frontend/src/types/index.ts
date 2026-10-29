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

export interface ChestXRayClassifyResponse {
  status: 'success' | 'not_trained' | 'error';
  model_name: string;
  prediction: string | null;
  predicted_class_id: number | null;
  confidence: number | null;
  threshold: number;
  id?: string | null;
  filename: string;
  file_type: string;
  message: string;
  disclaimer: string;
  image_preview?: string | null;
  gradcam_preview?: string | null;
  heatmap_preview?: string | null;
}

export interface PredictionHistoryItem {
  id: string;
  timestamp: string;
  modality: string;
  model_name: string;
  filename: string;
  file_type: string;
  prediction: string | null;
  predicted_class_id: number | null;
  confidence: number | null;
  threshold: number;
  status: string;
  disclaimer?: string;
  image_preview?: string | null;
  gradcam_preview?: string | null;
  heatmap_preview?: string | null;
  risk_tier?: string | null;
  risk_probability?: number | null;
  top_contributing_factors?: string[] | null;
  patient_data?: Record<string, any> | null;
  volumetric_metrics?: Record<string, any> | null;
  slices?: Record<string, any> | null;
  detections?: YOLOBoundingBox[] | null;
  num_detections?: number | null;
}

export interface PredictionHistoryResponse {
  total: number;
  records: PredictionHistoryItem[];
  database_connected: boolean;
}

export interface ActiveAnalysisCase {
  recordId?: string | null;
  filename: string;
  fileType: string;
  isDicom: boolean;
  imagePreview: string | null;
  gradcamPreview?: string | null;
  heatmapPreview?: string | null;
  prediction: string | null;
  predictedClassId?: number | null;
  confidence: number | null;
  threshold: number;
  modelName: string;
  status: string;
  message?: string;
  disclaimer?: string;
  timestamp: string;
  viewMode?: 'overlay' | 'heatmap' | 'original' | 'split' | 'gradcam';
  explanation?: string;
  modality?: string;
}

export type ActiveChestXRayCase = ActiveAnalysisCase;

export interface ActiveClinicalCase {
  formData: ClinicalPatientInput;
  result: ClinicalRiskResponse | null;
  activePreset: string;
}

export interface ClinicalPatientInput {
  age: number;
  sex: number;
  cp: number;
  trestbps: number;
  chol: number;
  fbs: number;
  restecg: number;
  thalach: number;
  exang: number;
  oldpeak: number;
  slope: number;
  ca: number;
  thal: number;
  patient_id?: string;
}

export interface ClinicalRiskResponse {
  id?: string | null;
  status: 'success' | 'not_trained' | 'error';
  risk_probability: number | null;
  predicted_class_id: number | null;
  risk_tier: 'Low Risk' | 'Moderate Risk' | 'Elevated Risk' | 'Unconfigured' | 'Error';
  prediction: string;
  model_name: string;
  dataset_source: string;
  top_contributing_factors: string[];
  metrics?: Record<string, any>;
  message: string;
  disclaimer: string;
}

export interface ClinicalModelInfoResponse {
  model_name: string;
  status: string;
  dataset_name: string;
  total_training_samples: number;
  test_samples: number;
  metrics: {
    accuracy?: number;
    balanced_accuracy?: number;
    roc_auc?: number;
    f1?: number;
    sensitivity_recall?: number;
    specificity?: number;
    precision?: number;
  };
  confusion_matrix: {
    true_negatives?: number;
    false_positives?: number;
    false_negatives?: number;
    true_positives?: number;
  };
  top_features: Array<{ feature: string; importance: number }>;
  disclaimer: string;
}

export interface GradCAMResponse {
  id?: string | null;
  status: 'success' | 'not_trained' | 'error';
  attribution_method: string;
  target_layer: string;
  prediction?: string | null;
  confidence?: number | null;
  threshold: number;
  filename: string;
  original_preview?: string | null;
  overlay_preview?: string | null;
  heatmap_preview?: string | null;
  explanation?: string | null;
  disclaimer: string;
  message?: string;
}

export interface BrainMRIStatusResponse {
  status: 'ready' | 'not_trained' | 'dataset_missing';
  stage_phase?: 'architecture_ready' | 'dataset_ready' | 'training_in_progress' | 'model_trained' | 'experimental_prototype';
  is_pilot?: boolean;
  is_experimental?: boolean;
  model_name: string;
  architecture: string;
  checkpoint_path: string | null;
  pilot_checkpoint_path?: string | null;
  experimental_checkpoint_path?: string | null;
  device: string;
  dataset_status: {
    status: string;
    message: string;
    patient_count: number;
    complete_patient_count: number;
    data_dir: string;
    setup_guide?: string;
  };
  metrics: Record<string, any>;
  pilot_metrics?: Record<string, any>;
  custom_metrics?: Record<string, any>;
  disclaimer: string;
}

export interface BrainMRIValidationResponse {
  status: 'valid' | 'invalid' | 'incomplete';
  filename: string;
  detected_modalities: string[];
  missing_modalities: string[];
  dimensions?: number[];
  voxel_spacing_mm?: number[];
  message: string;
}

export interface BrainMRISegmentResponse {
  id?: string | null;
  status: 'success' | 'not_trained' | 'error';
  is_experimental?: boolean;
  model_name: string;
  filename: string;
  patient_id?: string | null;
  volumetric_metrics: {
    whole_tumor_cm3?: number;
    tumor_core_cm3?: number;
    enhancing_tumor_cm3?: number;
    edema_cm3?: number;
    necrotic_core_cm3?: number;
    total_lesion_voxels?: number;
    voxel_spacing_mm?: number[];
  };
  dominant_finding: string;
  slices: {
    plane?: 'axial' | 'coronal' | 'sagittal';
    slice_idx?: number;
    total_slices?: number;
    mri_preview?: string | null;
    mask_preview?: string | null;
    overlay_preview?: string | null;
  };
  message: string;
  disclaimer: string;
}

export interface BrainMRISliceResponse {
  plane: 'axial' | 'coronal' | 'sagittal';
  slice_idx: number;
  total_slices: number;
  mri_preview: string;
  mask_preview?: string | null;
  overlay_preview?: string | null;
}

export interface ActiveBrainMRICase {
  filename: string;
  recordId?: string | null;
  patientId?: string | null;
  result: BrainMRISegmentResponse | null;
  selectedPlane: 'axial' | 'coronal' | 'sagittal';
  selectedSlice: number;
  viewMode: 'overlay' | 'original' | 'mask';
}

export interface YOLOBoundingBox {
  id: number;
  class_name: string;
  class_id: number;
  confidence: number;
  bbox_pixels: [number, number, number, number];
  bbox_normalized: [number, number, number, number];
  area_pixels: number;
}

export interface YOLODetectionStatusResponse {
  model_architecture: string;
  trained: boolean;
  status: 'ready' | 'not_trained' | 'unconfigured';
  checkpoint_path: string | null;
  classes: string[];
  metrics: Record<string, any> | null;
}

export interface YOLODetectionResponse {
  id?: string | null;
  status: 'success' | 'not_trained' | 'error';
  message: string;
  filename?: string | null;
  detections: YOLOBoundingBox[];
  num_detections: number;
  image_dimensions?: { width: number; height: number } | null;
  conf_threshold: number;
  iou_threshold: number;
  preview_image?: string | null;
  model_status: YOLODetectionStatusResponse;
  persisted?: boolean;
  analysis_id?: string | null;
  highest_candidate_confidence?: number | null;
  disclaimer: string;
}

export interface ActiveYOLOCase {
  recordId?: string | null;
  filename: string;
  imagePreview: string | null;
  result: YOLODetectionResponse | null;
  confThreshold: number;
  iouThreshold: number;
  timestamp?: string;
}

