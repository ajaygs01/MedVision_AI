import React, { useState, useEffect, useCallback } from 'react';
import {
  FileHeart,
  Brain,
  Microscope,
  Crosshair,
  TableProperties,
  Eye,
  Bot,
  BarChart3,
  History,
  Settings as SettingsIcon,
  AlertTriangle,
  Cpu,
  Layers,
  FileCode,
  CheckCircle,
  Database,
  RefreshCw,
  Download,
  AlertCircle,
} from 'lucide-react';
import type { NavTabId, PredictionHistoryItem } from '../../types';
import { fetchPredictionHistory, downloadAnalysisReport } from '../../services/api';
import { StatusBadge } from '../common/StatusBadge';
import { useAnalysis } from '../../context/useAnalysis';
import { ClinicalMLWorkbench } from './ClinicalMLWorkbench';
import { ExplainableAIWorkbench } from './ExplainableAIWorkbench';
import { BrainMRIWorkbench } from './BrainMRIWorkbench';
import { YOLODetectionWorkbench } from './YOLODetectionWorkbench';

interface ModuleWorkbenchProps {
  tabId: NavTabId;
  onBackToDashboard: () => void;
  onSelectTab?: (tab: NavTabId) => void;
}

interface BlueprintConfig {
  title: string;
  subtitle: string;
  stageBadge: string;
  icon: React.ComponentType<{ className?: string }>;
  modality: string;
  dataset: string;
  backbone: string;
  framework: string;
  inputFormat: string;
  lossFunction: string;
  targetMetrics: string[];
  pipelineSteps: { title: string; desc: string }[];
}

const BLUEPRINTS: Record<string, BlueprintConfig> = {
  'chest-xray': {
    title: 'Chest X-ray Multi-Abnormality Analysis',
    subtitle: 'Deep Convolutional Neural Network for Cardiothoracic Pathology Detection',
    stageBadge: 'NOT TRAINED • Stage 2 Pending',
    icon: FileHeart,
    modality: 'CXR (Posteroanterior & Anteroposterior views)',
    dataset: 'NIH ChestX-ray14 & Stanford CheXpert (112,120 annotated radiographs)',
    backbone: 'DenseNet-121 / ResNet-50 with Pretrained ImageNet Weights',
    framework: 'PyTorch 2.x & Torchvision',
    inputFormat: 'DICOM (.dcm), High-Res PNG, JPEG (1024x1024 -> 224x224 normalized)',
    lossFunction: 'Weighted Binary Cross-Entropy with Asymmetric Loss (Multi-label)',
    targetMetrics: ['Mean AUROC >= 0.88', 'F1-Score across 14 classes', 'Precision-Recall AUC'],
    pipelineSteps: [
      { title: '1. Ingestion & Quality Filter', desc: 'DICOM pixel array extraction, lung field alignment, aspect ratio preservation.' },
      { title: '2. Normalization & Augmentations', desc: 'Histogram equalization (CLAHE), random rotation (-10° to 10°), horizontal flip.' },
      { title: '3. CNN Feature Extraction', desc: 'Dense connectivity layers capturing subtle opacity and consolidation patterns.' },
      { title: '4. Stage 4 XAI Coupling', desc: 'Target layer attribution for Grad-CAM activation heatmap generation.' },
    ],
  },
  'brain-mri': {
    title: 'Brain MRI Lesion & Tumor Segmentation',
    subtitle: 'Volumetric Attention 3D U-Net for Precise Anatomical Delineation',
    stageBadge: 'PIPELINE ACTIVE • 3D U-Net',
    icon: Brain,
    modality: 'Multi-parametric Magnetic Resonance Imaging (FLAIR, T1, T1ce, T2)',
    dataset: 'BraTS 2023 / 2024 Multi-Modal Brain Tumor Segmentation Cohort',
    backbone: 'Attention 3D U-Net with Squeeze-and-Excitation Skip Connections',
    framework: 'PyTorch / MONAI (Medical Open Network for AI)',
    inputFormat: 'NIfTI 3D Volumes (.nii / .nii.gz), 4 channels co-registered',
    lossFunction: 'Generalized Dice Loss + Focal Binary Cross-Entropy',
    targetMetrics: ['Dice Similarity Coefficient (DSC) >= 0.86', '95% Hausdorff Distance (HD95) < 6.0mm', 'Sensitivity >= 0.88'],
    pipelineSteps: [
      { title: '1. Multi-Modal Co-Registration', desc: 'Spatial reslicing to 1mm isotropic voxels and rigid skull stripping.' },
      { title: '2. Z-Score Intensity Normalization', desc: 'Channel-wise intensity scaling restricted to non-zero brain tissue mask.' },
      { title: '3. Attention Encoder-Decoder', desc: 'Multi-scale spatial feature learning with additive attention gates.' },
      { title: '4. Post-processing & Mesh Export', desc: 'Connected component filtering and volumetric 3D lesion calculation.' },
    ],
  },
  'skin-lesion': {
    title: 'Skin Lesion Dermatoscopy Classifier',
    subtitle: 'High-Resolution Visual Classifier for Early Melanoma Identification',
    stageBadge: 'NOT TRAINED • Stage 2 Pending',
    icon: Microscope,
    modality: 'Dermoscopic & Macroscopic Epiluminescence Photography',
    dataset: 'ISIC (International Skin Imaging Collaboration) 2024 Archive',
    backbone: 'EfficientNet-B4 / Vision Transformer (ViT-Base/16)',
    framework: 'PyTorch & Timm',
    inputFormat: 'RGB Clinical Imagery (.png, .jpg), Color-calibrated',
    lossFunction: 'Focal Cross-Entropy with Class Balance Re-weighting',
    targetMetrics: ['Balanced Accuracy >= 0.90', 'Melanoma Sensitivity >= 0.95', 'Specificity >= 0.85'],
    pipelineSteps: [
      { title: '1. Vignette & Hair Removal', desc: 'DullRazor morphological filtering to suppress hair artifact occlusion.' },
      { title: '2. Color Constancy Normalization', desc: 'Gray-World color constancy algorithm to handle illumination variances.' },
      { title: '3. Transformer / CNN Inference', desc: 'Attention token pooling across pigment network patterns and borders.' },
      { title: '4. Clinical Risk Scoring', desc: '7-point checklist integration with ABCD rule confidence alignment.' },
    ],
  },
  'yolo-detection': {
    title: 'Anatomical Lesion & Foreign Object YOLO Localization',
    subtitle: 'Single-Stage High-Velocity Spatial Bounding Box Detector',
    stageBadge: 'CHECKPOINT LOADED • Research Prototype',
    icon: Crosshair,
    modality: 'Chest Radiographs (CXR - DICOM & Standard Images)',
    dataset: 'RSNA Pneumonia Detection Cohort (26,684 radiographs, 9,555 bounding boxes)',
    backbone: 'YOLOv8n Single-Stage Spatial Feature Pyramid',
    framework: 'Ultralytics PyTorch Engine',
    inputFormat: 'Bounding Box Annotated Radiographs (Darknet normalized xc, yc, w, h)',
    lossFunction: 'Complete IoU (CIoU) Loss + Varifocal Classification Loss',
    targetMetrics: ['mAP@50 >= 0.82', 'mAP@50-95 >= 0.65', 'Inference Latency < 15ms'],
    pipelineSteps: [
      { title: '1. Anchor & Spatial Calibration', desc: 'Aspect ratio anchor optimization tuned for nodular and linear foreign objects.' },
      { title: '2. Path Aggregation Network (PAN)', desc: 'Bi-directional feature pyramid for tiny nodule detection.' },
      { title: '3. Decoupled Head Evaluation', desc: 'Separation of objectness, class score, and spatial box regression.' },
      { title: '4. Non-Maximum Suppression (NMS)', desc: 'Soft-NMS suppression for overlapping radiological lesions.' },
    ],
  },
  'clinical-ml': {
    title: 'Clinical Tabular Risk Predictor',
    subtitle: 'Random Forest Ensemble for Patient Cardiovascular Risk Stratification',
    stageBadge: 'TRAINED • Active Model',
    icon: TableProperties,
    modality: 'Structured EHR, Vital Signs, Lab Chemistries & Demographics',
    dataset: 'UCI Cleveland Clinic Heart Disease Cohort (303 genuine patient records)',
    backbone: 'Random Forest Ensemble (150 trees, max_depth=5, balanced class weights)',
    framework: 'Python 3 / Scikit-Learn & Joblib',
    inputFormat: '13 Standard Clinical Attributes (Age, Sex, CP, BP, Chol, FBS, ECG, HR, Angina, ST, Slope, Vessels, Thal)',
    lossFunction: 'Gini Impurity / Cross-Entropy with Stratified 5-Fold Validation',
    targetMetrics: ['AUROC = 0.9328 (Test Split)', 'Balanced Accuracy = 86.97%', 'Sensitivity = 85.71%', 'Specificity = 88.24%'],
    pipelineSteps: [
      { title: '1. Missing Value Imputation', desc: 'Encapsulated ColumnTransformer median/mode imputer fitted strictly on training data.' },
      { title: '2. Standard Scaling & One-Hot Encoding', desc: 'StandardScaler for numerical attributes and OneHotEncoder for categorical codes.' },
      { title: '3. Ensemble Random Forest Training', desc: '150 decision trees trained with zero patient data leakage and 5-fold CV.' },
      { title: '4. Feature Importance Decomposition', desc: 'Extraction of top predictive clinical markers and risk factor highlighting.' },
    ],
  },
  'explainable-ai': {
    title: 'Explainable AI (XAI) Attribution Suite',
    subtitle: 'ResNet-50 Grad-CAM Saliency Maps & Attribution Layer Hooks',
    stageBadge: 'OPERATIONAL • Grad-CAM Ready',
    icon: Eye,
    modality: 'Convolutional Feature Attribution Hooks (layer4 Bottleneck)',
    dataset: 'Validation Benchmarks with Radiologist Ground-Truth Bounding Boxes',
    backbone: 'Grad-CAM (Gradient-Weighted Class Activation Mapping)',
    framework: 'PyTorch / Torchvision & Matplotlib (Jet Colormap)',
    inputFormat: 'Radiograph image tensors passed through trained ResNet-50 forward & backward hooks',
    lossFunction: 'Pointwise ReLU Activation Gradient Weighting',
    targetMetrics: ['Layer4 Saliency Localization', 'Real-time CPU Latency < 1.2s', 'Alpha Blended Heatmap Overlay'],
    pipelineSteps: [
      { title: '1. Forward Activation Hooking', desc: 'Gradient tap attached to ResNet-50 layer4 final convolutional block.' },
      { title: '2. Gradient Weight Pooling', desc: 'Global average pooling of gradients with positive ReLU rectification.' },
      { title: '3. Bilinear Heatmap Upsampling', desc: 'Overlay generation with customized Jet colormap and alpha transparency.' },
      { title: '4. Diagnostic Overlay Export', desc: 'Lightweight Base64 JPEG data URL encoding for fast browser visualization.' },
    ],
  },
  'research-assistant': {
    title: 'Medical Literature & Protocol Research Assistant (RAG)',
    subtitle: 'Evidence-Based Research Copilot Grounded in Peer-Reviewed Citations',
    stageBadge: 'NOT TRAINED • Stage 4 Pending',
    icon: Bot,
    modality: 'Biomedical Text, Clinical Trials & Diagnostic Protocols',
    dataset: 'PubMed Central Open Access & Cochrane Systematic Reviews',
    backbone: 'RAG Pipeline (BioLinkBERT Vector Store + Dense Retrieval + LLM)',
    framework: 'LangChain / LlamaIndex / FAISS / MongoDB Atlas Vector Search',
    inputFormat: 'Natural Language Research Queries & Clinical Literature PDFs',
    lossFunction: 'Context Recall, Faithfulness & Citation Grounding Evaluation (Ragas)',
    targetMetrics: ['Hallucination Rate < 1.0%', 'Citation Precision >= 95%', 'Clinical Answer Relevancy >= 92%'],
    pipelineSteps: [
      { title: '1. Literature Ingestion & Chunking', desc: 'Semantic section-aware chunking preserving tables and bibliography.' },
      { title: '2. Dense Vector Embedding', desc: 'Domain-specific BioLinkBERT embedding generation and Atlas vector indexing.' },
      { title: '3. Hybrid Sparse/Dense Search', desc: 'BM25 + Cosine similarity retrieval with maximal marginal relevance.' },
      { title: '4. Grounded Synthesis with Citations', desc: 'Constrained generative synthesis enforcing strict DOI attribution links.' },
    ],
  },
  'model-evaluation': {
    title: 'Comprehensive Model Evaluation & Benchmarking Suite',
    subtitle: 'Standardized Validation Harness for Multi-Disease AI Systems',
    stageBadge: 'Stage 2 Ingestion Pending',
    icon: BarChart3,
    modality: 'Cross-Modality Validation Matrices',
    dataset: 'Stratified 5-Fold Cross Validation Sets',
    backbone: 'Scikit-Learn Evaluation & MONAI Metric Framework',
    framework: 'Python NumPy / SciPy / Scikit-Learn',
    inputFormat: 'Ground Truth Labels + Model Probability Tensors',
    lossFunction: 'Brier Score, Expected Calibration Error (ECE)',
    targetMetrics: ['Multi-Class Confusion Matrix', 'ROC & Precision-Recall Curves', 'Statistical Confidence Intervals (95% CI)'],
    pipelineSteps: [
      { title: '1. Test Set Isolation', desc: 'Strict patient-level leakage prevention between train and test partitions.' },
      { title: '2. Threshold Optimization', desc: 'Youden J-statistic and cost-matrix tuning based on clinical severity.' },
      { title: '3. Calibration Assessment', desc: 'Platt scaling and isotonic regression for dependable probability outputs.' },
      { title: '4. Regulatory Export', desc: 'Automated generation of model cards adhering to medical AI reporting guidelines.' },
    ],
  },
  'analysis-history': {
    title: 'Immutable Analysis Audit Trail & Case History',
    subtitle: 'Regulatory Compliance Logging for Every Ingested Study and Metric',
    stageBadge: 'Active in Stage 1 Foundation',
    icon: History,
    modality: 'Audit Metadata & System Execution Telemetry',
    dataset: 'MongoDB Atlas Historical Case Collection',
    backbone: 'Async Event Logger with Schema Validation',
    framework: 'Motor / PyMongo / Pydantic v2',
    inputFormat: 'JSON Audit Payloads with SHA-256 Checksums',
    lossFunction: 'N/A (Cryptographic Tamper-Evident Ledger)',
    targetMetrics: ['Audit Completeness 100%', 'Query Retrieval Latency < 50ms', 'Zero Data Mutation'],
    pipelineSteps: [
      { title: '1. Telemetry Capture', desc: 'Automated recording of model version, input checksum, and inference timestamp.' },
      { title: '2. MongoDB Atlas Persistence', desc: 'Indexed storage in collection for fast chronological and modality queries.' },
      { title: '3. Case Review Interface', desc: 'Radiologist verification flags and clinical discrepancy tracking.' },
      { title: '4. Anonymized Export', desc: 'DICOM de-identification for multi-center research data sharing.' },
    ],
  },
  settings: {
    title: 'Platform Configuration & Research Environment',
    subtitle: 'Backend API Connectivity, MongoDB Atlas Settings, and Environment Parameters',
    stageBadge: 'Active in Stage 1 Foundation',
    icon: SettingsIcon,
    modality: 'Environment & Configuration Management',
    dataset: 'Environment Configuration (.env / Settings Class)',
    backbone: 'Pydantic v2 BaseSettings Engine',
    framework: 'Python FastAPI & Vite Client Config',
    inputFormat: 'Environment Variables & Application Parameters',
    lossFunction: 'N/A',
    targetMetrics: ['Configuration Validation 100%', 'Secret Leakage 0%'],
    pipelineSteps: [
      { title: '1. Environment Loading', desc: 'Strict validation of MONGODB_URI, PORT, and CORS_ORIGINS from .env.' },
      { title: '2. Connection Validation', desc: 'Non-blocking asynchronous health ping to cloud MongoDB Atlas cluster.' },
      { title: '3. Dependency Boundary Lock', desc: 'Ensures zero leakage between Python backend packages and Node modules.' },
      { title: '4. Stage Progression Safeguard', desc: 'Strict gatekeeping preventing premature model calls before training.' },
    ],
  },
};

export const ModuleWorkbench: React.FC<ModuleWorkbenchProps> = ({
  tabId,
  onBackToDashboard,
  onSelectTab,
}) => {
  const config = BLUEPRINTS[tabId] || BLUEPRINTS['chest-xray'];
  const Icon = config.icon;

  const { reopenHistoricalCase } = useAnalysis();

  const [historyItems, setHistoryItems] = useState<PredictionHistoryItem[]>([]);
  const [historyLoading, setHistoryLoading] = useState<boolean>(false);
  const [dbConnected, setDbConnected] = useState<boolean>(false);
  const [historyError, setHistoryError] = useState<string | null>(null);
  const [downloadingId, setDownloadingId] = useState<string | null>(null);
  const [reportError, setReportError] = useState<string | null>(null);

  const handleDownloadReport = async (recordId: string, item?: PredictionHistoryItem) => {
    if (!recordId && !item) return;
    setDownloadingId(recordId || 'direct');
    setReportError(null);
    try {
      const targetFilename = `MediVision_Report_${recordId || 'case'}.pdf`;
      const fallback = item ? (item as unknown as Record<string, unknown>) : undefined;
      const res = await downloadAnalysisReport(recordId, targetFilename, fallback);
      if (!res.success) {
        setReportError(res.error || `Failed to download report for record ${recordId}`);
      }
    } catch (err: unknown) {
      setReportError(err instanceof Error ? err.message : 'Report download failed');
    } finally {
      setDownloadingId(null);
    }
  };

  const loadHistory = useCallback(async () => {
    setHistoryLoading(true);
    setHistoryError(null);
    try {
      const res = await fetchPredictionHistory(30);
      if (res.data) {
        setHistoryItems(res.data.records);
        setDbConnected(res.data.database_connected);
      } else {
        setHistoryError(res.error || 'Failed to fetch history');
      }
    } catch (err: unknown) {
      setHistoryError(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      setHistoryLoading(false);
    }
  }, []);

  useEffect(() => {
    let active = true;
    if (tabId === 'analysis-history') {
      fetchPredictionHistory(30).then((res) => {
        if (!active) return;
        if (res.data) {
          setHistoryItems(res.data.records);
          setDbConnected(res.data.database_connected);
        } else {
          setHistoryError(res.error || 'Failed to fetch history');
        }
      }).catch((err: unknown) => {
        if (!active) return;
        setHistoryError(err instanceof Error ? err.message : 'Unknown error');
      });
    }
    return () => {
      active = false;
    };
  }, [tabId]);

  const isTrained = tabId === 'clinical-ml' || tabId === 'explainable-ai' || tabId === 'brain-mri' || tabId === 'yolo-detection';

  return (
    <div className="space-y-6">
      {/* Top Banner: Module Title & Safeguard */}
      <div className="rounded-2xl border border-[#EAE3DD] bg-white/95 p-6 backdrop-blur-sm shadow-sm">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div className="flex items-start gap-4">
            <div className="p-3.5 rounded-2xl bg-[#014BAA]/10 text-[#014BAA] border border-[#014BAA]/20 shrink-0">
              <Icon className="h-7 w-7" />
            </div>
            <div>
              <div className="flex items-center gap-2.5 flex-wrap">
                <h2 className="text-xl font-black text-[#1A1512] tracking-tight">{config.title}</h2>
                <StatusBadge status={isTrained ? 'online' : 'untrained'} label={config.stageBadge} />
              </div>
              <p className="mt-1 text-xs text-[#61554E] max-w-2xl">{config.subtitle}</p>
            </div>
          </div>

          <button
            onClick={onBackToDashboard}
            className="self-start md:self-center px-4 py-2.5 rounded-xl bg-[#F8F3F0] hover:bg-[#014BAA] hover:text-[#F8F3F0] text-xs font-bold text-[#1A1512] border border-[#EAE3DD] transition-all shadow-xs"
          >
            ← Return to Dashboard
          </button>
        </div>
      </div>

      {/* Live MongoDB Atlas Audit Trail (active when tabId === 'analysis-history') */}
      {tabId === 'analysis-history' && (
        <div className="rounded-2xl border border-[#EAE3DD] bg-white/95 p-6 shadow-sm backdrop-blur-sm space-y-4">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-3 border-b border-[#EAE3DD]">
            <div className="flex items-center gap-2.5">
              <Database className="h-5 w-5 text-[#014BAA]" />
              <div>
                <h3 className="text-sm font-bold text-[#1A1512]">
                  Live MongoDB Atlas Case History & Telemetry Log
                </h3>
                <p className="text-[11px] text-[#61554E]">
                  Collection: <code className="font-mono text-[#014BAA] bg-[#014BAA]/10 px-1.5 py-0.5 rounded">analysis_history</code> • Database: <code className="font-mono text-[#014BAA] bg-[#014BAA]/10 px-1.5 py-0.5 rounded">medivision_ai</code>
                </p>
              </div>
            </div>
            <button
              onClick={loadHistory}
              disabled={historyLoading}
              className="self-start sm:self-center flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-[#014BAA] hover:bg-[#01357A] text-white text-xs font-semibold shadow-xs disabled:opacity-50 transition-all"
            >
              <RefreshCw className={`h-3.5 w-3.5 ${historyLoading ? 'animate-spin' : ''}`} />
              <span>Refresh Atlas Log</span>
            </button>
          </div>

          {reportError && (
            <div className="p-3 rounded-xl bg-rose-50 border border-rose-300 text-rose-800 text-xs flex items-center justify-between shadow-xs">
              <div className="flex items-center gap-2">
                <AlertCircle className="w-4 h-4 text-rose-600 shrink-0" />
                <span>{reportError}</span>
              </div>
              <button
                type="button"
                onClick={() => setReportError(null)}
                className="text-xs font-bold text-rose-700 hover:text-rose-900 ml-2"
              >
                ✕
              </button>
            </div>
          )}

          {historyItems.length > 0 ? (
            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs border-collapse">
                <thead>
                  <tr className="border-b border-[#EAE3DD] text-[#7C6E66] font-mono text-[10px] uppercase">
                    <th className="py-2.5 px-3">Preview</th>
                    <th className="py-2.5 px-3">Timestamp</th>
                    <th className="py-2.5 px-3">Modality</th>
                    <th className="py-2.5 px-3">Filename</th>
                    <th className="py-2.5 px-3">Classification</th>
                    <th className="py-2.5 px-3">Confidence / Risk</th>
                    <th className="py-2.5 px-3">Status</th>
                    <th className="py-2.5 px-3 text-right">Actions</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-[#F0EAE5]">
                  {historyItems.map((item) => {
                    const isAbnormal = item.predicted_class_id === 1;
                    return (
                      <tr key={item.id} className="hover:bg-[#FAF7F5] transition-colors">
                        <td className="py-2 px-3">
                          {item.image_preview ? (
                            <img
                              src={item.image_preview}
                              alt={item.filename}
                              className="w-9 h-9 rounded-lg object-cover border border-[#DFD7D0] shadow-2xs bg-black"
                            />
                          ) : item.modality === 'clinical_ml' ? (
                            <div className="w-9 h-9 rounded-lg bg-[#014BAA]/10 border border-[#014BAA]/20 flex flex-col items-center justify-center text-[#014BAA]" title="Clinical Tabular EHR Record">
                              <TableProperties className="w-4 h-4 text-[#014BAA]" />
                              <span className="text-[7px] font-bold font-mono">EHR</span>
                            </div>
                          ) : (
                            <div className="w-9 h-9 rounded-lg bg-[#FAF7F5] border border-[#EAE3DD] flex items-center justify-center text-[#7C6E66]">
                              <Eye className="w-4 h-4 text-[#014BAA]" />
                            </div>
                          )}
                        </td>
                        <td className="py-2.5 px-3 font-mono text-[11px] text-[#61554E]">
                          {item.timestamp ? new Date(item.timestamp).toLocaleString() : 'N/A'}
                        </td>
                        <td className="py-2.5 px-3 font-medium text-[#1A1512]">
                          {item.modality === 'chest_xray'
                            ? 'Chest Radiograph'
                            : item.modality === 'clinical_ml'
                            ? 'Clinical Tabular ML'
                            : item.modality === 'brain_mri'
                            ? 'Brain MRI 3D U-Net'
                            : item.modality === 'yolo_detection'
                            ? 'YOLO Lesion Localization'
                            : item.modality}
                        </td>
                        <td className="py-2.5 px-3 font-mono text-[11px] text-[#014BAA] truncate max-w-[150px]" title={item.filename}>
                          {item.filename}
                        </td>
                        <td className="py-2.5 px-3">
                          <span
                            className={`inline-block px-2 py-0.5 rounded-full font-bold text-[10px] border ${
                              isAbnormal || (item.risk_tier && item.risk_tier.includes('Elevated'))
                                ? 'bg-amber-50 text-amber-900 border-amber-300'
                                : 'bg-emerald-50 text-emerald-900 border-emerald-300'
                            }`}
                          >
                            {item.risk_tier || item.prediction || 'Classified'}
                          </span>
                        </td>
                        <td className="py-2.5 px-3 font-mono font-bold text-[#1A1512]">
                          {item.confidence !== null && item.confidence !== undefined
                            ? `${(item.confidence * 100).toFixed(1)}%`
                            : item.risk_probability !== null && item.risk_probability !== undefined
                            ? `${(item.risk_probability * 100).toFixed(1)}%`
                            : item.num_detections !== null && item.num_detections !== undefined
                            ? `${item.num_detections} finding(s)`
                            : 'N/A'}
                        </td>
                        <td className="py-2.5 px-3 font-mono text-[10px] text-emerald-700 font-semibold">
                          {item.status}
                        </td>
                        <td className="py-2.5 px-3 text-right">
                          <div className="flex items-center justify-end gap-1.5 whitespace-nowrap">
                            <button
                              type="button"
                              onClick={() => handleDownloadReport(item.id, item)}
                              disabled={downloadingId === item.id}
                              className="inline-flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg bg-[#F8F3F0] hover:bg-[#014BAA] text-[#014BAA] hover:text-white border border-[#014BAA]/30 font-bold text-[11px] shadow-xs active:scale-95 transition-all disabled:opacity-50"
                              title="Download executive clinical research PDF report"
                            >
                              {downloadingId === item.id ? (
                                <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                              ) : (
                                <Download className="w-3.5 h-3.5" />
                              )}
                              <span>{downloadingId === item.id ? 'Generating...' : 'Download Report'}</span>
                            </button>
                            <button
                              type="button"
                              onClick={async () => {
                                await reopenHistoricalCase(item);
                                if (onSelectTab) {
                                  if (item.modality === 'clinical_ml') {
                                    onSelectTab('clinical-ml');
                                  } else if (item.modality === 'brain_mri') {
                                    onSelectTab('brain-mri');
                                  } else if (item.modality === 'yolo_detection') {
                                    onSelectTab('yolo-detection');
                                  } else {
                                    onSelectTab('chest-xray');
                                  }
                                }
                              }}
                              className="inline-flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg bg-[#014BAA] hover:bg-[#01357A] text-[#F8F3F0] font-bold text-[11px] shadow-xs active:scale-95 transition-all"
                              title={
                                item.modality === 'clinical_ml'
                                  ? 'Reopen case in Clinical ML Workbench'
                                  : item.modality === 'brain_mri'
                                  ? 'Reopen case in Brain MRI Workbench'
                                  : item.modality === 'yolo_detection'
                                  ? 'Reopen case in YOLO Detection Workbench'
                                  : 'Reopen case and review radiograph in Chest X-ray Workbench'
                              }
                            >
                              <Eye className="w-3.5 h-3.5" />
                              <span>Reopen</span>
                            </button>
                          </div>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          ) : (
            <div className="py-8 text-center space-y-2 border border-dashed border-[#DFD7D0] rounded-xl bg-[#FAF7F5]">
              <Database className="h-6 w-6 text-[#014BAA] mx-auto" />
              <p className="text-xs font-bold text-[#1A1512]">
                {dbConnected ? 'MongoDB Atlas Cluster Connected' : 'MongoDB Atlas Telemetry Ready'}
              </p>
              <p className="text-[11px] text-[#61554E] max-w-sm mx-auto">
                {dbConnected
                  ? 'No prediction records logged yet. Run a classification in the Chest X-ray Workbench to log your first case!'
                  : historyError || 'Waiting for first case or active network connection.'}
              </p>
            </div>
          )}
        </div>
      )}

      {/* Module 4: Clinical Tabular ML Risk Predictor (Trained & Active) */}
      {tabId === 'clinical-ml' && (
        <ClinicalMLWorkbench />
      )}

      {/* Module 5: Explainable AI Attribution Suite (Grad-CAM Operational) */}
      {tabId === 'explainable-ai' && (
        <ExplainableAIWorkbench
          onNavigateToChestXRay={() => onSelectTab && onSelectTab('chest-xray')}
        />
      )}

      {/* Module 2: Brain MRI Lesion & Tumor Segmentation (3D U-Net Pipeline) */}
      {tabId === 'brain-mri' && (
        <BrainMRIWorkbench />
      )}

      {/* Module 3: YOLOv8 Anatomical Lesion & Bounding Box Localization */}
      {tabId === 'yolo-detection' && (
        <YOLODetectionWorkbench />
      )}

      {/* Untrained Model Sandbox Notice & Planned Specs (Active for unconfigured modules) */}
      {!isTrained && tabId !== 'analysis-history' && (
        <>
          <div className="rounded-2xl border border-amber-300 bg-amber-50/90 p-5 shadow-xs">
            <div className="flex items-start gap-3">
              <AlertTriangle className="h-5 w-5 text-amber-600 shrink-0 mt-0.5" />
              <div className="flex-1">
                <h4 className="text-sm font-bold text-amber-900">
                  Research Blueprint: Execution Locked Until Model Training
                </h4>
                <p className="mt-1 text-xs text-amber-800 leading-relaxed">
                  In strict accordance with project boundaries, no mock predictions or fake inferences are permitted. This module blueprint outlines the deep learning architecture, dataset targets, input specifications, and loss formulations designed for implementation.
                </p>
              </div>
            </div>
          </div>

          {/* Two Column Grid: Technical Specifications & Pipeline Architecture */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Column 1: Technical Specifications */}
            <div className="rounded-2xl border border-[#EAE3DD] bg-white/95 p-6 space-y-4 shadow-sm">
              <div className="flex items-center gap-2 pb-3 border-b border-[#EAE3DD]">
                <Cpu className="h-4 w-4 text-[#014BAA]" />
                <h3 className="text-sm font-bold text-[#1A1512] tracking-tight">
                  Planned Technical Specifications
                </h3>
              </div>

              <div className="space-y-3 text-xs">
                <div className="p-3.5 rounded-xl bg-[#FAF7F5] border border-[#EAE3DD]">
                  <span className="text-[#61554E] font-medium block mb-1">Target Modality:</span>
                  <span className="text-[#1A1512] font-mono text-[11px] font-semibold">{config.modality}</span>
                </div>

                <div className="p-3.5 rounded-xl bg-[#FAF7F5] border border-[#EAE3DD]">
                  <span className="text-[#61554E] font-medium block mb-1">Target Training Dataset:</span>
                  <span className="text-[#014BAA] font-mono text-[11px] font-bold">{config.dataset}</span>
                </div>

                <div className="p-3.5 rounded-xl bg-[#FAF7F5] border border-[#EAE3DD]">
                  <span className="text-[#61554E] font-medium block mb-1">Backbone Architecture:</span>
                  <span className="text-[#1A1512] font-mono text-[11px] font-semibold">{config.backbone}</span>
                </div>

                <div className="p-3.5 rounded-xl bg-[#FAF7F5] border border-[#EAE3DD]">
                  <span className="text-[#61554E] font-medium block mb-1">Framework:</span>
                  <span className="text-[#014BAA] font-mono text-[11px] font-semibold">{config.framework}</span>
                </div>

                <div className="p-3.5 rounded-xl bg-[#FAF7F5] border border-[#EAE3DD]">
                  <span className="text-[#61554E] font-medium block mb-1">Input Format Specification:</span>
                  <span className="text-[#1A1512] font-mono text-[11px]">{config.inputFormat}</span>
                </div>

                <div className="p-3.5 rounded-xl bg-[#FAF7F5] border border-[#EAE3DD]">
                  <span className="text-[#61554E] font-medium block mb-1">Loss Formulation:</span>
                  <span className="text-amber-800 font-mono text-[11px] font-semibold">{config.lossFunction}</span>
                </div>
              </div>
            </div>

            {/* Column 2: Planned Pipeline Steps & Metrics */}
            <div className="rounded-2xl border border-[#EAE3DD] bg-white/95 p-6 space-y-4 flex flex-col justify-between shadow-sm">
              <div>
                <div className="flex items-center gap-2 pb-3 border-b border-[#EAE3DD]">
                  <Layers className="h-4 w-4 text-[#014BAA]" />
                  <h3 className="text-sm font-bold text-[#1A1512] tracking-tight">
                    Architectural Pipeline Flow
                  </h3>
                </div>

                <div className="mt-4 space-y-3">
                  {config.pipelineSteps.map((step, idx) => (
                    <div key={idx} className="p-3.5 rounded-xl bg-[#FAF7F5] border border-[#EAE3DD]">
                      <div className="flex items-center gap-2 text-[#014BAA] font-bold text-xs mb-1">
                        <CheckCircle className="h-3.5 w-3.5 text-[#014BAA] shrink-0" />
                        <span>{step.title}</span>
                      </div>
                      <p className="text-[11px] text-[#61554E] leading-relaxed pl-5.5">{step.desc}</p>
                    </div>
                  ))}
                </div>

                {/* Target Validation Metrics */}
                <div className="mt-5 pt-4 border-t border-[#EAE3DD]">
                  <h4 className="text-xs font-bold text-[#1A1512] mb-2">
                    Planned Evaluation Metrics:
                  </h4>
                  <div className="flex flex-wrap gap-2">
                    {config.targetMetrics.map((metric, i) => (
                      <span
                        key={i}
                        className="px-2.5 py-1 rounded-full bg-[#014BAA]/10 border border-[#014BAA]/20 text-[11px] text-[#014BAA] font-bold font-mono"
                      >
                        {metric}
                      </span>
                    ))}
                  </div>
                </div>
              </div>

              {/* Inactive Upload Blueprint Box */}
              <div className="mt-6 p-5 rounded-2xl border border-dashed border-[#DFD7D0] bg-[#FAF7F5] text-center space-y-2">
                <FileCode className="h-6 w-6 text-[#014BAA] mx-auto" />
                <p className="text-xs font-bold text-[#1A1512]">
                  Interactive Clinical Ingestion Sandbox
                </p>
                <p className="text-[11px] text-[#61554E] max-w-md mx-auto">
                  File upload handlers, DICOM viewer integration, and forward-pass prediction routes will be activated once training pipelines for this modality are complete.
                </p>
                <div className="pt-2">
                  <span className="inline-block px-3 py-1 rounded-full bg-amber-50 border border-amber-300 text-[10px] font-bold text-amber-900 uppercase tracking-wider">
                    MODEL NOT TRAINED • WEIGHTS PENDING
                  </span>
                </div>
              </div>
            </div>
          </div>
        </>
      )}
    </div>
  );
};
