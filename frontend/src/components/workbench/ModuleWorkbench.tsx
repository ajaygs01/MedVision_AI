import React from 'react';
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
} from 'lucide-react';
import type { NavTabId } from '../../types';
import { StatusBadge } from '../common/StatusBadge';

interface ModuleWorkbenchProps {
  tabId: NavTabId;
  onBackToDashboard: () => void;
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
    subtitle: 'Volumetric Attention U-Net for Precise Anatomical Delineation',
    stageBadge: 'NOT TRAINED • Stage 2 Pending',
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
    stageBadge: 'NOT TRAINED • Stage 3 Pending',
    icon: Crosshair,
    modality: 'Radiographs, CT Slices & Fluoroscopy',
    dataset: 'DeepLesion & Curated Surgical Foreign Body Datasets',
    backbone: 'YOLOv8x / YOLOv11 Medical Spatial Feature Pyramid',
    framework: 'Ultralytics PyTorch Engine',
    inputFormat: 'Bounding Box Annotated Imagery (YOLO darknet / COCO JSON format)',
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
    title: 'Clinical Tabular Risk & Sepsis Predictor',
    subtitle: 'Gradient-Boosted Decision Trees for Patient Electronic Health Records',
    stageBadge: 'NOT TRAINED • Stage 3 Pending',
    icon: TableProperties,
    modality: 'Structured EHR, Vital Signs, Lab Chemistries & Demographics',
    dataset: 'MIMIC-IV Intensive Care Cohort & UCI Clinical Records',
    backbone: 'XGBoost / LightGBM / CatBoost Ensemble',
    framework: 'Scikit-Learn & XGBoost',
    inputFormat: 'Tabular CSV, Parquet, or HL7 / FHIR JSON Data Feeds',
    lossFunction: 'Binary Log-Loss with Cost-Sensitive Sample Weighting',
    targetMetrics: ['AUROC >= 0.91', 'Brier Score < 0.10 (Calibrated Probabilities)', 'F1-Macro >= 0.84'],
    pipelineSteps: [
      { title: '1. Missing Value Imputation', desc: 'MICE (Multivariate Imputation by Chained Equations) with clinical bounds.' },
      { title: '2. Dynamic Vital Feature Engineering', desc: 'Rolling window delta values for heart rate, blood pressure, and lactate.' },
      { title: '3. Tree Boosting Ensemble', desc: 'Gradient boosted trees with monotonic constraints for physiological safety.' },
      { title: '4. TreeSHAP Attribution Integration', desc: 'Patient-level feature impact decomposition for clinical explainability.' },
    ],
  },
  'explainable-ai': {
    title: 'Explainable AI (XAI) & Attribution Suite',
    subtitle: 'Saliency Heatmaps, Score-CAM & Mathematical Feature Attribution',
    stageBadge: 'NOT TRAINED • Stage 4 Pending',
    icon: Eye,
    modality: 'Cross-Modality Attribution Hooks (Imaging & Tabular)',
    dataset: 'Validation Benchmarks with Radiologist Ground-Truth Bounding Boxes',
    backbone: 'Grad-CAM++, Score-CAM, Integrated Gradients & KernelSHAP',
    framework: 'PyTorch Captum & SHAP Library',
    inputFormat: 'Layer feature activation tensors & trained model forward hooks',
    lossFunction: 'Pointing Game Localization Accuracy & Deletion AUC',
    targetMetrics: ['Pointing Game Accuracy >= 78%', 'Attribution Faithfulness Score', 'Human Radiologist Agreement'],
    pipelineSteps: [
      { title: '1. Forward Activation Hooking', desc: 'Gradient tap attached to final convolutional feature map layer.' },
      { title: '2. Gradient Weight Pooling', desc: 'Global average pooling of gradients with positive ReLU rectification.' },
      { title: '3. Bilinear Heatmap Upsampling', desc: 'Overlay generation with customized colormap (Jet / Turbo / Viridis).' },
      { title: '4. Thresholded Diagnostic Contour', desc: 'Extraction of high-activation contours with anatomical landmark mapping.' },
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
}) => {
  const config = BLUEPRINTS[tabId] || BLUEPRINTS['chest-xray'];
  const Icon = config.icon;

  return (
    <div className="space-y-6">
      {/* Top Banner: Module Title & Safeguard */}
      <div className="rounded-xl border border-slate-800 bg-[#0C1430]/80 p-6 backdrop-blur-sm">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div className="flex items-start gap-4">
            <div className="p-3 rounded-xl bg-teal-500/10 text-teal-400 border border-teal-500/20 shrink-0">
              <Icon className="h-7 w-7" />
            </div>
            <div>
              <div className="flex items-center gap-2.5 flex-wrap">
                <h2 className="text-xl font-bold text-white tracking-tight">{config.title}</h2>
                <StatusBadge status="untrained" label={config.stageBadge} />
              </div>
              <p className="mt-1 text-xs text-slate-300 max-w-2xl">{config.subtitle}</p>
            </div>
          </div>

          <button
            onClick={onBackToDashboard}
            className="self-start md:self-center px-4 py-2 rounded-lg bg-slate-900 hover:bg-slate-800 text-xs font-semibold text-slate-200 border border-slate-800 transition-colors"
          >
            ← Return to Dashboard
          </button>
        </div>
      </div>

      {/* Untrained Model Sandbox Notice */}
      <div className="rounded-xl border border-amber-500/30 bg-amber-950/20 p-5">
        <div className="flex items-start gap-3">
          <AlertTriangle className="h-5 w-5 text-amber-400 shrink-0 mt-0.5" />
          <div className="flex-1">
            <h4 className="text-sm font-semibold text-amber-200">
              Stage 1 Architecture Blueprint: Execution Locked
            </h4>
            <p className="mt-1 text-xs text-amber-300/80 leading-relaxed">
              In strict accordance with project boundaries, no mock predictions or fake inferences are permitted. This module blueprint outlines the exact deep learning architecture, dataset targets, input specifications, and loss formulations designed for Stage 2 & 3 implementation.
            </p>
          </div>
        </div>
      </div>

      {/* Two Column Grid: Technical Specifications & Pipeline Architecture */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Column 1: Technical Specifications */}
        <div className="rounded-xl border border-slate-800 bg-[#0C1430]/70 p-5 space-y-4">
          <div className="flex items-center gap-2 pb-3 border-b border-slate-800">
            <Cpu className="h-4 w-4 text-teal-400" />
            <h3 className="text-sm font-semibold text-white tracking-tight">
              Planned Technical Specifications
            </h3>
          </div>

          <div className="space-y-3 text-xs">
            <div className="p-3 rounded-lg bg-slate-900/60 border border-slate-800/80">
              <span className="text-slate-400 font-medium block mb-1">Target Modality:</span>
              <span className="text-slate-200 font-mono text-[11px]">{config.modality}</span>
            </div>

            <div className="p-3 rounded-lg bg-slate-900/60 border border-slate-800/80">
              <span className="text-slate-400 font-medium block mb-1">Target Training Dataset:</span>
              <span className="text-teal-300 font-mono text-[11px]">{config.dataset}</span>
            </div>

            <div className="p-3 rounded-lg bg-slate-900/60 border border-slate-800/80">
              <span className="text-slate-400 font-medium block mb-1">Backbone Architecture:</span>
              <span className="text-slate-200 font-mono text-[11px]">{config.backbone}</span>
            </div>

            <div className="p-3 rounded-lg bg-slate-900/60 border border-slate-800/80">
              <span className="text-slate-400 font-medium block mb-1">Framework:</span>
              <span className="text-cyan-300 font-mono text-[11px]">{config.framework}</span>
            </div>

            <div className="p-3 rounded-lg bg-slate-900/60 border border-slate-800/80">
              <span className="text-slate-400 font-medium block mb-1">Input Format Specification:</span>
              <span className="text-slate-300 font-mono text-[11px]">{config.inputFormat}</span>
            </div>

            <div className="p-3 rounded-lg bg-slate-900/60 border border-slate-800/80">
              <span className="text-slate-400 font-medium block mb-1">Loss Formulation:</span>
              <span className="text-amber-300 font-mono text-[11px]">{config.lossFunction}</span>
            </div>
          </div>
        </div>

        {/* Column 2: Planned Pipeline Steps & Metrics */}
        <div className="rounded-xl border border-slate-800 bg-[#0C1430]/70 p-5 space-y-4 flex flex-col justify-between">
          <div>
            <div className="flex items-center gap-2 pb-3 border-b border-slate-800">
              <Layers className="h-4 w-4 text-teal-400" />
              <h3 className="text-sm font-semibold text-white tracking-tight">
                Architectural Pipeline Flow
              </h3>
            </div>

            <div className="mt-4 space-y-3">
              {config.pipelineSteps.map((step, idx) => (
                <div key={idx} className="p-3 rounded-lg bg-slate-900/60 border border-slate-800/80">
                  <div className="flex items-center gap-2 text-teal-300 font-semibold text-xs mb-1">
                    <CheckCircle className="h-3.5 w-3.5 text-teal-400 shrink-0" />
                    <span>{step.title}</span>
                  </div>
                  <p className="text-[11px] text-slate-300 leading-relaxed pl-5.5">{step.desc}</p>
                </div>
              ))}
            </div>

            {/* Target Validation Metrics */}
            <div className="mt-5 pt-4 border-t border-slate-800">
              <h4 className="text-xs font-semibold text-slate-300 mb-2">
                Planned Stage 2 Evaluation Metrics:
              </h4>
              <div className="flex flex-wrap gap-2">
                {config.targetMetrics.map((metric, i) => (
                  <span
                    key={i}
                    className="px-2.5 py-1 rounded-md bg-teal-950/60 border border-teal-500/30 text-[11px] text-teal-200 font-mono"
                  >
                    {metric}
                  </span>
                ))}
              </div>
            </div>
          </div>

          {/* Inactive Upload Blueprint Box */}
          <div className="mt-6 p-4 rounded-xl border border-dashed border-slate-700 bg-slate-900/40 text-center space-y-2">
            <FileCode className="h-6 w-6 text-slate-400 mx-auto" />
            <p className="text-xs font-medium text-slate-300">
              Interactive Clinical Ingestion Sandbox (Stage 2)
            </p>
            <p className="text-[11px] text-slate-400 max-w-md mx-auto">
              File upload handlers, DICOM viewer integration, and forward-pass prediction routes will be activated once datasets are mounted and models are trained.
            </p>
            <div className="pt-2">
              <span className="inline-block px-3 py-1.5 rounded-lg bg-amber-950/60 border border-amber-500/40 text-[10px] font-bold text-amber-300 uppercase tracking-wider">
                MODEL NOT TRAINED • WEIGHTS UNLOADED
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
