import React, { useState, useRef } from 'react';
import {
  ArrowLeft,
  Upload,
  FileUp,
  AlertTriangle,
  CheckCircle2,
  AlertCircle,
  Cpu,
  Layers,
  Activity,
  FileText,
  RefreshCw,
  Info,
  ShieldAlert,
  Sliders,
  Database,
} from 'lucide-react';
import type { ModelItemStatus, ChestXRayClassifyResponse } from '../../types';
import { classifyChestXRay } from '../../services/api';

interface ChestXRayWorkbenchProps {
  onBackToDashboard: () => void;
  modelStatus?: ModelItemStatus;
}

export function ChestXRayWorkbench({
  onBackToDashboard,
  modelStatus,
}: ChestXRayWorkbenchProps) {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [isDicom, setIsDicom] = useState<boolean>(false);
  const [isDragging, setIsDragging] = useState<boolean>(false);

  const [analyzing, setAnalyzing] = useState<boolean>(false);
  const [analysisResult, setAnalysisResult] = useState<ChestXRayClassifyResponse | null>(null);
  const [analysisError, setAnalysisError] = useState<string | null>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);

  const isModelTrained = modelStatus?.trained === true;

  const handleFileChange = (file: File) => {
    setSelectedFile(file);
    setAnalysisResult(null);
    setAnalysisError(null);

    const isDcm = file.name.toLowerCase().endsWith('.dcm') || file.type === 'application/dicom';
    setIsDicom(isDcm);

    if (isDcm) {
      setPreviewUrl(null);
    } else {
      const url = URL.createObjectURL(file);
      setPreviewUrl(url);
    }
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileChange(e.dataTransfer.files[0]);
    }
  };

  const handleAnalyze = async () => {
    if (!selectedFile) return;

    setAnalyzing(true);
    setAnalysisError(null);
    setAnalysisResult(null);

    try {
      const response = await classifyChestXRay(selectedFile);
      if (response.data) {
        setAnalysisResult(response.data);
      } else {
        setAnalysisError(response.error || 'Inference request failed.');
      }
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'An unexpected error occurred during classification.';
      setAnalysisError(message);
    } finally {
      setAnalyzing(false);
    }
  };

  const handleReset = () => {
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }
    setSelectedFile(null);
    setPreviewUrl(null);
    setIsDicom(false);
    setAnalysisResult(null);
    setAnalysisError(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  return (
    <div className="space-y-6">
      {/* Top Header & Breadcrumb */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 pb-4 border-b border-slate-800">
        <div className="flex items-center gap-3">
          <button
            onClick={onBackToDashboard}
            className="p-2 rounded-lg bg-slate-800/80 hover:bg-slate-700/80 text-slate-300 hover:text-white transition-colors"
            title="Return to Dashboard"
          >
            <ArrowLeft className="w-5 h-5" />
          </button>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="text-xl md:text-2xl font-bold text-white tracking-tight">
                Chest X-ray CNN Classification Workbench
              </h1>
              <span className="text-xs font-semibold px-2 py-0.5 rounded-full bg-teal-500/10 text-teal-400 border border-teal-500/20">
                Stage 2
              </span>
            </div>
            <p className="text-xs text-slate-400 mt-0.5">
              RSNA Pneumonia Detection Challenge • ResNet-50 ImageNet Transfer Learning Pipeline
            </p>
          </div>
        </div>

        {/* Real Status Badge from Model Registry */}
        <div className="flex items-center gap-2">
          {isModelTrained ? (
            <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-emerald-500/10 border border-emerald-500/30 text-emerald-300 text-xs font-semibold">
              <CheckCircle2 className="w-4 h-4 text-emerald-400" />
              <span>WEIGHTS LOADED • ACTIVE</span>
            </div>
          ) : (
            <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-amber-500/10 border border-amber-500/30 text-amber-300 text-xs font-semibold">
              <AlertTriangle className="w-4 h-4 text-amber-400 animate-pulse" />
              <span>NOT TRAINED • WEIGHTS PENDING</span>
            </div>
          )}
        </div>
      </div>

      {/* Prominent Medical Integrity Warning Banner */}
      {!isModelTrained && (
        <div className="p-4 rounded-xl bg-amber-950/20 border border-amber-500/30 text-amber-200/90 text-sm flex items-start gap-3 shadow-lg shadow-black/20">
          <AlertTriangle className="w-5 h-5 text-amber-400 flex-shrink-0 mt-0.5" />
          <div className="space-y-1">
            <h4 className="font-semibold text-amber-300 text-sm">
              Model Status: NOT TRAINED — Clinical Diagnostic Integrity Safeguard
            </h4>
            <p className="text-xs text-amber-200/80 leading-relaxed">
              No trained weights file (<code className="bg-amber-950/40 px-1 py-0.5 rounded font-mono text-[11px] text-amber-200">model_artifacts/chest_xray_cnn/best_model.pth</code>) has been loaded into the inference engine.
              In accordance with medical AI safety protocols, <strong>no fabricated confidence values or mock predictions are ever generated</strong>.
              To train this model, download the RSNA Pneumonia Detection dataset into <code className="bg-amber-950/40 px-1 py-0.5 rounded font-mono text-[11px] text-amber-200">datasets/rsna_pneumonia/</code> and execute the training pipeline.
            </p>
          </div>
        </div>
      )}

      {/* Main Analysis Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Column: Image Ingestion & Preview (7 Cols) */}
        <div className="lg:col-span-7 space-y-4">
          <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-5 space-y-4 backdrop-blur-sm">
            <div className="flex items-center justify-between">
              <h3 className="text-sm font-semibold text-white flex items-center gap-2">
                <FileUp className="w-4 h-4 text-teal-400" />
                Radiograph Input Ingestion
              </h3>
              <span className="text-[11px] text-slate-400">
                Supports DICOM (.dcm), PNG, JPEG
              </span>
            </div>

            {/* Drag & Drop Area */}
            <div
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
              onClick={() => fileInputRef.current?.click()}
              className={`border-2 border-dashed rounded-xl p-6 text-center cursor-pointer transition-all duration-200 flex flex-col items-center justify-center min-h-[220px] ${
                isDragging
                  ? 'border-teal-400 bg-teal-500/10'
                  : 'border-slate-700/80 hover:border-slate-600 bg-slate-950/40'
              }`}
            >
              <input
                ref={fileInputRef}
                type="file"
                accept=".dcm,image/png,image/jpeg,image/jpg"
                className="hidden"
                onChange={(e) => {
                  if (e.target.files && e.target.files[0]) {
                    handleFileChange(e.target.files[0]);
                  }
                }}
              />

              {selectedFile ? (
                <div className="space-y-3 w-full max-w-md">
                  <div className="w-12 h-12 rounded-full bg-teal-500/10 border border-teal-500/30 flex items-center justify-center mx-auto text-teal-400">
                    {isDicom ? <FileText className="w-6 h-6" /> : <Upload className="w-6 h-6" />}
                  </div>
                  <div className="text-center">
                    <p className="text-sm font-semibold text-white truncate">{selectedFile.name}</p>
                    <p className="text-xs text-slate-400">
                      {(selectedFile.size / 1024).toFixed(1)} KB • {isDicom ? 'DICOM Medical Object' : selectedFile.type || 'Standard Image'}
                    </p>
                  </div>
                  <p className="text-[11px] text-teal-400/80">Click or drag another file to replace</p>
                </div>
              ) : (
                <div className="space-y-2 text-center">
                  <div className="w-12 h-12 rounded-full bg-slate-800/80 flex items-center justify-center mx-auto text-slate-400 group-hover:text-teal-400">
                    <Upload className="w-6 h-6" />
                  </div>
                  <div>
                    <p className="text-sm font-medium text-slate-200">
                      Drag & drop a chest radiograph or <span className="text-teal-400 font-semibold underline underline-offset-2">browse</span>
                    </p>
                    <p className="text-xs text-slate-400 mt-1">
                      Max file size: 30 MB. Supports DICOM (.dcm) with photometric interpretation & 16-bit depth.
                    </p>
                  </div>
                </div>
              )}
            </div>

            {/* Image Preview Window */}
            {selectedFile && (
              <div className="rounded-xl border border-slate-800 bg-slate-950/60 p-4 space-y-3">
                <div className="flex items-center justify-between text-xs text-slate-400">
                  <span className="font-semibold uppercase tracking-wider text-slate-300">
                    Radiograph Inspection Preview
                  </span>
                  <span>{isDicom ? 'DICOM Stream' : 'RGB / Grayscale Ingest'}</span>
                </div>

                <div className="relative aspect-square max-h-[340px] w-full mx-auto rounded-lg overflow-hidden bg-black flex items-center justify-center border border-slate-800/80">
                  {isDicom ? (
                    <div className="text-center p-6 space-y-2">
                      <FileText className="w-12 h-12 text-teal-400 mx-auto" />
                      <p className="text-sm font-semibold text-slate-200">DICOM Object Attached</p>
                      <p className="text-xs text-slate-400 max-w-xs">
                        Raw pixel array and DICOM metadata will be decoded server-side via <code className="text-teal-300 font-mono">pydicom</code> during analysis.
                      </p>
                    </div>
                  ) : previewUrl ? (
                    <img
                      src={previewUrl}
                      alt="Uploaded Radiograph"
                      className="w-full h-full object-contain filter contrast-105"
                    />
                  ) : null}
                </div>

                {/* Controls */}
                <div className="flex items-center justify-between pt-2">
                  <button
                    onClick={handleReset}
                    className="text-xs text-slate-400 hover:text-slate-200 transition-colors"
                  >
                    Clear selection
                  </button>

                  <button
                    onClick={handleAnalyze}
                    disabled={analyzing}
                    className={`px-5 py-2 rounded-lg font-semibold text-xs tracking-wide flex items-center gap-2 transition-all ${
                      analyzing
                        ? 'bg-slate-700 text-slate-400 cursor-not-allowed'
                        : 'bg-teal-500 hover:bg-teal-400 text-slate-950 shadow-lg shadow-teal-500/20'
                    }`}
                  >
                    {analyzing ? (
                      <>
                        <RefreshCw className="w-4 h-4 animate-spin text-slate-300" />
                        <span>Running Inference...</span>
                      </>
                    ) : (
                      <>
                        <Activity className="w-4 h-4" />
                        <span>Run Model Inference</span>
                      </>
                    )}
                  </button>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Right Column: Inference Results & Model Specifications (5 Cols) */}
        <div className="lg:col-span-5 space-y-4">
          {/* Inference Output Card */}
          <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-5 space-y-4 backdrop-blur-sm">
            <h3 className="text-sm font-semibold text-white flex items-center gap-2">
              <Activity className="w-4 h-4 text-teal-400" />
              CNN Inference Response
            </h3>

            {/* State 1: Error during API call */}
            {analysisError && (
              <div className="p-4 rounded-xl bg-rose-950/20 border border-rose-500/30 text-rose-200 text-xs space-y-1">
                <div className="flex items-center gap-2 text-rose-400 font-semibold">
                  <AlertCircle className="w-4 h-4" />
                  <span>Inference Request Failed</span>
                </div>
                <p className="text-rose-300/80 leading-relaxed">{analysisError}</p>
              </div>
            )}

            {/* State 2: Actual Model Response */}
            {analysisResult && (
              <div className="space-y-4">
                {analysisResult.status === 'not_trained' ? (
                  <div className="p-4 rounded-xl bg-amber-500/10 border border-amber-500/30 space-y-3">
                    <div className="flex items-center gap-2 text-amber-400 font-semibold text-xs">
                      <AlertTriangle className="w-4 h-4" />
                      <span>BACKEND NOTIFICATION: MODEL NOT TRAINED</span>
                    </div>
                    <p className="text-xs text-amber-200/90 leading-relaxed">
                      {analysisResult.message}
                    </p>
                    <div className="pt-2 border-t border-amber-500/20 text-[11px] text-amber-300/70 space-y-1 font-mono">
                      <p>Model: {analysisResult.model_name}</p>
                      <p>Status: NOT TRAINED</p>
                      <p>Input File: {analysisResult.filename} ({analysisResult.file_type})</p>
                    </div>
                  </div>
                ) : analysisResult.status === 'success' ? (
                  <div className="space-y-4">
                    {/* Prediction summary */}
                    <div className="p-4 rounded-xl bg-slate-950/60 border border-slate-800 space-y-3">
                      <div className="flex items-center justify-between">
                        <span className="text-xs text-slate-400 font-medium">Model Classification</span>
                        <span
                          className={`text-xs font-semibold px-2.5 py-1 rounded-full ${
                            analysisResult.predicted_class_id === 1
                              ? 'bg-rose-500/10 text-rose-400 border border-rose-500/30'
                              : 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/30'
                          }`}
                        >
                          {analysisResult.prediction || 'Unknown'}
                        </span>
                      </div>

                      {/* Confidence bar */}
                      <div className="space-y-1.5">
                        <div className="flex justify-between text-xs">
                          <span className="text-slate-400">Predicted Probability (Opacity / Pneumonia)</span>
                          <span className="font-mono font-semibold text-white">
                            {((analysisResult.confidence || 0) * 100).toFixed(2)}%
                          </span>
                        </div>
                        <div className="h-2 w-full rounded-full bg-slate-800 overflow-hidden">
                          <div
                            className={`h-full transition-all duration-500 ${
                              (analysisResult.confidence || 0) > (analysisResult.threshold || 0.5)
                                ? 'bg-gradient-to-r from-amber-500 to-rose-500'
                                : 'bg-gradient-to-r from-teal-500 to-emerald-500'
                            }`}
                            style={{
                              width: `${Math.min(100, Math.max(0, (analysisResult.confidence || 0) * 100))}%`,
                            }}
                          />
                        </div>
                      </div>

                      {/* Threshold info */}
                      <div className="flex justify-between text-[11px] text-slate-400 pt-1 border-t border-slate-800/80">
                        <span>Classification Threshold: {analysisResult.threshold}</span>
                        <span>File: {analysisResult.filename}</span>
                      </div>
                    </div>

                    {/* Telemetry info */}
                    <div className="p-3 rounded-lg bg-slate-950/40 border border-slate-800/80 text-xs">
                      <p className="text-[11px] text-slate-400">Clinical Safeguard Notice</p>
                      <p className="text-[11px] text-slate-300 mt-1 italic">
                        {analysisResult.disclaimer}
                      </p>
                    </div>
                  </div>
                ) : null}
              </div>
            )}

            {/* State 3: Empty state when no analysis performed */}
            {!analysisResult && !analysisError && !analyzing && (
              <div className="py-8 px-4 text-center space-y-2 border border-dashed border-slate-800 rounded-xl bg-slate-950/20">
                <Info className="w-6 h-6 text-slate-400 mx-auto" />
                <p className="text-xs font-medium text-slate-300">Awaiting Ingestion & Analysis</p>
                <p className="text-[11px] text-slate-400 max-w-xs mx-auto">
                  Select a frontal chest radiograph (DICOM or PNG/JPEG) and click &quot;Run Model Inference&quot; to test the pipeline.
                </p>
              </div>
            )}

            {/* State 4: Loading state */}
            {analyzing && (
              <div className="py-8 px-4 text-center space-y-3 border border-slate-800 rounded-xl bg-slate-950/40 animate-pulse">
                <RefreshCw className="w-8 h-8 text-teal-400 mx-auto animate-spin" />
                <div className="space-y-1">
                  <p className="text-xs font-semibold text-slate-200">Evaluating Radiograph via ResNet-50</p>
                  <p className="text-[11px] text-slate-400">
                    Executing intensity scaling, 224×224 bilinear resizing, and PyTorch forward pass...
                  </p>
                </div>
              </div>
            )}
          </div>

          {/* Technical Architecture Specs Card */}
          <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-5 space-y-3 backdrop-blur-sm">
            <h3 className="text-sm font-semibold text-white flex items-center gap-2">
              <Sliders className="w-4 h-4 text-teal-400" />
              Stage 2 Technical Architecture
            </h3>

            <div className="space-y-2.5 text-xs text-slate-300">
              <div className="flex items-start justify-between py-1 border-b border-slate-800/80">
                <span className="text-slate-400 flex items-center gap-1.5">
                  <Cpu className="w-3.5 h-3.5 text-teal-400" />
                  Backbone Network
                </span>
                <span className="font-mono text-slate-200 text-right">torchvision ResNet-50 (ImageNet-1K)</span>
              </div>

              <div className="flex items-start justify-between py-1 border-b border-slate-800/80">
                <span className="text-slate-400 flex items-center gap-1.5">
                  <Database className="w-3.5 h-3.5 text-teal-400" />
                  Primary Dataset
                </span>
                <span className="font-mono text-slate-200 text-right">RSNA Pneumonia Detection Challenge</span>
              </div>

              <div className="flex items-start justify-between py-1 border-b border-slate-800/80">
                <span className="text-slate-400 flex items-center gap-1.5">
                  <Layers className="w-3.5 h-3.5 text-teal-400" />
                  Resolution & Channels
                </span>
                <span className="font-mono text-slate-200 text-right">224 × 224 px • 3-Channel RGB</span>
              </div>

              <div className="flex items-start justify-between py-1 border-b border-slate-800/80">
                <span className="text-slate-400 flex items-center gap-1.5">
                  <Activity className="w-3.5 h-3.5 text-teal-400" />
                  Loss & Optimization
                </span>
                <span className="font-mono text-slate-200 text-right">BCEWithLogitsLoss (Class-Weighted)</span>
              </div>

              <div className="flex items-start justify-between py-1">
                <span className="text-slate-400 flex items-center gap-1.5">
                  <ShieldAlert className="w-3.5 h-3.5 text-teal-400" />
                  Splitting Strategy
                </span>
                <span className="font-mono text-slate-200 text-right">Patient-Level Grouping (Zero Leakage)</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
