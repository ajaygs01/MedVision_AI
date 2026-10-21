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
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 pb-4 border-b border-[#EAE3DD]">
        <div className="flex items-center gap-3">
          <button
            onClick={onBackToDashboard}
            className="p-2.5 rounded-xl bg-white border border-[#EAE3DD] hover:border-[#014BAA] text-[#1A1512] hover:text-[#014BAA] transition-all shadow-xs"
            title="Return to Dashboard"
          >
            <ArrowLeft className="w-5 h-5" />
          </button>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="text-xl md:text-2xl font-black text-[#1A1512] tracking-tight">
                Chest X-ray CNN Classification Workbench
              </h1>
              <span className="text-xs font-bold px-2.5 py-0.5 rounded-full bg-[#014BAA]/10 text-[#014BAA] border border-[#014BAA]/25">
                Stage 2
              </span>
            </div>
            <p className="text-xs text-[#61554E] mt-0.5">
              RSNA Pneumonia Detection Challenge • ResNet-50 ImageNet Transfer Learning Pipeline
            </p>
          </div>
        </div>

        {/* Real Status Badge from Model Registry */}
        <div className="flex items-center gap-2">
          {isModelTrained ? (
            <div className="flex items-center gap-2 px-3 py-1.5 rounded-xl bg-emerald-50 border border-emerald-300 text-emerald-800 text-xs font-bold shadow-xs">
              <CheckCircle2 className="w-4 h-4 text-emerald-600" />
              <span>WEIGHTS LOADED • ACTIVE</span>
            </div>
          ) : (
            <div className="flex items-center gap-2 px-3 py-1.5 rounded-xl bg-amber-50 border border-amber-300 text-amber-900 text-xs font-bold shadow-xs">
              <AlertTriangle className="w-4 h-4 text-amber-600 animate-pulse" />
              <span>NOT TRAINED • WEIGHTS PENDING</span>
            </div>
          )}
        </div>
      </div>

      {/* Prominent Medical Integrity Warning Banner */}
      {!isModelTrained && (
        <div className="p-4 rounded-2xl bg-amber-50/90 border border-amber-300 text-amber-950 text-sm flex items-start gap-3 shadow-xs">
          <AlertTriangle className="w-5 h-5 text-amber-600 flex-shrink-0 mt-0.5" />
          <div className="space-y-1">
            <h4 className="font-bold text-amber-900 text-sm">
              Model Status: NOT TRAINED — Clinical Diagnostic Integrity Safeguard
            </h4>
            <p className="text-xs text-amber-800 leading-relaxed">
              No trained weights file (<code className="bg-amber-100 px-1.5 py-0.5 rounded font-mono text-[11px] text-amber-900">backend/model_artifacts/chest_xray/resnet50_best.pth</code>) has been loaded into the inference engine.
              In accordance with medical AI safety protocols, <strong>no fabricated confidence values or mock predictions are ever generated</strong>.
              The RSNA dataset is now verified in <code className="bg-amber-100 px-1.5 py-0.5 rounded font-mono text-[11px] text-amber-900">datasets/rsna_pneumonia/</code>. Execute the training pipeline to generate active weights.
            </p>
          </div>
        </div>
      )}

      {/* Main Analysis Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Column: Image Ingestion & Preview (7 Cols) */}
        <div className="lg:col-span-7 space-y-4">
          <div className="bg-white/95 border border-[#EAE3DD] rounded-2xl p-6 space-y-4 shadow-sm backdrop-blur-sm">
            <div className="flex items-center justify-between">
              <h3 className="text-sm font-bold text-[#1A1512] flex items-center gap-2">
                <FileUp className="w-4 h-4 text-[#014BAA]" />
                Radiograph Input Ingestion
              </h3>
              <span className="text-[11px] text-[#61554E] font-medium">
                Supports DICOM (.dcm), PNG, JPEG
              </span>
            </div>

            {/* Drag & Drop Area */}
            <div
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
              onClick={() => fileInputRef.current?.click()}
              className={`border-2 border-dashed rounded-2xl p-8 text-center cursor-pointer transition-all duration-200 flex flex-col items-center justify-center min-h-[220px] ${
                isDragging
                  ? 'border-[#014BAA] bg-[#014BAA]/5'
                  : 'border-[#DFD7D0] hover:border-[#014BAA] bg-[#FAF7F5]/70'
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
                  <div className="w-12 h-12 rounded-2xl bg-[#014BAA]/10 border border-[#014BAA]/20 flex items-center justify-center mx-auto text-[#014BAA]">
                    {isDicom ? <FileText className="w-6 h-6" /> : <Upload className="w-6 h-6" />}
                  </div>
                  <div className="text-center">
                    <p className="text-sm font-bold text-[#1A1512] truncate">{selectedFile.name}</p>
                    <p className="text-xs text-[#61554E]">
                      {(selectedFile.size / 1024).toFixed(1)} KB • {isDicom ? 'DICOM Medical Object' : selectedFile.type || 'Standard Image'}
                    </p>
                  </div>
                  <p className="text-[11px] text-[#014BAA] font-medium">Click or drag another file to replace</p>
                </div>
              ) : (
                <div className="space-y-2 text-center">
                  <div className="w-12 h-12 rounded-2xl bg-[#014BAA]/10 flex items-center justify-center mx-auto text-[#014BAA]">
                    <Upload className="w-6 h-6" />
                  </div>
                  <div>
                    <p className="text-sm font-semibold text-[#1A1512]">
                      Drag & drop a chest radiograph or <span className="text-[#014BAA] font-bold underline underline-offset-2">browse files</span>
                    </p>
                    <p className="text-xs text-[#61554E] mt-1">
                      Max file size: 30 MB. Supports DICOM (.dcm) with 16-bit depth & photometric decoding.
                    </p>
                  </div>
                </div>
              )}
            </div>

            {/* Image Preview Window */}
            {selectedFile && (
              <div className="rounded-2xl border border-[#EAE3DD] bg-[#FAF7F5] p-4 space-y-3">
                <div className="flex items-center justify-between text-xs text-[#61554E]">
                  <span className="font-bold uppercase tracking-wider text-[#1A1512]">
                    Radiograph Inspection View
                  </span>
                  <span className="font-mono">{isDicom ? 'DICOM Stream' : 'RGB Ingest'}</span>
                </div>

                <div className="relative aspect-square max-h-[340px] w-full mx-auto rounded-xl overflow-hidden bg-[#0C060A] flex items-center justify-center border border-[#DFD7D0]">
                  {isDicom ? (
                    <div className="text-center p-6 space-y-2">
                      <FileText className="w-12 h-12 text-[#014BAA] mx-auto" />
                      <p className="text-sm font-bold text-[#F8F3F0]">DICOM Object Attached</p>
                      <p className="text-xs text-[#F8F3F0]/70 max-w-xs">
                        Pixel array and DICOM metadata are decoded server-side via <code className="text-[#F8F3F0] font-mono font-bold">pydicom</code>.
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
                    className="text-xs text-[#61554E] hover:text-[#1A1512] font-medium transition-colors"
                  >
                    Clear selection
                  </button>

                  <button
                    onClick={handleAnalyze}
                    disabled={analyzing}
                    className={`px-5 py-2.5 rounded-xl font-bold text-xs tracking-wide flex items-center gap-2 transition-all ${
                      analyzing
                        ? 'bg-slate-300 text-slate-500 cursor-not-allowed'
                        : 'bg-[#014BAA] hover:bg-[#01357A] text-[#F8F3F0] shadow-md shadow-[#014BAA]/20 active:scale-95'
                    }`}
                  >
                    {analyzing ? (
                      <>
                        <RefreshCw className="w-4 h-4 animate-spin text-[#F8F3F0]" />
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
          <div className="bg-white/95 border border-[#EAE3DD] rounded-2xl p-6 space-y-4 shadow-sm backdrop-blur-sm">
            <h3 className="text-sm font-bold text-[#1A1512] flex items-center gap-2">
              <Activity className="w-4 h-4 text-[#014BAA]" />
              CNN Inference Response
            </h3>

            {/* State 1: Error during API call */}
            {analysisError && (
              <div className="p-4 rounded-xl bg-rose-50 border border-rose-300 text-rose-900 text-xs space-y-1">
                <div className="flex items-center gap-2 text-rose-700 font-bold">
                  <AlertCircle className="w-4 h-4" />
                  <span>Inference Request Failed</span>
                </div>
                <p className="text-rose-800 leading-relaxed">{analysisError}</p>
              </div>
            )}

            {/* State 2: Actual Model Response */}
            {analysisResult && (
              <div className="space-y-4">
                {analysisResult.status === 'not_trained' ? (
                  <div className="p-4 rounded-xl bg-amber-50 border border-amber-300 space-y-3">
                    <div className="flex items-center gap-2 text-amber-900 font-bold text-xs">
                      <AlertTriangle className="w-4 h-4 text-amber-600" />
                      <span>BACKEND NOTIFICATION: MODEL NOT TRAINED</span>
                    </div>
                    <p className="text-xs text-amber-800 leading-relaxed">
                      {analysisResult.message}
                    </p>
                    <div className="pt-2 border-t border-amber-200 text-[11px] text-amber-900 space-y-1 font-mono">
                      <p>Model: {analysisResult.model_name}</p>
                      <p>Status: NOT TRAINED</p>
                      <p>Input File: {analysisResult.filename} ({analysisResult.file_type})</p>
                    </div>
                  </div>
                ) : analysisResult.status === 'success' ? (
                  <div className="space-y-4">
                    {/* Prediction summary */}
                    <div className="p-4 rounded-xl bg-[#FAF7F5] border border-[#EAE3DD] space-y-3">
                      <div className="flex items-center justify-between">
                        <span className="text-xs text-[#61554E] font-medium">Model Classification</span>
                        <span
                          className={`text-xs font-bold px-3 py-1 rounded-full ${
                            analysisResult.predicted_class_id === 1
                              ? 'bg-rose-100 text-rose-800 border border-rose-300'
                              : 'bg-emerald-100 text-emerald-800 border border-emerald-300'
                          }`}
                        >
                          {analysisResult.prediction || 'Unknown'}
                        </span>
                      </div>

                      {/* Confidence bar */}
                      <div className="space-y-1.5">
                        <div className="flex justify-between text-xs">
                          <span className="text-[#61554E]">Predicted Probability</span>
                          <span className="font-mono font-bold text-[#1A1512]">
                            {((analysisResult.confidence || 0) * 100).toFixed(2)}%
                          </span>
                        </div>
                        <div className="h-2 w-full rounded-full bg-[#EAE3DD] overflow-hidden">
                          <div
                            className={`h-full transition-all duration-500 ${
                              (analysisResult.confidence || 0) > (analysisResult.threshold || 0.5)
                                ? 'bg-gradient-to-r from-amber-500 to-[#014BAA]'
                                : 'bg-gradient-to-r from-teal-500 to-emerald-600'
                            }`}
                            style={{
                              width: `${Math.min(100, Math.max(0, (analysisResult.confidence || 0) * 100))}%`,
                            }}
                          />
                        </div>
                      </div>

                      {/* Threshold info */}
                      <div className="flex justify-between text-[11px] text-[#61554E] pt-1 border-t border-[#EAE3DD]">
                        <span>Classification Threshold: {analysisResult.threshold}</span>
                        <span>File: {analysisResult.filename}</span>
                      </div>
                    </div>

                    {/* Telemetry info */}
                    <div className="p-3.5 rounded-xl bg-[#FAF7F5] border border-[#EAE3DD] text-xs">
                      <p className="text-[11px] font-bold text-[#014BAA]">Clinical Safeguard Notice</p>
                      <p className="text-[11px] text-[#61554E] mt-1 italic">
                        {analysisResult.disclaimer}
                      </p>
                    </div>
                  </div>
                ) : null}
              </div>
            )}

            {/* State 3: Empty state when no analysis performed */}
            {!analysisResult && !analysisError && !analyzing && (
              <div className="py-8 px-4 text-center space-y-2 border border-dashed border-[#DFD7D0] rounded-xl bg-[#FAF7F5]/60">
                <Info className="w-6 h-6 text-[#014BAA] mx-auto" />
                <p className="text-xs font-bold text-[#1A1512]">Awaiting Radiograph Ingestion</p>
                <p className="text-[11px] text-[#61554E] max-w-xs mx-auto">
                  Select a frontal chest radiograph (DICOM or PNG/JPEG) and click &quot;Run Model Inference&quot; to test the pipeline.
                </p>
              </div>
            )}

            {/* State 4: Loading state */}
            {analyzing && (
              <div className="py-8 px-4 text-center space-y-3 border border-[#014BAA]/30 rounded-xl bg-[#014BAA]/5 animate-pulse">
                <RefreshCw className="w-8 h-8 text-[#014BAA] mx-auto animate-spin" />
                <div className="space-y-1">
                  <p className="text-xs font-bold text-[#1A1512]">Evaluating Radiograph via ResNet-50</p>
                  <p className="text-[11px] text-[#61554E]">
                    Executing intensity scaling, 224×224 bilinear resizing, and PyTorch forward pass...
                  </p>
                </div>
              </div>
            )}
          </div>

          {/* Technical Architecture Specs Card */}
          <div className="bg-white/95 border border-[#EAE3DD] rounded-2xl p-6 space-y-3 shadow-sm backdrop-blur-sm">
            <h3 className="text-sm font-bold text-[#1A1512] flex items-center gap-2">
              <Sliders className="w-4 h-4 text-[#014BAA]" />
              Stage 2 Technical Architecture
            </h3>

            <div className="space-y-2.5 text-xs text-[#1A1512]">
              <div className="flex items-start justify-between py-1 border-b border-[#EAE3DD]">
                <span className="text-[#61554E] flex items-center gap-1.5">
                  <Cpu className="w-3.5 h-3.5 text-[#014BAA]" />
                  Backbone Network
                </span>
                <span className="font-mono text-[#1A1512] font-semibold text-right">ResNet-50 (ImageNet-1K)</span>
              </div>

              <div className="flex items-start justify-between py-1 border-b border-[#EAE3DD]">
                <span className="text-[#61554E] flex items-center gap-1.5">
                  <Database className="w-3.5 h-3.5 text-[#014BAA]" />
                  Primary Dataset
                </span>
                <span className="font-mono text-[#014BAA] font-bold text-right">RSNA Pneumonia Challenge</span>
              </div>

              <div className="flex items-start justify-between py-1 border-b border-[#EAE3DD]">
                <span className="text-[#61554E] flex items-center gap-1.5">
                  <Layers className="w-3.5 h-3.5 text-[#014BAA]" />
                  Resolution & Channels
                </span>
                <span className="font-mono text-[#1A1512] font-semibold text-right">224 × 224 px • 3-Channel RGB</span>
              </div>

              <div className="flex items-start justify-between py-1 border-b border-[#EAE3DD]">
                <span className="text-[#61554E] flex items-center gap-1.5">
                  <Activity className="w-3.5 h-3.5 text-[#014BAA]" />
                  Loss & Optimization
                </span>
                <span className="font-mono text-[#1A1512] font-semibold text-right">BCEWithLogits (Weighted)</span>
              </div>

              <div className="flex items-start justify-between py-1">
                <span className="text-[#61554E] flex items-center gap-1.5">
                  <ShieldAlert className="w-3.5 h-3.5 text-[#014BAA]" />
                  Splitting Strategy
                </span>
                <span className="font-mono text-emerald-700 font-bold text-right">Zero Patient Leakage</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
