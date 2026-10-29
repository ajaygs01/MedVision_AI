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
  RotateCcw,
  History,
  Eye,
  Download,
} from 'lucide-react';
import type { ModelItemStatus, ChestXRayClassifyResponse } from '../../types';
import { classifyChestXRay, downloadAnalysisReport } from '../../services/api';
import { useAnalysis } from '../../context/useAnalysis';

interface ChestXRayWorkbenchProps {
  onBackToDashboard: () => void;
  onNavigateToHistory?: () => void;
  modelStatus?: ModelItemStatus;
}

export function ChestXRayWorkbench({
  onBackToDashboard,
  onNavigateToHistory,
  modelStatus,
}: ChestXRayWorkbenchProps) {
  const { activeCase, saveActiveCase, clearActiveCase } = useAnalysis();

  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [isDicom, setIsDicom] = useState<boolean>(false);
  const [isDragging, setIsDragging] = useState<boolean>(false);

  const [analyzing, setAnalyzing] = useState<boolean>(false);
  const [analysisResult, setAnalysisResult] = useState<ChestXRayClassifyResponse | null>(null);
  const [analysisError, setAnalysisError] = useState<string | null>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const activeRequestIdRef = useRef<number>(0);

  const isModelTrained = modelStatus?.trained === true;

  const [viewMode, setViewMode] = useState<'original' | 'gradcam' | 'split'>('original');
  const [downloadingReport, setDownloadingReport] = useState<boolean>(false);
  const [reportError, setReportError] = useState<string | null>(null);

  // Active or restored case determination
  const hasSelectedFile = selectedFile !== null;
  const isViewingRestoredCase = !hasSelectedFile && activeCase !== null;

  // The active radiograph preview
  const activePreview = hasSelectedFile
    ? previewUrl
    : (activeCase?.imagePreview || null);

  // The active Grad-CAM heatmap overlay
  const activeGradcam = hasSelectedFile
    ? (analysisResult?.gradcam_preview || null)
    : (analysisResult?.gradcam_preview || activeCase?.gradcamPreview || null);

  // The active filename
  const activeFilename = hasSelectedFile
    ? selectedFile.name
    : (activeCase?.filename || null);

  const activeIsDicom = hasSelectedFile
    ? isDicom
    : (activeCase?.isDicom || false);

  // Reconstruct result for restored cases or display freshly analyzed result
  const currentResult: ChestXRayClassifyResponse | null = hasSelectedFile
    ? analysisResult
    : (analysisResult || (activeCase ? {
        id: activeCase.recordId || null,
        status: (activeCase.status as 'success' | 'not_trained' | 'error') || 'success',
        model_name: activeCase.modelName,
        prediction: activeCase.prediction,
        predicted_class_id: activeCase.predictedClassId ?? null,
        confidence: activeCase.confidence,
        threshold: activeCase.threshold,
        filename: activeCase.filename,
        file_type: activeCase.fileType,
        message: activeCase.message || 'Analyzed case restored',
        disclaimer: activeCase.disclaimer || 'Academic research prototype. Results are not clinically validated and must not be used as a standalone medical diagnosis.',
        image_preview: activeCase.imagePreview,
        gradcam_preview: activeCase.gradcamPreview || null,
      } : null));

  const handleFileChange = (file: File) => {
    // Invalidate any in-flight inference requests
    activeRequestIdRef.current++;
    setAnalyzing(false);
    setAnalysisResult(null); // Clear result immediately to prevent stale association
    setAnalysisError(null);

    // Validate file extension
    const nameLower = file.name.toLowerCase();
    const isDcm = nameLower.endsWith('.dcm') || file.type === 'application/dicom';
    const isPng = nameLower.endsWith('.png') || file.type === 'image/png';
    const isJpg = nameLower.endsWith('.jpg') || nameLower.endsWith('.jpeg') || file.type === 'image/jpeg';

    if (!isDcm && !isPng && !isJpg) {
      setAnalysisError(`Unsupported file format "${file.name}". Please upload a supported Chest X-ray file: DICOM (.dcm), PNG (.png), or JPEG (.jpg, .jpeg).`);
      setSelectedFile(null);
      if (previewUrl && previewUrl.startsWith('blob:')) {
        URL.revokeObjectURL(previewUrl);
        setPreviewUrl(null);
      }
      setIsDicom(false);
      return;
    }

    // Validate file size (25 MB max)
    const MAX_SIZE_BYTES = 25 * 1024 * 1024;
    if (file.size > MAX_SIZE_BYTES) {
      setAnalysisError(`File size (${(file.size / (1024 * 1024)).toFixed(1)} MB) exceeds the 25 MB limit.`);
      setSelectedFile(null);
      if (previewUrl && previewUrl.startsWith('blob:')) {
        URL.revokeObjectURL(previewUrl);
        setPreviewUrl(null);
      }
      setIsDicom(false);
      return;
    }

    if (file.size === 0) {
      setAnalysisError('The selected file is empty (0 bytes). Please upload a valid radiograph image.');
      setSelectedFile(null);
      if (previewUrl && previewUrl.startsWith('blob:')) {
        URL.revokeObjectURL(previewUrl);
        setPreviewUrl(null);
      }
      setIsDicom(false);
      return;
    }

    setSelectedFile(file);
    setIsDicom(isDcm);

    if (isDcm) {
      if (previewUrl && previewUrl.startsWith('blob:')) {
        URL.revokeObjectURL(previewUrl);
      }
      setPreviewUrl(null);
    } else {
      if (previewUrl && previewUrl.startsWith('blob:')) {
        URL.revokeObjectURL(previewUrl);
      }
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

    const currentRequestId = ++activeRequestIdRef.current;
    setAnalyzing(true);
    setAnalysisError(null);
    setAnalysisResult(null);

    try {
      const response = await classifyChestXRay(selectedFile);
      if (currentRequestId !== activeRequestIdRef.current) {
        return; // Discard response from stale request
      }

      if (response.data) {
        setAnalysisResult(response.data);
        const resolvedPreview = response.data.image_preview || previewUrl;
        if (response.data.image_preview) {
          setPreviewUrl(response.data.image_preview);
        }

        // Persist to shared state, sessionStorage, and IndexedDB
        saveActiveCase({
          recordId: response.data.id || null,
          filename: response.data.filename,
          fileType: response.data.file_type,
          isDicom: isDicom,
          imagePreview: resolvedPreview,
          gradcamPreview: response.data.gradcam_preview || null,
          heatmapPreview: response.data.heatmap_preview || null,
          prediction: response.data.prediction,
          predictedClassId: response.data.predicted_class_id,
          confidence: response.data.confidence,
          threshold: response.data.threshold,
          modelName: response.data.model_name,
          status: response.data.status,
          message: response.data.message,
          disclaimer: response.data.disclaimer,
          timestamp: new Date().toISOString(),
          viewMode: viewMode,
          modality: 'chest_xray',
        });
      } else {
        setAnalysisError(response.error || 'Inference request failed.');
      }
    } catch (err: unknown) {
      if (currentRequestId !== activeRequestIdRef.current) {
        return; // Discard error from stale request
      }
      const message = err instanceof Error ? err.message : 'An unexpected error occurred during classification.';
      setAnalysisError(message);
    } finally {
      if (currentRequestId === activeRequestIdRef.current) {
        setAnalyzing(false);
      }
    }
  };

  const handleNewAnalysis = () => {
    activeRequestIdRef.current++;
    if (previewUrl && previewUrl.startsWith('blob:')) {
      URL.revokeObjectURL(previewUrl);
    }
    setSelectedFile(null);
    setPreviewUrl(null);
    setIsDicom(false);
    setAnalyzing(false);
    setAnalysisResult(null);
    setAnalysisError(null);
    clearActiveCase();
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const activeRecordId = currentResult?.id || activeCase?.recordId || null;

  const handleDownloadReport = async () => {
    if (!currentResult) {
      setReportError('No analysis result available to generate report. Please run inference first.');
      return;
    }
    setDownloadingReport(true);
    setReportError(null);
    try {
      const payload: Record<string, unknown> = {
        id: activeRecordId,
        modality: 'chest_xray',
        model_name: currentResult.model_name || 'Chest X-ray ResNet50 Classifier',
        filename: activeFilename || currentResult.filename || 'chest_radiograph.png',
        file_type: activeIsDicom ? 'dicom' : (currentResult.file_type || 'standard_image'),
        prediction: currentResult.prediction,
        predicted_class_id: currentResult.predicted_class_id,
        confidence: currentResult.confidence,
        threshold: currentResult.threshold || 0.5,
        status: currentResult.status || 'success',
        message: currentResult.message,
        disclaimer: currentResult.disclaimer,
        timestamp: activeCase?.timestamp || new Date().toISOString(),
        image_preview: activePreview || currentResult.image_preview,
        gradcam_preview: activeGradcam || currentResult.gradcam_preview,
        view_mode: viewMode,
      };
      const filename = `MediVision_CXR_Report_${activeRecordId || activeFilename || 'case'}.pdf`;
      const res = await downloadAnalysisReport(activeRecordId, filename, payload);
      if (!res.success) {
        setReportError(res.error || 'Failed to download report.');
      }
    } catch (err) {
      setReportError(err instanceof Error ? err.message : 'Download failed.');
    } finally {
      setDownloadingReport(false);
    }
  };

  const handleCancelSelection = () => {
    if (previewUrl && previewUrl.startsWith('blob:')) {
      URL.revokeObjectURL(previewUrl);
    }
    setSelectedFile(null);
    setPreviewUrl(activeCase?.imagePreview || null);
    setIsDicom(activeCase?.isDicom || false);
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
            <div className="flex items-center gap-2 flex-wrap">
              <h1 className="text-xl md:text-2xl font-black text-[#1A1512] tracking-tight">
                Chest X-ray CNN Classification Workbench
              </h1>
              <span className="text-xs font-bold px-2.5 py-0.5 rounded-full bg-[#014BAA]/10 text-[#014BAA] border border-[#014BAA]/25">
                Stage 2
              </span>
              {isViewingRestoredCase && activeCase && (
                <span className="flex items-center gap-1 text-[11px] font-bold px-2.5 py-0.5 rounded-full bg-emerald-50 text-emerald-800 border border-emerald-300">
                  <History className="w-3 h-3 text-emerald-600" />
                  <span>Restored Case</span>
                </span>
              )}
            </div>
            <p className="text-xs text-[#61554E] mt-0.5">
              RSNA Pneumonia Detection Challenge • ResNet-50 ImageNet Transfer Learning Pipeline
            </p>
          </div>
        </div>

        {/* Action Controls & Real Status Badge from Model Registry */}
        <div className="flex items-center gap-2 flex-wrap">
          {(hasSelectedFile || activeCase) && (
            <button
              onClick={handleNewAnalysis}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-white hover:bg-rose-50 border border-[#EAE3DD] hover:border-rose-300 text-xs font-bold text-[#61554E] hover:text-rose-700 transition-all shadow-xs"
              title="Start a new analysis (clear active case and reset)"
            >
              <RotateCcw className="w-3.5 h-3.5" />
              <span>New Analysis</span>
            </button>
          )}

          {onNavigateToHistory && (
            <button
              onClick={onNavigateToHistory}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-white hover:bg-[#FAF7F5] border border-[#EAE3DD] text-xs font-semibold text-[#1A1512] hover:text-[#014BAA] transition-all shadow-xs"
              title="View analysis history and audit log"
            >
              <History className="w-3.5 h-3.5 text-[#014BAA]" />
              <span>View History</span>
            </button>
          )}

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
              className={`border-2 border-dashed rounded-2xl p-8 text-center cursor-pointer transition-all duration-200 flex flex-col items-center justify-center min-h-[200px] ${
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

              {hasSelectedFile ? (
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
              ) : isViewingRestoredCase && activeCase ? (
                <div className="space-y-3 w-full max-w-md">
                  <div className="w-12 h-12 rounded-2xl bg-emerald-50 border border-emerald-300 flex items-center justify-center mx-auto text-emerald-700">
                    <History className="w-6 h-6" />
                  </div>
                  <div className="text-center">
                    <p className="text-sm font-bold text-[#1A1512] truncate">{activeCase.filename}</p>
                    <p className="text-xs text-[#61554E]">
                      {activeCase.isDicom ? 'DICOM Object' : 'Standard Radiograph'} • Persisted Analysis ({new Date(activeCase.timestamp).toLocaleTimeString()})
                    </p>
                  </div>
                  <p className="text-[11px] text-[#014BAA] font-medium">Click or drag to analyze a new radiograph</p>
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
                      Max file size: 25 MB. Supports DICOM (.dcm) with 16-bit depth & photometric decoding.
                    </p>
                  </div>
                </div>
              )}
            </div>

            {/* Image Preview Window (shown for selected file or restored case) */}
            {(hasSelectedFile || isViewingRestoredCase) && (
              <div className="rounded-2xl border border-[#EAE3DD] bg-[#FAF7F5] p-4 space-y-3">
                <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 text-xs text-[#61554E]">
                  <span className="font-bold uppercase tracking-wider text-[#1A1512] flex items-center gap-1.5">
                    <span>Radiograph Inspection View</span>
                    {isViewingRestoredCase && (
                      <span className="font-normal text-[10px] text-[#014BAA] font-mono">(Restored Case)</span>
                    )}
                  </span>

                  {/* Grad-CAM View Selector */}
                  {activeGradcam ? (
                    <div className="flex items-center gap-1 bg-[#EAE3DD]/70 p-1 rounded-xl">
                      <button
                        type="button"
                        onClick={() => setViewMode('original')}
                        className={`px-2.5 py-1 rounded-lg text-[11px] font-bold transition-all ${
                          viewMode === 'original'
                            ? 'bg-[#014BAA] text-white shadow-xs'
                            : 'text-[#61554E] hover:text-[#1A1512]'
                        }`}
                      >
                        Original Radiograph
                      </button>
                      <button
                        type="button"
                        onClick={() => setViewMode('gradcam')}
                        className={`px-2.5 py-1 rounded-lg text-[11px] font-bold transition-all ${
                          viewMode === 'gradcam'
                            ? 'bg-[#014BAA] text-white shadow-xs'
                            : 'text-[#61554E] hover:text-[#1A1512]'
                        }`}
                      >
                        Grad-CAM (ResNet50)
                      </button>
                      <button
                        type="button"
                        onClick={() => setViewMode('split')}
                        className={`px-2.5 py-1 rounded-lg text-[11px] font-bold transition-all ${
                          viewMode === 'split'
                            ? 'bg-[#014BAA] text-white shadow-xs'
                            : 'text-[#61554E] hover:text-[#1A1512]'
                        }`}
                      >
                        Side-by-Side
                      </button>
                    </div>
                  ) : (
                    <span className="font-mono">
                      {activeIsDicom ? 'DICOM Modality' : 'RGB Radiograph'}
                    </span>
                  )}
                </div>

                {/* Multi-view Display Area */}
                {viewMode === 'split' && activeGradcam && activePreview ? (
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                    <div className="space-y-1">
                      <p className="text-[10px] font-bold uppercase tracking-wider text-[#7C6E66] text-center">Original Radiograph</p>
                      <div className="relative aspect-square max-h-[300px] w-full rounded-xl overflow-hidden bg-[#0C060A] flex items-center justify-center border border-[#DFD7D0]">
                        <img src={activePreview} alt="Original Radiograph" className="w-full h-full object-contain filter contrast-105" />
                      </div>
                    </div>
                    <div className="space-y-1">
                      <p className="text-[10px] font-bold uppercase tracking-wider text-[#014BAA] text-center">Grad-CAM Overlay (layer4)</p>
                      <div className="relative aspect-square max-h-[300px] w-full rounded-xl overflow-hidden bg-[#0C060A] flex items-center justify-center border border-[#014BAA]/30">
                        <img src={activeGradcam} alt="Grad-CAM Saliency Overlay" className="w-full h-full object-contain" />
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="relative aspect-square max-h-[340px] w-full mx-auto rounded-xl overflow-hidden bg-[#0C060A] flex items-center justify-center border border-[#DFD7D0]">
                    {viewMode === 'gradcam' && activeGradcam ? (
                      <div className="relative w-full h-full flex items-center justify-center">
                        <img
                          src={activeGradcam}
                          alt="Grad-CAM Activation Overlay"
                          className="w-full h-full object-contain"
                        />
                        <div className="absolute top-2 left-2 px-2.5 py-1 rounded-lg bg-black/75 backdrop-blur-xs text-[10px] text-amber-300 font-mono font-bold border border-amber-400/40">
                          ResNet-50 layer4 Grad-CAM Overlay
                        </div>
                      </div>
                    ) : activePreview ? (
                      <img
                        src={activePreview}
                        alt={activeFilename || 'Chest Radiograph'}
                        className="w-full h-full object-contain filter contrast-105"
                      />
                    ) : hasSelectedFile && activeIsDicom ? (
                      <div className="text-center p-6 space-y-2">
                        <FileText className="w-12 h-12 text-[#014BAA] mx-auto" />
                        <p className="text-sm font-bold text-[#F8F3F0]">DICOM Object Attached</p>
                        <p className="text-xs text-[#F8F3F0]/70 max-w-xs">
                          Pixel array and DICOM metadata are decoded server-side via <code className="text-[#F8F3F0] font-mono font-bold">pydicom</code>.
                        </p>
                      </div>
                    ) : (
                      <div className="text-center p-6 space-y-2 text-[#F8F3F0]/80">
                        <AlertCircle className="w-10 h-10 text-amber-400 mx-auto" />
                        <p className="text-xs font-semibold text-[#F8F3F0]">Radiograph preview unavailable for this case</p>
                        <p className="text-[11px] text-[#F8F3F0]/60 max-w-xs">
                          Model classification and telemetry are preserved in the audit panel.
                        </p>
                      </div>
                    )}
                  </div>
                )}

                {/* Controls */}
                <div className="flex items-center justify-between pt-2">
                  {hasSelectedFile ? (
                    <>
                      <button
                        onClick={activeCase ? handleCancelSelection : handleNewAnalysis}
                        className="text-xs text-[#61554E] hover:text-[#1A1512] font-medium transition-colors"
                      >
                        {activeCase ? 'Cancel selection' : 'Clear selection'}
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
                    </>
                  ) : (
                    <>
                      <button
                        onClick={handleNewAnalysis}
                        className="text-xs text-rose-700 hover:text-rose-900 font-bold transition-colors"
                      >
                        Remove Current Image
                      </button>

                      <button
                        onClick={() => fileInputRef.current?.click()}
                        className="px-4 py-2 rounded-xl bg-white border border-[#EAE3DD] hover:border-[#014BAA] text-[#014BAA] font-bold text-xs shadow-xs active:scale-95 transition-all"
                      >
                        Upload Different Radiograph
                      </button>
                    </>
                  )}
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Right Column: Inference Results & Model Specifications (5 Cols) */}
        <div className="lg:col-span-5 space-y-4">
          {/* Inference Output Card */}
          <div className="bg-white/95 border border-[#EAE3DD] rounded-2xl p-6 space-y-4 shadow-sm backdrop-blur-sm">
            <div className="flex items-center justify-between">
              <h3 className="text-sm font-bold text-[#1A1512] flex items-center gap-2">
                <Activity className="w-4 h-4 text-[#014BAA]" />
                CNN Inference Response
              </h3>
              {isViewingRestoredCase && activeCase && (
                <span className="text-[10px] font-mono text-[#014BAA] bg-[#014BAA]/10 px-2 py-0.5 rounded-full font-bold">
                  PERSISTED
                </span>
              )}
            </div>

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

            {/* State 2: Actual Model Response (either live or restored) */}
            {currentResult && !analyzing && (
              <div className="space-y-4">
                {currentResult.status === 'not_trained' ? (
                  <div className="p-4 rounded-xl bg-amber-50 border border-amber-300 space-y-3">
                    <div className="flex items-center gap-2 text-amber-900 font-bold text-xs">
                      <AlertTriangle className="w-4 h-4 text-amber-600" />
                      <span>BACKEND NOTIFICATION: MODEL NOT TRAINED</span>
                    </div>
                    <p className="text-xs text-amber-800 leading-relaxed">
                      {currentResult.message}
                    </p>
                    <div className="pt-2 border-t border-amber-200 text-[11px] text-amber-900 space-y-1 font-mono">
                      <p>Model: {currentResult.model_name}</p>
                      <p>Status: NOT TRAINED</p>
                      <p>Input File: {currentResult.filename} ({currentResult.file_type})</p>
                    </div>
                  </div>
                ) : currentResult.status === 'success' ? (
                  <div className="space-y-4">
                    {/* Prediction summary */}
                    <div className="p-4 rounded-xl bg-[#FAF7F5] border border-[#EAE3DD] space-y-3">
                      <div className="flex items-center justify-between">
                        <span className="text-xs text-[#61554E] font-medium">Model Classification</span>
                        <span
                          className={`text-xs font-bold px-3 py-1 rounded-full ${
                            currentResult.predicted_class_id === 1
                              ? 'bg-rose-100 text-rose-800 border border-rose-300'
                              : 'bg-emerald-100 text-emerald-800 border border-emerald-300'
                          }`}
                        >
                          {currentResult.prediction || 'Unknown'}
                        </span>
                      </div>

                      {/* Confidence bar */}
                      <div className="space-y-1.5">
                        <div className="flex justify-between text-xs">
                          <span className="text-[#61554E]">Predicted Probability (Sigmoid Output)</span>
                          <span className="font-mono font-bold text-[#1A1512]">
                            {currentResult.confidence !== null && currentResult.confidence !== undefined
                              ? `${(currentResult.confidence * 100).toFixed(2)}%`
                              : 'N/A'}
                          </span>
                        </div>
                        <div className="h-2 w-full rounded-full bg-[#EAE3DD] overflow-hidden">
                          <div
                            className={`h-full transition-all duration-500 ${
                              (currentResult.confidence ?? 0) >= (currentResult.threshold || 0.5)
                                ? 'bg-gradient-to-r from-amber-500 to-[#014BAA]'
                                : 'bg-gradient-to-r from-teal-500 to-emerald-600'
                            }`}
                            style={{
                              width: `${Math.min(100, Math.max(0, (currentResult.confidence ?? 0) * 100))}%`,
                            }}
                          />
                        </div>
                        <p className="text-[10px] text-[#7C6E66]">
                          * Sigmoid probability relative to decision threshold {currentResult.threshold} (model score, not definitive clinical diagnosis).
                        </p>
                      </div>

                      {/* Complete Response Fields Metadata */}
                      <div className="space-y-1.5 text-[11px] text-[#61554E] pt-2 border-t border-[#EAE3DD]">
                        <div className="flex justify-between items-center">
                          <span className="text-[#7C6E66]">Inference Model:</span>
                          <span className="font-mono font-semibold text-[#1A1512]">{currentResult.model_name}</span>
                        </div>
                        <div className="flex justify-between items-center">
                          <span className="text-[#7C6E66]">Processed File:</span>
                          <span className="font-mono text-[#1A1512] truncate max-w-[200px]" title={currentResult.filename}>
                            {currentResult.filename} ({currentResult.file_type})
                          </span>
                        </div>
                        <div className="flex justify-between items-center">
                          <span className="text-[#7C6E66]">Decision Threshold:</span>
                          <span className="font-mono text-[#1A1512]">{currentResult.threshold}</span>
                        </div>
                        {currentResult.message && (
                          <div className="flex justify-between items-center">
                            <span className="text-[#7C6E66]">Execution Status:</span>
                            <span className="font-mono text-emerald-700 font-medium truncate max-w-[200px]" title={currentResult.message}>
                              {currentResult.message}
                            </span>
                          </div>
                        )}
                        {isViewingRestoredCase && activeCase && (
                          <div className="flex justify-between items-center pt-1 border-t border-[#EAE3DD]/60">
                            <span className="text-[#7C6E66]">Timestamp:</span>
                            <span className="font-mono text-[10px] text-[#61554E]">
                              {new Date(activeCase.timestamp).toLocaleString()}
                            </span>
                          </div>
                        )}
                      </div>
                    </div>

                    {/* Explainable AI Grad-CAM Card */}
                    {activeGradcam && (
                      <div className="p-4 rounded-xl bg-[#014BAA]/5 border border-[#014BAA]/20 space-y-2.5">
                        <div className="flex items-center justify-between">
                          <span className="text-xs font-bold text-[#014BAA] flex items-center gap-1.5">
                            <Eye className="w-3.5 h-3.5" />
                            Explainable AI (Grad-CAM Saliency)
                          </span>
                          <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-[#014BAA]/10 text-[#014BAA] font-bold">
                            LAYER4 ACTIVE
                          </span>
                        </div>
                        <p className="text-[11px] text-[#61554E] leading-relaxed">
                          Gradient flow hooked onto ResNet-50 <code className="font-mono text-[#014BAA] bg-white px-1 py-0.5 rounded border border-[#014BAA]/20">layer4</code>. Warm red/yellow zones represent the anatomical pixels most heavily influencing this prediction.
                        </p>
                        <button
                          type="button"
                          onClick={() => setViewMode(viewMode === 'gradcam' ? 'original' : 'gradcam')}
                          className="w-full py-2 px-3 rounded-lg bg-white border border-[#014BAA]/30 text-[#014BAA] text-xs font-bold hover:bg-[#014BAA] hover:text-white transition-all shadow-xs flex items-center justify-center gap-1.5"
                        >
                          <Eye className="w-3.5 h-3.5" />
                          <span>{viewMode === 'gradcam' ? 'Show Original Radiograph' : 'Inspect Grad-CAM Heatmap'}</span>
                        </button>
                      </div>
                    )}

                    {/* Download Clinical PDF Report Action */}
                    <div className="pt-1">
                      <button
                        type="button"
                        onClick={handleDownloadReport}
                        disabled={downloadingReport || !currentResult}
                        className={`w-full py-2.5 px-4 rounded-xl text-xs font-bold transition-all shadow-xs flex items-center justify-center gap-2 ${
                          downloadingReport || !currentResult
                            ? 'bg-[#EAE3DD] text-[#7C6E66] cursor-not-allowed'
                            : 'bg-[#014BAA] hover:bg-[#01357A] text-[#F8F3F0] active:scale-95 shadow-md shadow-[#014BAA]/20'
                        }`}
                        title="Download comprehensive official PDF clinical diagnostic report"
                      >
                        {downloadingReport ? (
                          <>
                            <RefreshCw className="w-4 h-4 animate-spin" />
                            <span>Synthesizing PDF Report...</span>
                          </>
                        ) : (
                          <>
                            <Download className="w-4 h-4" />
                            <span>Download Official Report (PDF)</span>
                          </>
                        )}
                      </button>
                      {reportError && (
                        <p className="mt-1.5 text-[11px] text-rose-600 font-medium text-center">{reportError}</p>
                      )}
                    </div>

                    {/* Telemetry info */}
                    <div className="p-3.5 rounded-xl bg-[#FAF7F5] border border-[#EAE3DD] text-xs">
                      <p className="text-[11px] font-bold text-[#014BAA]">Clinical Safeguard Notice</p>
                      <p className="text-[11px] text-[#61554E] mt-1 italic">
                        {currentResult.disclaimer}
                      </p>
                    </div>
                  </div>
                ) : null}
              </div>
            )}

            {/* State 3: Empty state when no analysis performed and no restored case */}
            {!currentResult && !analysisError && !analyzing && (
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
