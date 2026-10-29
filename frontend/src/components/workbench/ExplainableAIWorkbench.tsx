import React, { useState, useRef } from 'react';
import {
  Eye,
  Layers,
  Upload,
  CheckCircle2,
  RefreshCw,
  FileText,
  ArrowRight,
  Download,
  AlertCircle,
  RotateCcw,
} from 'lucide-react';
import type { GradCAMResponse } from '../../types';
import { generateGradCAM, downloadAnalysisReport } from '../../services/api';
import { useAnalysis } from '../../context/useAnalysis';

interface ExplainableAIWorkbenchProps {
  onNavigateToChestXRay?: () => void;
}

export const ExplainableAIWorkbench: React.FC<ExplainableAIWorkbenchProps> = ({
  onNavigateToChestXRay,
}) => {
  const { activeCase, saveActiveCase, updateViewMode, clearActiveCase } = useAnalysis();

  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [localPreviewUrl, setLocalPreviewUrl] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState<boolean>(false);
  const [analyzing, setAnalyzing] = useState<boolean>(false);
  const [downloadingReport, setDownloadingReport] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [reportError, setReportError] = useState<string | null>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);

  // Active view mode from persisted case or local fallback
  const viewTab = activeCase?.viewMode || 'overlay';

  // Resolved active image assets (persisted case takes precedence when returning from other tabs)
  const activeFilename = selectedFile?.name || activeCase?.filename || null;
  const activePreview = localPreviewUrl || activeCase?.imagePreview || null;
  const activeGradcam = activeCase?.gradcamPreview || null;
  const activeHeatmap = activeCase?.heatmapPreview || null;
  const activePrediction = activeCase?.prediction || null;
  const activeConfidence = activeCase?.confidence ?? null;
  const activeExplanation = activeCase?.explanation || null;
  const activeRecordId = activeCase?.recordId || null;

  const hasImage = Boolean(activePreview);
  const hasAttribution = Boolean(activeGradcam || activeHeatmap);

  const handleFileChange = (file: File) => {
    setSelectedFile(file);
    setError(null);
    setReportError(null);
    if (localPreviewUrl && localPreviewUrl.startsWith('blob:')) {
      URL.revokeObjectURL(localPreviewUrl);
    }
    const url = URL.createObjectURL(file);
    setLocalPreviewUrl(url);
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

  const handleGenerateGradCAM = async () => {
    if (!selectedFile && !activePreview) {
      setError('Please select or upload a chest radiograph first.');
      return;
    }

    setAnalyzing(true);
    setError(null);
    setReportError(null);

    try {
      let fileToUpload: File | null = selectedFile;

      // If restoring from persisted preview without an active File object, convert base64 to File
      if (!fileToUpload && activePreview) {
        if (activePreview.startsWith('data:')) {
          const res = await fetch(activePreview);
          const blob = await res.blob();
          fileToUpload = new File([blob], activeFilename || 'restored_xray.jpg', { type: 'image/jpeg' });
        }
      }

      if (!fileToUpload) {
        throw new Error('Please select an image file to compute Grad-CAM attribution.');
      }

      const res = await generateGradCAM(fileToUpload);
      if (res.data && res.data.status === 'success') {
        const data: GradCAMResponse = res.data;
        // Persist to central AnalysisContext and IndexedDB so it survives navigation and refresh
        await saveActiveCase({
          recordId: data.id || activeRecordId || null,
          filename: fileToUpload.name,
          fileType: fileToUpload.name.toLowerCase().endsWith('.dcm') ? 'dicom' : 'standard_image',
          isDicom: fileToUpload.name.toLowerCase().endsWith('.dcm'),
          imagePreview: data.original_preview || activePreview,
          gradcamPreview: data.overlay_preview || null,
          heatmapPreview: data.heatmap_preview || null,
          prediction: data.prediction || 'Classified',
          confidence: data.confidence ?? null,
          threshold: data.threshold ?? 0.5,
          modelName: 'Chest X-ray ResNet50 Classifier (Grad-CAM)',
          status: 'success',
          explanation: data.explanation || undefined,
          disclaimer: data.disclaimer,
          timestamp: new Date().toISOString(),
          viewMode: 'overlay',
          modality: 'chest_xray',
        });
      } else {
        setError(res.error || res.data?.message || 'Failed to generate Grad-CAM attribution.');
      }
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Unknown network error');
    } finally {
      setAnalyzing(false);
    }
  };

  const handleDownloadReport = async () => {
    if (!hasAttribution && !activePreview) {
      setReportError('Please select a radiograph and generate Grad-CAM attribution first.');
      return;
    }

    setDownloadingReport(true);
    setReportError(null);
    try {
      const payload: Record<string, unknown> = {
        id: activeRecordId,
        modality: 'chest_xray',
        model_name: activeCase?.modelName || 'Chest X-ray ResNet50 Classifier (Grad-CAM XAI)',
        filename: activeFilename || 'chest_radiograph.png',
        file_type: activeFilename?.toLowerCase().endsWith('.dcm') ? 'dicom' : 'standard_image',
        prediction: activePrediction || 'Normal / Opacity Evaluated',
        confidence: activeConfidence,
        threshold: activeCase?.threshold || 0.5,
        status: 'success',
        explanation: activeExplanation,
        disclaimer: activeCase?.disclaimer,
        timestamp: activeCase?.timestamp || new Date().toISOString(),
        image_preview: activePreview,
        gradcam_preview: activeGradcam,
        heatmap_preview: activeHeatmap,
        view_mode: viewTab,
      };
      const filename = `MediVision_GradCAM_Report_${activeRecordId || activeFilename || 'case'}.pdf`;
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

  const handleReset = async () => {
    setSelectedFile(null);
    if (localPreviewUrl && localPreviewUrl.startsWith('blob:')) {
      URL.revokeObjectURL(localPreviewUrl);
    }
    setLocalPreviewUrl(null);
    setError(null);
    setReportError(null);
    await clearActiveCase();
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  return (
    <div className="space-y-6">
      {/* Module Operational Status Banner */}
      <div className="p-4 rounded-2xl bg-emerald-50/90 border border-emerald-300 text-emerald-950 flex flex-col md:flex-row md:items-center justify-between gap-3 shadow-xs">
        <div className="flex items-center gap-3">
          <div className="p-2 rounded-xl bg-emerald-600 text-white shrink-0">
            <CheckCircle2 className="w-5 h-5" />
          </div>
          <div>
            <h4 className="font-bold text-emerald-900 text-sm">
              Module 5 Status: GRAD-CAM OPERATIONAL • ResNet-50 Layer4 Hook Active
            </h4>
            <p className="text-xs text-emerald-800">
              Coupled with the trained <strong>ResNet-50 Chest Radiograph Classifier</strong>. Backward gradient flow is captured on <code className="bg-emerald-100 font-mono text-[11px] px-1 rounded">layer4</code> to calculate spatial attribution heatmaps.
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2 self-start md:self-center">
          {hasImage && (
            <button
              onClick={handleReset}
              className="px-3 py-2 rounded-xl bg-white hover:bg-rose-50 text-rose-700 hover:text-rose-900 border border-[#EAE3DD] text-xs font-bold transition-all shadow-2xs flex items-center gap-1.5"
              title="Clear current radiograph and reset visualizer"
            >
              <RotateCcw className="w-3.5 h-3.5" />
              <span>Reset</span>
            </button>
          )}

          {onNavigateToChestXRay && (
            <button
              onClick={onNavigateToChestXRay}
              className="px-4 py-2 rounded-xl bg-[#014BAA] hover:bg-[#01357A] text-white text-xs font-bold transition-all shadow-xs flex items-center gap-1.5"
            >
              <span>Open in Chest X-ray</span>
              <ArrowRight className="w-3.5 h-3.5" />
            </button>
          )}
        </div>
      </div>

      {/* Main Grid: Upload & Visualizer */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Column: Image Ingestion (6 Cols) */}
        <div className="lg:col-span-6 space-y-4">
          <div className="bg-white/95 border border-[#EAE3DD] rounded-2xl p-6 space-y-4 shadow-sm backdrop-blur-sm">
            <div className="flex items-center justify-between">
              <h3 className="text-sm font-bold text-[#1A1512] flex items-center gap-2">
                <Upload className="w-4 h-4 text-[#014BAA]" />
                Chest Radiograph Ingestion for Saliency
              </h3>
              <span className="text-[11px] text-[#61554E]">PNG, JPEG, DICOM</span>
            </div>

            {/* Drag & Drop Area */}
            <div
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
              onClick={() => fileInputRef.current?.click()}
              className={`border-2 border-dashed rounded-2xl p-8 text-center cursor-pointer transition-all duration-200 flex flex-col items-center justify-center min-h-[190px] ${
                isDragging
                  ? 'border-[#014BAA] bg-[#014BAA]/5'
                  : 'border-[#DFD7D0] hover:border-[#014BAA] bg-[#FAF7F5]'
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

              {activeFilename ? (
                <div className="space-y-2 text-center">
                  <div className="w-10 h-10 rounded-xl bg-[#014BAA]/10 text-[#014BAA] flex items-center justify-center mx-auto">
                    <FileText className="w-5 h-5" />
                  </div>
                  <p className="text-xs font-bold text-[#1A1512] truncate max-w-xs mx-auto">{activeFilename}</p>
                  {selectedFile && (
                    <p className="text-[11px] text-[#61554E]">{(selectedFile.size / 1024).toFixed(1)} KB</p>
                  )}
                  {activeCase && !selectedFile && (
                    <p className="text-[10px] text-[#014BAA] font-bold">Persisted Radiograph Active</p>
                  )}
                  <p className="text-[10px] text-[#61554E]">Click or drag another image to replace</p>
                </div>
              ) : (
                <div className="space-y-2 text-center">
                  <div className="w-10 h-10 rounded-xl bg-[#014BAA]/10 text-[#014BAA] flex items-center justify-center mx-auto">
                    <Upload className="w-5 h-5" />
                  </div>
                  <p className="text-xs font-bold text-[#1A1512]">
                    Select radiograph or <span className="text-[#014BAA] underline">browse files</span>
                  </p>
                  <p className="text-[11px] text-[#61554E]">Input will be evaluated through the trained ResNet-50</p>
                </div>
              )}
            </div>

            {/* Action Bar */}
            <div className="flex flex-col sm:flex-row items-stretch sm:items-center justify-between gap-3 pt-2">
              <span className="text-[11px] text-[#61554E]">
                {hasImage ? 'Radiograph ready for layer4 gradient extraction' : 'No image loaded'}
              </span>

              <button
                type="button"
                onClick={handleGenerateGradCAM}
                disabled={!hasImage || analyzing}
                className={`px-5 py-2.5 rounded-xl text-xs font-bold transition-all shadow-xs flex items-center justify-center gap-2 ${
                  !hasImage || analyzing
                    ? 'bg-[#EAE3DD] text-[#7C6E66] cursor-not-allowed'
                    : 'bg-[#014BAA] hover:bg-[#01357A] text-white shadow-md active:scale-95'
                }`}
              >
                {analyzing ? (
                  <>
                    <RefreshCw className="w-4 h-4 animate-spin" />
                    <span>Extracting Gradients...</span>
                  </>
                ) : (
                  <>
                    <Eye className="w-4 h-4" />
                    <span>Compute Grad-CAM Saliency</span>
                  </>
                )}
              </button>
            </div>
          </div>

          {/* Mathematical Formulation Explainer */}
          <div className="bg-white/95 border border-[#EAE3DD] rounded-2xl p-6 space-y-3 shadow-sm text-xs">
            <h4 className="font-bold text-[#1A1512] flex items-center gap-2">
              <Layers className="w-4 h-4 text-[#014BAA]" />
              Grad-CAM Mathematical Formulation
            </h4>
            <div className="p-3 rounded-xl bg-[#FAF7F5] border border-[#EAE3DD] font-mono text-[11px] text-[#014BAA] space-y-1">
              <p>1. Importance Weight: α_k^c = (1/Z) Σ_i Σ_j (∂y^c / ∂A_[i,j]^k)</p>
              <p>2. Saliency Heatmap: L_Grad-CAM = ReLU( Σ_k α_k^c A^k )</p>
            </div>
            <p className="text-[11px] text-[#61554E] leading-relaxed">
              Global average pooling across feature map gradients yields channel weights α_k. The rectified linear unit (ReLU) ensures only features that positively influence the predicted pathology are displayed.
            </p>
          </div>
        </div>

        {/* Right Column: Visual Saliency Attribution Display (6 Cols) */}
        <div className="lg:col-span-6 space-y-4">
          <div className="bg-white/95 border border-[#EAE3DD] rounded-2xl p-6 space-y-4 shadow-sm backdrop-blur-sm">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
              <div className="flex items-center gap-2">
                <h3 className="text-sm font-bold text-[#1A1512] flex items-center gap-2">
                  <Eye className="w-4 h-4 text-[#014BAA]" />
                  Visual Attribution Saliency
                </h3>
                {activeRecordId && (
                  <span className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-emerald-100 text-emerald-800 font-bold">
                    PERSISTED
                  </span>
                )}
              </div>

              {/* View Switcher */}
              {hasAttribution && (
                <div className="flex items-center gap-1 bg-[#FAF7F5] p-1 rounded-xl border border-[#EAE3DD]">
                  <button
                    type="button"
                    onClick={() => updateViewMode('overlay')}
                    className={`px-2.5 py-1 rounded-lg text-[10px] font-bold transition-all ${
                      viewTab === 'overlay' ? 'bg-[#014BAA] text-white shadow-xs' : 'text-[#61554E]'
                    }`}
                  >
                    Overlay
                  </button>
                  <button
                    type="button"
                    onClick={() => updateViewMode('heatmap')}
                    className={`px-2.5 py-1 rounded-lg text-[10px] font-bold transition-all ${
                      viewTab === 'heatmap' ? 'bg-[#014BAA] text-white shadow-xs' : 'text-[#61554E]'
                    }`}
                  >
                    Heatmap
                  </button>
                  <button
                    type="button"
                    onClick={() => updateViewMode('original')}
                    className={`px-2.5 py-1 rounded-lg text-[10px] font-bold transition-all ${
                      viewTab === 'original' ? 'bg-[#014BAA] text-white shadow-xs' : 'text-[#61554E]'
                    }`}
                  >
                    Original
                  </button>
                  <button
                    type="button"
                    onClick={() => updateViewMode('split')}
                    className={`px-2.5 py-1 rounded-lg text-[10px] font-bold transition-all ${
                      viewTab === 'split' ? 'bg-[#014BAA] text-white shadow-xs' : 'text-[#61554E]'
                    }`}
                  >
                    Side-by-Side
                  </button>
                </div>
              )}
            </div>

            {error && (
              <div className="p-4 rounded-xl bg-rose-50 border border-rose-300 text-rose-900 text-xs flex items-center gap-2">
                <AlertCircle className="w-4 h-4 shrink-0 text-rose-600" />
                <span>{error}</span>
              </div>
            )}

            {hasAttribution && activePreview ? (
              <div className="space-y-4">
                {/* Visualizer Display Box with Exact Matching Dimensions */}
                {viewTab === 'split' ? (
                  <div className="grid grid-cols-2 gap-3">
                    <div className="space-y-1">
                      <p className="text-[10px] font-bold uppercase tracking-wider text-[#7C6E66] text-center">Original Radiograph</p>
                      <div className="relative aspect-square max-h-[300px] w-full rounded-xl overflow-hidden bg-[#0C060A] flex items-center justify-center border border-[#DFD7D0]">
                        <img src={activePreview} alt="Original" className="w-full h-full object-contain" />
                      </div>
                    </div>
                    <div className="space-y-1">
                      <p className="text-[10px] font-bold uppercase tracking-wider text-[#014BAA] text-center">Grad-CAM Overlay (layer4)</p>
                      <div className="relative aspect-square max-h-[300px] w-full rounded-xl overflow-hidden bg-[#0C060A] flex items-center justify-center border border-[#014BAA]/30">
                        <img src={activeGradcam || activeHeatmap || ''} alt="Grad-CAM Overlay" className="w-full h-full object-contain" />
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="relative aspect-square max-h-[320px] w-full mx-auto rounded-xl overflow-hidden bg-[#0C060A] flex items-center justify-center border border-[#DFD7D0]">
                    <img
                      src={
                        viewTab === 'heatmap'
                          ? activeHeatmap || activeGradcam || ''
                          : viewTab === 'original'
                          ? activePreview
                          : activeGradcam || activePreview
                      }
                      alt="Grad-CAM Visualization"
                      className="w-full h-full object-contain"
                    />
                    <div className="absolute top-2 left-2 px-2.5 py-1 rounded-lg bg-black/75 backdrop-blur-xs text-[10px] text-amber-300 font-mono font-bold border border-amber-400/40">
                      {viewTab === 'heatmap'
                        ? 'Standalone Jet Saliency Heatmap'
                        : viewTab === 'original'
                        ? 'Original Radiograph'
                        : 'ResNet-50 layer4 Grad-CAM Blend (α=0.45)'}
                    </div>
                  </div>
                )}

                {/* Explanation & Inference Card */}
                <div className="p-4 rounded-xl bg-[#FAF7F5] border border-[#EAE3DD] space-y-2 text-xs">
                  <div className="flex items-center justify-between">
                    <span className="font-bold text-[#1A1512]">Classification Attribution</span>
                    {activePrediction && (
                      <span className="font-mono text-[11px] font-bold text-[#014BAA]">
                        {activePrediction} ({activeConfidence !== null ? `${(activeConfidence * 100).toFixed(1)}%` : 'N/A'})
                      </span>
                    )}
                  </div>
                  <p className="text-[11px] text-[#61554E] leading-relaxed">
                    {activeExplanation ||
                      "Warm red and yellow regions highlight localized anatomical zones driving the ResNet-50 prediction. Cooler blue regions indicate neutral background or normal parenchymal tissue."}
                  </p>
                </div>

                {/* Download Report Button & Action Section */}
                <div className="pt-1 flex flex-col sm:flex-row items-stretch sm:items-center justify-between gap-2.5">
                  <button
                    type="button"
                    onClick={handleDownloadReport}
                    disabled={downloadingReport || (!hasAttribution && !activePreview)}
                    className={`flex-1 py-2.5 px-4 rounded-xl text-xs font-bold transition-all shadow-xs flex items-center justify-center gap-2 ${
                      downloadingReport || (!hasAttribution && !activePreview)
                        ? 'bg-[#EAE3DD] text-[#7C6E66] cursor-not-allowed'
                        : 'bg-[#014BAA] hover:bg-[#01357A] text-[#F8F3F0] active:scale-95 shadow-md shadow-[#014BAA]/20'
                    }`}
                    title="Download formal clinical diagnostic report with X-ray and Grad-CAM in PDF"
                  >
                    {downloadingReport ? (
                      <>
                        <RefreshCw className="w-4 h-4 animate-spin" />
                        <span>Synthesizing PDF Report...</span>
                      </>
                    ) : (
                      <>
                        <Download className="w-4 h-4" />
                        <span>Download Report (PDF)</span>
                      </>
                    )}
                  </button>

                  <button
                    type="button"
                    onClick={handleReset}
                    className="px-4 py-2.5 rounded-xl border border-[#DFD7D0] hover:border-rose-400 text-xs font-bold text-[#61554E] hover:text-rose-700 bg-white transition-all shadow-2xs"
                  >
                    Clear Case
                  </button>
                </div>

                {reportError && (
                  <p className="text-[11px] text-rose-600 font-medium text-center">{reportError}</p>
                )}

                {/* Regulatory Safeguard Notice */}
                <div className="p-3.5 rounded-xl bg-amber-50/70 border border-amber-200 text-[11px] text-amber-900 italic">
                  Explainable AI visual saliency maps are research visualizations of CNN gradient flow and do not constitute an anatomical or radiological boundary diagnosis.
                </div>
              </div>
            ) : hasImage ? (
              <div className="py-12 text-center space-y-3 border border-dashed border-[#DFD7D0] rounded-xl bg-[#FAF7F5]">
                <div className="relative aspect-square max-h-[220px] w-full max-w-[220px] mx-auto rounded-xl overflow-hidden bg-black border border-[#DFD7D0]">
                  <img src={activePreview || ''} alt="Loaded radiograph" className="w-full h-full object-contain" />
                </div>
                <p className="text-xs font-bold text-[#1A1512]">Radiograph Loaded</p>
                <p className="text-[11px] text-[#61554E] max-w-xs mx-auto">
                  Click 'Compute Grad-CAM Saliency' to extract convolutional feature gradients from ResNet-50 layer4.
                </p>
              </div>
            ) : (
              <div className="py-16 text-center space-y-2 border border-dashed border-[#DFD7D0] rounded-xl bg-[#FAF7F5]">
                <Eye className="h-8 w-8 text-[#014BAA] mx-auto opacity-70" />
                <p className="text-xs font-bold text-[#1A1512]">Grad-CAM Visualizer Ready</p>
                <p className="text-[11px] text-[#61554E] max-w-xs mx-auto">
                  Upload an image on the left and click 'Compute Grad-CAM Saliency' to visualize CNN feature attribution.
                </p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
