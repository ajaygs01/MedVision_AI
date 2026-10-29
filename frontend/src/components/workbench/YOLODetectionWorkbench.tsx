import React, { useState, useEffect, useRef } from 'react';
import {
  Crosshair,
  UploadCloud,
  Sliders,
  AlertCircle,
  CheckCircle,
  Layers,
  FileText,
  RefreshCw,
  ShieldAlert,
  RotateCcw,
} from 'lucide-react';
import { fetchYOLOStatus, detectYOLO } from '../../services/api';
import type { YOLODetectionResponse, YOLODetectionStatusResponse } from '../../types';
import { useAnalysis } from '../../context/useAnalysis';

export const YOLODetectionWorkbench: React.FC = () => {
  const { activeYOLOCase, saveActiveYOLOCase, clearActiveYOLOCase } = useAnalysis();

  const [statusInfo, setStatusInfo] = useState<YOLODetectionStatusResponse | null>(null);
  const [loadingStatus, setLoadingStatus] = useState<boolean>(true);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [filePreview, setFilePreview] = useState<string | null>(activeYOLOCase?.imagePreview || null);
  const [confThreshold, setConfThreshold] = useState<number>(activeYOLOCase?.confThreshold ?? 0.25);
  const [iouThreshold, setIouThreshold] = useState<number>(activeYOLOCase?.iouThreshold ?? 0.45);
  const [isInferring, setIsInferring] = useState<boolean>(false);
  const [result, setResult] = useState<YOLODetectionResponse | null>(activeYOLOCase?.result || null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [statusError, setStatusError] = useState<string | null>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);

  const loadStatus = async () => {
    setLoadingStatus(true);
    setStatusError(null);
    const res = await fetchYOLOStatus();
    if (res.data) {
      setStatusInfo(res.data);
      setStatusError(null);
    } else {
      setStatusInfo(null);
      setStatusError(res.error || 'Backend not responding');
    }
    setLoadingStatus(false);
  };

  useEffect(() => {
    loadStatus();
  }, []);

  // Synchronize state when activeYOLOCase changes (e.g. reopened from Analysis History)
  useEffect(() => {
    if (activeYOLOCase) {
      if (activeYOLOCase.imagePreview) {
        setFilePreview(activeYOLOCase.imagePreview);
      }
      setResult(activeYOLOCase.result || null);
      if (activeYOLOCase.confThreshold !== undefined) {
        setConfThreshold(activeYOLOCase.confThreshold);
      }
      if (activeYOLOCase.iouThreshold !== undefined) {
        setIouThreshold(activeYOLOCase.iouThreshold);
      }
    }
  }, [activeYOLOCase]);

  const activeFilename = selectedFile?.name || activeYOLOCase?.filename || null;
  const activePreview = result?.preview_image || filePreview || activeYOLOCase?.imagePreview || null;
  const isDicomFile = activeFilename?.toLowerCase().endsWith('.dcm');

  const highestCandidateConfidence: number | null =
    result && typeof (result as any)?.max_proposal_confidence === 'number'
      ? (result as any).max_proposal_confidence
      : result && typeof (result as any)?.highest_candidate_confidence === 'number'
      ? (result as any).highest_candidate_confidence
      : result && typeof (result as any)?.peak_confidence === 'number'
      ? (result as any).peak_confidence
      : null;

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    // Reset results and errors immediately so old results are never paired with a new file
    setSelectedFile(file);
    setResult(null);
    setErrorMsg(null);

    let previewUrl: string | null = null;
    if (!file.name.toLowerCase().endsWith('.dcm')) {
      previewUrl = URL.createObjectURL(file);
      setFilePreview(previewUrl);
    } else {
      setFilePreview(null);
    }

    // Persist active case without previous results (prevents stale results for new file)
    saveActiveYOLOCase({
      filename: file.name,
      imagePreview: previewUrl,
      result: null,
      confThreshold,
      iouThreshold,
      timestamp: new Date().toISOString(),
    });
  };

  const handleResetCase = () => {
    setSelectedFile(null);
    setFilePreview(null);
    setResult(null);
    setErrorMsg(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
    clearActiveYOLOCase();
  };

  const handleConfChange = (val: number) => {
    setConfThreshold(val);
    if (activeYOLOCase) {
      saveActiveYOLOCase({
        ...activeYOLOCase,
        confThreshold: val,
      });
    }
  };

  const handleIouChange = (val: number) => {
    setIouThreshold(val);
    if (activeYOLOCase) {
      saveActiveYOLOCase({
        ...activeYOLOCase,
        iouThreshold: val,
      });
    }
  };

  const handleRunDetection = async () => {
    if (!selectedFile) {
      if (activeYOLOCase?.filename) {
        setErrorMsg('Please re-select the original radiograph file to re-run localization with updated thresholds.');
      } else {
        setErrorMsg('Please select a chest radiograph image or DICOM file.');
      }
      return;
    }

    setIsInferring(true);
    setErrorMsg(null);

    const res = await detectYOLO(selectedFile, confThreshold, iouThreshold);
    if (res.error) {
      setErrorMsg(res.error);
    } else if (res.data) {
      setResult(res.data);
      const finalPreview = res.data.preview_image || filePreview;
      saveActiveYOLOCase({
        recordId: res.data.id || res.data.analysis_id || null,
        filename: selectedFile.name,
        imagePreview: finalPreview,
        result: res.data,
        confThreshold,
        iouThreshold,
        timestamp: new Date().toISOString(),
      });
    }

    setIsInferring(false);
  };

  const isOffline = statusError !== null || (!loadingStatus && !statusInfo);
  const isModelTrained = !isOffline && (statusInfo?.trained ?? false);

  return (
    <div className="space-y-6">
      {/* Module Overview & Status Banner */}
      <div className="rounded-2xl border border-[#EAE3DD] bg-white/95 p-6 shadow-sm">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div className="flex items-start gap-3.5">
            <div className="p-3 rounded-xl bg-[#014BAA]/10 text-[#014BAA] border border-[#014BAA]/20 shrink-0">
              <Crosshair className="h-6 w-6" />
            </div>
            <div>
              <div className="flex items-center gap-2.5 flex-wrap">
                <h3 className="text-base font-black text-[#1A1512] tracking-tight">
                  YOLOv8 Anatomical Lesion Localization
                </h3>
                <span
                  className={`inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-[11px] font-bold uppercase tracking-wider ${
                    isOffline
                      ? 'bg-rose-50 text-rose-800 border border-rose-200'
                      : isModelTrained
                      ? 'bg-emerald-50 text-emerald-800 border border-emerald-200'
                      : 'bg-amber-50 text-amber-900 border border-amber-300'
                  }`}
                >
                  <span
                    className={`h-1.5 w-1.5 rounded-full ${
                      isOffline
                        ? 'bg-rose-500'
                        : isModelTrained
                        ? 'bg-emerald-500'
                        : 'bg-amber-500'
                    }`}
                  />
                  {isOffline
                    ? 'BACKEND OFFLINE • RECONNECTING'
                    : isModelTrained
                    ? 'CHECKPOINT LOADED • RESEARCH PROTOTYPE'
                    : 'MODEL NOT TRAINED • WEIGHTS PENDING'}
                </span>
              </div>
              <p className="mt-1 text-xs text-[#61554E] leading-relaxed">
                Single-stage spatial bounding box localization for lung opacities and pulmonary findings using Ultralytics YOLOv8 architecture.
              </p>
            </div>
          </div>

          <button
            onClick={loadStatus}
            disabled={loadingStatus}
            className="self-start md:self-center inline-flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-[#F8F3F0] hover:bg-[#EAE3DD] text-[#1A1512] text-xs font-semibold border border-[#EAE3DD] transition-all disabled:opacity-50"
            title="Refresh module status"
          >
            <RefreshCw className={`h-3.5 w-3.5 ${loadingStatus ? 'animate-spin text-[#014BAA]' : ''}`} />
            <span>Check Status</span>
          </button>
        </div>

        {/* Technical specs pill row */}
        <div className="mt-4 pt-4 border-t border-[#EAE3DD] grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs">
          <div className="p-2.5 rounded-xl bg-[#FAF7F5] border border-[#EAE3DD]">
            <span className="text-[#61554E] text-[10px] uppercase font-bold block">Architecture</span>
            <span className="font-mono text-[#014BAA] font-bold text-xs">
              {statusInfo?.model_architecture || 'YOLOv8n'}
            </span>
          </div>
          <div className="p-2.5 rounded-xl bg-[#FAF7F5] border border-[#EAE3DD]">
            <span className="text-[#61554E] text-[10px] uppercase font-bold block">Detectable Classes</span>
            <span className="font-mono text-[#1A1512] font-semibold text-xs">
              {statusInfo?.classes?.join(', ') || 'lung_opacity'}
            </span>
          </div>
          <div className="p-2.5 rounded-xl bg-[#FAF7F5] border border-[#EAE3DD]">
            <span className="text-[#61554E] text-[10px] uppercase font-bold block">Checkpoint</span>
            <span className="font-mono text-[11px] truncate block text-[#61554E]" title={statusInfo?.checkpoint_path || 'None'}>
              {statusInfo?.checkpoint_path ? 'yolov8_best.pt' : isOffline ? 'Backend Offline' : 'None (Untrained)'}
            </span>
          </div>
          <div className="p-2.5 rounded-xl bg-[#FAF7F5] border border-[#EAE3DD]">
            <span className="text-[#61554E] text-[10px] uppercase font-bold block">Inference Device</span>
            <span className="font-mono text-emerald-800 font-bold text-xs">CPU Native (12 Threads)</span>
          </div>
        </div>
      </div>

      {/* Untrained Safeguard Notice */}
      {!isModelTrained && !isOffline && (
        <div className="rounded-2xl border border-amber-300 bg-amber-50/90 p-4 shadow-xs">
          <div className="flex items-start gap-3">
            <AlertCircle className="h-5 w-5 text-amber-600 shrink-0 mt-0.5" />
            <div className="flex-1 text-xs">
              <h4 className="font-bold text-amber-900">
                Untrained Architecture Mode — Zero Hallucination Guarantee
              </h4>
              <p className="mt-0.5 text-amber-800 leading-relaxed">
                In strict compliance with medical validation standards, no mock bounding boxes or fabricated detections are returned when model weights are unconfigured. You may still upload radiographs to verify image decoding, dimensions, and API connectivity.
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Backend Offline Notice */}
      {isOffline && (
        <div className="rounded-2xl border border-rose-300 bg-rose-50/90 p-4 shadow-xs">
          <div className="flex items-start gap-3">
            <AlertCircle className="h-5 w-5 text-rose-600 shrink-0 mt-0.5" />
            <div className="flex-1 text-xs">
              <h4 className="font-bold text-rose-900">
                Backend Service Offline
              </h4>
              <p className="mt-0.5 text-rose-800 leading-relaxed">
                Unable to reach FastAPI service at <code className="font-mono text-rose-900 font-bold">/api/yolo/status</code>. Bounding box localization requires the backend service to be running.
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Main Workbench Grid: Controls & Visualizer */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Column: Input, Controls, Thresholds (5 cols) */}
        <div className="lg:col-span-5 space-y-4">
          <div className="rounded-2xl border border-[#EAE3DD] bg-white/95 p-5 shadow-sm space-y-5">
            <div className="flex items-center justify-between pb-3 border-b border-[#EAE3DD]">
              <div className="flex items-center gap-2">
                <UploadCloud className="h-4 w-4 text-[#014BAA]" />
                <h4 className="text-sm font-bold text-[#1A1512]">Input Radiograph</h4>
              </div>
              {(selectedFile || activeYOLOCase) && (
                <button
                  type="button"
                  onClick={handleResetCase}
                  className="inline-flex items-center gap-1 px-2 py-1 rounded-lg text-[11px] font-semibold text-[#61554E] hover:text-[#1A1512] hover:bg-[#FAF7F5] border border-transparent hover:border-[#EAE3DD] transition-all"
                  title="Clear active radiograph and results"
                >
                  <RotateCcw className="h-3 w-3" />
                  <span>Reset</span>
                </button>
              )}
            </div>

            {/* Upload Area */}
            <div
              onClick={() => fileInputRef.current?.click()}
              className="p-6 rounded-2xl border-2 border-dashed border-[#DFD7D0] hover:border-[#014BAA] bg-[#FAF7F5] hover:bg-[#014BAA]/5 cursor-pointer text-center transition-all space-y-2"
            >
              <input
                ref={fileInputRef}
                type="file"
                accept=".dcm,.png,.jpg,.jpeg"
                onChange={handleFileChange}
                className="hidden"
              />
              <Crosshair className="h-8 w-8 text-[#014BAA] mx-auto" />
              <div>
                <p className="text-xs font-bold text-[#1A1512]">
                  {activeFilename ? activeFilename : 'Click or drop radiograph file'}
                </p>
                <p className="text-[11px] text-[#61554E] mt-0.5">
                  {selectedFile
                    ? 'Selected for local inference'
                    : activeFilename
                    ? 'Active case restored from session'
                    : 'Supported: DICOM (.dcm), PNG, JPEG (up to 25 MB)'}
                </p>
              </div>
              {selectedFile ? (
                <div className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full bg-emerald-50 border border-emerald-200 text-[10px] font-bold text-emerald-800">
                  <CheckCircle className="h-3 w-3 text-emerald-600" />
                  <span>{(selectedFile.size / 1024).toFixed(1)} KB Selected</span>
                </div>
              ) : activeFilename ? (
                <div className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full bg-blue-50 border border-blue-200 text-[10px] font-bold text-[#014BAA]">
                  <CheckCircle className="h-3 w-3 text-[#014BAA]" />
                  <span>Case Restored</span>
                </div>
              ) : null}
            </div>

            {/* Threshold Sliders */}
            <div className="space-y-4 pt-2">
              <div className="flex items-center gap-2 text-xs font-bold text-[#1A1512]">
                <Sliders className="h-3.5 w-3.5 text-[#014BAA]" />
                <span>Detection Thresholds</span>
              </div>

              <div>
                <div className="flex justify-between text-xs mb-1">
                  <span className="text-[#61554E] font-medium">Confidence Threshold:</span>
                  <span className="font-mono font-bold text-[#014BAA]">{(confThreshold * 100).toFixed(0)}%</span>
                </div>
                <input
                  type="range"
                  min="0.01"
                  max="0.95"
                  step="0.01"
                  value={confThreshold}
                  onChange={(e) => handleConfChange(parseFloat(e.target.value))}
                  className="w-full accent-[#014BAA] cursor-pointer"
                />
                <span className="text-[10px] text-[#61554E]">Minimum probability to report a finding.</span>
              </div>

              <div>
                <div className="flex justify-between text-xs mb-1">
                  <span className="text-[#61554E] font-medium">IoU NMS Threshold:</span>
                  <span className="font-mono font-bold text-[#014BAA]">{(iouThreshold * 100).toFixed(0)}%</span>
                </div>
                <input
                  type="range"
                  min="0.10"
                  max="0.90"
                  step="0.05"
                  value={iouThreshold}
                  onChange={(e) => handleIouChange(parseFloat(e.target.value))}
                  className="w-full accent-[#014BAA] cursor-pointer"
                />
                <span className="text-[10px] text-[#61554E]">Intersection-over-Union threshold for non-max suppression.</span>
              </div>
            </div>

            {/* Execute Button */}
            <button
              onClick={handleRunDetection}
              disabled={(!selectedFile && !activeFilename) || isInferring}
              className="w-full py-2.5 px-4 rounded-xl bg-[#014BAA] hover:bg-[#01357A] disabled:bg-[#DFD7D0] disabled:cursor-not-allowed text-white font-bold text-xs shadow-sm transition-all flex items-center justify-center gap-2"
            >
              {isInferring ? (
                <>
                  <RefreshCw className="h-4 w-4 animate-spin" />
                  <span>Processing Radiograph...</span>
                </>
              ) : (
                <>
                  <Crosshair className="h-4 w-4" />
                  <span>Run Anatomical Localization</span>
                </>
              )}
            </button>

            {errorMsg && (
              <div className="p-3 rounded-xl bg-red-50 border border-red-200 text-xs text-red-700 flex items-start gap-2">
                <AlertCircle className="h-4 w-4 text-red-600 shrink-0 mt-0.5" />
                <span>{errorMsg}</span>
              </div>
            )}
          </div>
        </div>

        {/* Right Column: Visualization & Findings (7 cols) */}
        <div className="lg:col-span-7 space-y-4">
          <div className="rounded-2xl border border-[#EAE3DD] bg-white/95 p-5 shadow-sm space-y-4">
            <div className="flex items-center justify-between pb-3 border-b border-[#EAE3DD]">
              <div className="flex items-center gap-2">
                <Layers className="h-4 w-4 text-[#014BAA]" />
                <h4 className="text-sm font-bold text-[#1A1512]">Localization Visualizer</h4>
              </div>

              {result && (
                <span
                  className={`text-[11px] font-bold px-2 py-0.5 rounded-full ${
                    result.status === 'success'
                      ? 'bg-emerald-50 text-emerald-800 border border-emerald-200'
                      : 'bg-amber-50 text-amber-900 border border-amber-300'
                  }`}
                >
                  {result.status === 'success' ? `${result.num_detections} Findings` : 'Untrained Mode'}
                </span>
              )}
            </div>

            {/* Visualizer Canvas Area */}
            <div className="relative aspect-square max-h-[460px] w-full rounded-xl bg-[#1A1512] flex items-center justify-center overflow-hidden border border-[#EAE3DD]">
              {activePreview ? (
                <img
                  src={activePreview}
                  alt="YOLO Detection Overlay"
                  className="max-h-full max-w-full object-contain"
                />
              ) : isDicomFile ? (
                <div className="text-center p-6 text-[#FAF7F5] space-y-2">
                  <FileText className="h-10 w-10 text-[#014BAA] mx-auto" />
                  <p className="text-xs font-bold">DICOM Radiograph Selected ({activeFilename})</p>
                  <p className="text-[11px] text-[#DFD7D0]">
                    Click &ldquo;Run Anatomical Localization&rdquo; to decode and display the pixel array.
                  </p>
                </div>
              ) : activeFilename && !activePreview ? (
                <div className="text-center p-6 text-[#FAF7F5] space-y-2">
                  <Crosshair className="h-10 w-10 text-[#61554E] mx-auto opacity-60" />
                  <p className="text-xs font-bold text-[#FAF7F5]">Study Restored: {activeFilename}</p>
                  <p className="text-[11px] text-[#DFD7D0]">
                    Re-select original image file to preview and re-execute localization.
                  </p>
                </div>
              ) : (
                <div className="text-center p-6 text-[#FAF7F5] space-y-2">
                  <Crosshair className="h-10 w-10 text-[#61554E] mx-auto opacity-60" />
                  <p className="text-xs font-bold text-[#FAF7F5]">No Radiograph Loaded</p>
                  <p className="text-[11px] text-[#61554E]">
                    Select a chest X-ray image or DICOM study to visualize bounding boxes.
                  </p>
                </div>
              )}
            </div>

            {/* Findings Detail List */}
            {result && (
              <div className="space-y-3 pt-2">
                <div className="flex items-center justify-between">
                  <h5 className="text-xs font-bold text-[#1A1512]">Detected Anatomical Regions</h5>
                  <span className="text-[11px] font-mono text-[#61554E]">
                    {result.image_dimensions
                      ? `${result.image_dimensions.width} × ${result.image_dimensions.height} px`
                      : ''}
                  </span>
                </div>

                {result.num_detections > 0 ? (
                  <div className="overflow-x-auto rounded-xl border border-[#EAE3DD]">
                    <table className="w-full text-left text-xs">
                      <thead className="bg-[#FAF7F5] text-[#61554E] font-bold border-b border-[#EAE3DD]">
                        <tr>
                          <th className="py-2 px-3">#</th>
                          <th className="py-2 px-3">Finding Class</th>
                          <th className="py-2 px-3">Confidence</th>
                          <th className="py-2 px-3">Bounding Box [x1, y1, x2, y2]</th>
                          <th className="py-2 px-3">Area</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-[#EAE3DD]">
                        {result.detections.map((det) => (
                          <tr key={det.id} className="hover:bg-[#FAF7F5]">
                            <td className="py-2 px-3 font-mono font-bold text-[#014BAA]">{det.id}</td>
                            <td className="py-2 px-3 font-semibold text-[#1A1512]">{det.class_name}</td>
                            <td className="py-2 px-3 font-mono text-emerald-700 font-bold">
                              {(det.confidence * 100).toFixed(1)}%
                            </td>
                            <td className="py-2 px-3 font-mono text-[11px] text-[#61554E]">
                              [{det.bbox_pixels.join(', ')}]
                            </td>
                            <td className="py-2 px-3 font-mono text-[11px] text-[#61554E]">
                              {det.area_pixels} px²
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                ) : (
                  <div className="p-4 rounded-xl bg-[#FAF7F5] border border-[#EAE3DD] text-xs text-[#61554E] space-y-2.5">
                    <div className="flex items-start gap-2.5">
                      {result.status === 'not_trained' ? (
                        <AlertCircle className="h-4 w-4 text-amber-600 shrink-0 mt-0.5" />
                      ) : (
                        <CheckCircle className="h-4 w-4 text-emerald-600 shrink-0 mt-0.5" />
                      )}
                      <div className="space-y-1">
                        <p className="font-bold text-[#1A1512]">
                          {result.status === 'not_trained'
                            ? 'Untrained Model — Zero Fabricated Detections'
                            : `0 Anatomical Findings Detected (Above ${(confThreshold * 100).toFixed(0)}% Confidence)`}
                        </p>
                        <p className="text-[11px] leading-relaxed">
                          {result.status === 'not_trained'
                            ? 'Model weights are not configured. Zero bounding boxes were fabricated in adherence to clinical validation safeguards.'
                            : highestCandidateConfidence !== null
                            ? `Inference executed across spatial dimensions (${result.image_dimensions ? `${result.image_dimensions.width} × ${result.image_dimensions.height} px` : 'the radiograph'}). Highest candidate proposal confidence was ${(highestCandidateConfidence * 100).toFixed(1)}%, which did not exceed the active ${(confThreshold * 100).toFixed(0)}% threshold (NMS IoU: ${(iouThreshold * 100).toFixed(0)}%).`
                            : `Inference executed across spatial dimensions (${result.image_dimensions ? `${result.image_dimensions.width} × ${result.image_dimensions.height} px` : 'the radiograph'}). No candidate detections exceeded the active confidence threshold of ${(confThreshold * 100).toFixed(0)}% (NMS IoU: ${(iouThreshold * 100).toFixed(0)}%).`}
                        </p>
                      </div>
                    </div>

                    {result.status === 'success' && (
                      <div className="p-3 rounded-lg bg-blue-50/70 border border-blue-200/60 text-[11px] text-blue-900 space-y-1.5">
                        <div className="flex items-center justify-between">
                          <div className="flex items-center gap-1.5 font-semibold text-[#014BAA]">
                            <Sliders className="h-3.5 w-3.5" />
                            <span>Investigational Threshold Guidance</span>
                          </div>
                          {highestCandidateConfidence !== null && (
                            <span className="font-mono text-[10px] font-bold px-2 py-0.5 rounded-md bg-blue-100 text-[#014BAA] border border-blue-200">
                              Peak Proposal: {(highestCandidateConfidence * 100).toFixed(1)}%
                            </span>
                          )}
                        </div>
                        <p className="text-blue-800 text-[11px] leading-relaxed">
                          {highestCandidateConfidence !== null
                            ? `The highest raw candidate proposal scored ${(highestCandidateConfidence * 100).toFixed(1)}% confidence, which is below the active ${(confThreshold * 100).toFixed(0)}% cutoff. To inspect these localized candidate proposals for research exploration, lower the Confidence Threshold slider below ${(highestCandidateConfidence * 100).toFixed(1)}% (supported down to 1%). No boxes are fabricated or hallucinated.`
                            : `All candidate proposals scored below the active ${(confThreshold * 100).toFixed(0)}% threshold and were filtered out. If investigating lower-probability localization proposals for research exploration, you may lower the Confidence Threshold slider (supported down to 1%). No bounding boxes are fabricated or hallucinated.`}
                        </p>
                      </div>
                    )}
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Clinical & Academic Disclaimer */}
      <div className="rounded-2xl border border-[#EAE3DD] bg-white/95 p-4 shadow-xs">
        <div className="flex items-start gap-2.5">
          <ShieldAlert className="h-4 w-4 text-[#61554E] shrink-0 mt-0.5" />
          <p className="text-[11px] text-[#61554E] leading-relaxed">
            <strong className="text-[#1A1512]">Investigational Research Disclaimer: </strong>
            MediVision-AI anatomical lesion localization is an experimental research tool. Bounding box coordinates represent localized activation patterns and must not be used as standalone clinical diagnoses or surgical guides.
          </p>
        </div>
      </div>
    </div>
  );
};
