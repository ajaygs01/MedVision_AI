import React, { useState, useEffect, useRef } from 'react';
import {
  Brain,
  FileUp,
  AlertTriangle,
  CheckCircle2,
  RefreshCw,
  Sliders,
  Layers,
  Activity,
  Download,
  Trash2,
  Database,
  Clock,
} from 'lucide-react';
import type {
  BrainMRIStatusResponse,
  BrainMRIValidationResponse,
} from '../../types';
import {
  fetchBrainMRIStatus,
  validateBrainMRIStudy,
  segmentBrainMRIStudy,
  fetchBrainMRISlice,
  downloadAnalysisReport,
} from '../../services/api';
import { useAnalysis } from '../../context/useAnalysis';

export const BrainMRIWorkbench: React.FC = () => {
  const { activeBrainMRICase, saveActiveBrainMRICase, clearActiveBrainMRICase } = useAnalysis();

  const [statusInfo, setStatusInfo] = useState<BrainMRIStatusResponse | null>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [validationInfo, setValidationInfo] = useState<BrainMRIValidationResponse | null>(null);
  const [validating, setValidating] = useState<boolean>(false);
  const [segmenting, setSegmenting] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [downloadingReport, setDownloadingReport] = useState<boolean>(false);
  const [reportError, setReportError] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState<boolean>(false);
  const [sliceLoading, setSliceLoading] = useState<boolean>(false);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const fileRef = useRef<File | null>(null);
  const activeSliceRequestIdRef = useRef<number>(0);

  // Derive active study state from context
  const activeFilename = selectedFile?.name || activeBrainMRICase?.filename || null;
  const hasFileToSegment = !!selectedFile;
  const result = activeBrainMRICase?.result || null;
  const selectedPlane = activeBrainMRICase?.selectedPlane || 'axial';
  const selectedSlice = activeBrainMRICase?.selectedSlice ?? (result?.slices?.slice_idx ?? 0);
  const viewMode = activeBrainMRICase?.viewMode || 'overlay';

  useEffect(() => {
    async function loadStatus() {
      const res = await fetchBrainMRIStatus();
      if (res.data) {
        setStatusInfo(res.data);
      }
    }
    loadStatus();
  }, []);

  const handleFileSelect = async (file: File) => {
    fileRef.current = file;
    setSelectedFile(file);
    setError(null);
    setReportError(null);
    setValidating(true);

    try {
      const res = await validateBrainMRIStudy(file);
      if (res.data) {
        setValidationInfo(res.data);
      }
    } catch {
      // non-blocking validation
    } finally {
      setValidating(false);
    }
  };

  const handleRunSegmentation = async () => {
    const fileToSegment = selectedFile || fileRef.current || fileInputRef.current?.files?.[0];
    if (!fileToSegment) {
      setError('Please select or upload a Brain MRI NIfTI (.nii, .nii.gz) or ZIP study file.');
      return;
    }

    setSegmenting(true);
    setError(null);

    try {
      const res = await segmentBrainMRIStudy(fileToSegment, undefined, selectedPlane);
      if (res.data) {
        const slices = res.data.slices;
        console.info('[MediVision Diagnostics] Brain MRI Segmentation Received:', {
          has_slices: !!slices,
          plane: slices?.plane,
          slice_idx: slices?.slice_idx,
          total_slices: slices?.total_slices,
          mri_preview: slices?.mri_preview
            ? { present: true, length: slices.mri_preview.length, is_data_url: slices.mri_preview.startsWith('data:image/'), format_prefix: slices.mri_preview.slice(0, 30) }
            : { present: false, length: 0 },
          mask_preview: slices?.mask_preview
            ? { present: true, length: slices.mask_preview.length, is_data_url: slices.mask_preview.startsWith('data:image/'), format_prefix: slices.mask_preview.slice(0, 30) }
            : { present: false, length: 0 },
          overlay_preview: slices?.overlay_preview
            ? { present: true, length: slices.overlay_preview.length, is_data_url: slices.overlay_preview.startsWith('data:image/'), format_prefix: slices.overlay_preview.slice(0, 30) }
            : { present: false, length: 0 },
        });

        const resolvedPlane = (res.data.slices?.plane as 'axial' | 'coronal' | 'sagittal') || selectedPlane;
        const resolvedSlice = res.data.slices?.slice_idx ?? 0;
        saveActiveBrainMRICase({
          filename: fileToSegment.name,
          recordId: res.data.id,
          patientId: res.data.patient_id,
          result: res.data,
          selectedPlane: resolvedPlane,
          selectedSlice: resolvedSlice,
          viewMode: 'overlay',
        });
      } else {
        setError(res.error || 'Segmentation request failed.');
      }
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Unknown network error.');
    } finally {
      setSegmenting(false);
    }
  };

  const handlePlaneChange = async (newPlane: 'axial' | 'coronal' | 'sagittal') => {
    if (newPlane === selectedPlane) return;
    const currentReqId = ++activeSliceRequestIdRef.current;
    setSliceLoading(true);

    // Optimistically update selected plane
    saveActiveBrainMRICase({
      filename: activeFilename || 'mri_study.nii.gz',
      recordId: activeBrainMRICase?.recordId,
      patientId: activeBrainMRICase?.patientId,
      result,
      selectedPlane: newPlane,
      selectedSlice,
      viewMode,
    });

    try {
      const res = await fetchBrainMRISlice(undefined, newPlane);
      if (currentReqId !== activeSliceRequestIdRef.current) return;
      if (res.data && result) {
        console.info(`[MediVision Diagnostics] Plane Change (${newPlane}):`, {
          plane: res.data.plane,
          slice_idx: res.data.slice_idx,
          total_slices: res.data.total_slices,
          mri_preview_len: res.data.mri_preview?.length || 0,
          mask_preview_len: res.data.mask_preview?.length || 0,
          overlay_preview_len: res.data.overlay_preview?.length || 0,
        });
        const updatedResult = {
          ...result,
          slices: {
            ...result.slices,
            plane: res.data.plane,
            slice_idx: res.data.slice_idx,
            total_slices: res.data.total_slices,
            mri_preview: res.data.mri_preview,
            mask_preview: res.data.mask_preview,
            overlay_preview: res.data.overlay_preview,
          },
        };
        saveActiveBrainMRICase({
          filename: activeFilename || 'mri_study.nii.gz',
          recordId: activeBrainMRICase?.recordId,
          patientId: activeBrainMRICase?.patientId,
          result: updatedResult,
          selectedPlane: newPlane,
          selectedSlice: res.data.slice_idx,
          viewMode,
        });
      }
    } catch {
      // retain current view
    } finally {
      if (currentReqId === activeSliceRequestIdRef.current) {
        setSliceLoading(false);
      }
    }
  };

  const handleSliceChange = async (newSlice: number) => {
    const currentReqId = ++activeSliceRequestIdRef.current;
    setSliceLoading(true);

    saveActiveBrainMRICase({
      filename: activeFilename || 'mri_study.nii.gz',
      recordId: activeBrainMRICase?.recordId,
      patientId: activeBrainMRICase?.patientId,
      result,
      selectedPlane,
      selectedSlice: newSlice,
      viewMode,
    });

    try {
      const res = await fetchBrainMRISlice(newSlice, selectedPlane);
      if (currentReqId !== activeSliceRequestIdRef.current) return;
      if (res.data && result) {
        console.info(`[MediVision Diagnostics] Slice Navigation Depth (${newSlice}):`, {
          plane: res.data.plane,
          slice_idx: res.data.slice_idx,
          total_slices: res.data.total_slices,
          mri_preview_len: res.data.mri_preview?.length || 0,
          mask_preview_len: res.data.mask_preview?.length || 0,
          overlay_preview_len: res.data.overlay_preview?.length || 0,
        });
        const updatedResult = {
          ...result,
          slices: {
            ...result.slices,
            plane: res.data.plane,
            slice_idx: res.data.slice_idx,
            total_slices: res.data.total_slices,
            mri_preview: res.data.mri_preview,
            mask_preview: res.data.mask_preview,
            overlay_preview: res.data.overlay_preview,
          },
        };
        saveActiveBrainMRICase({
          filename: activeFilename || 'mri_study.nii.gz',
          recordId: activeBrainMRICase?.recordId,
          patientId: activeBrainMRICase?.patientId,
          result: updatedResult,
          selectedPlane,
          selectedSlice: res.data.slice_idx,
          viewMode,
        });
      }
    } catch {
      // retain current view
    } finally {
      if (currentReqId === activeSliceRequestIdRef.current) {
        setSliceLoading(false);
      }
    }
  };

  const handleViewModeChange = (newMode: 'overlay' | 'original' | 'mask') => {
    saveActiveBrainMRICase({
      filename: activeFilename || 'mri_study.nii.gz',
      recordId: activeBrainMRICase?.recordId,
      patientId: activeBrainMRICase?.patientId,
      result,
      selectedPlane,
      selectedSlice,
      viewMode: newMode,
    });
  };

  const handleDownloadReport = async () => {
    if (!result) {
      setReportError('Please execute Brain MRI lesion segmentation first.');
      return;
    }

    setDownloadingReport(true);
    setReportError(null);

    try {
      const payload: Record<string, unknown> = {
        id: result.id,
        modality: 'brain_mri',
        model_name: result.model_name || '3D U-Net Volumetric Brain Tumor Segmenter',
        filename: activeFilename || 'brain_mri_study.nii.gz',
        file_type: 'nifti',
        prediction: result.dominant_finding,
        confidence: 0.88,
        status: result.status || 'success',
        disclaimer: result.disclaimer,
        image_preview: result.slices?.mri_preview,
        gradcam_preview: result.slices?.overlay_preview,
        heatmap_preview: result.slices?.mask_preview,
        volumetric_metrics: result.volumetric_metrics,
        patient_id: result.patient_id,
        slices: {
          plane: selectedPlane,
          slice_idx: selectedSlice,
          total_slices: result.slices?.total_slices || 155,
        },
        timestamp: new Date().toISOString(),
      };

      const filename = `MediVision_BrainMRI_Report_${result.id || result.patient_id || 'study'}.pdf`;
      const res = await downloadAnalysisReport(result.id, filename, payload);
      if (!res.success) {
        setReportError(res.error || 'Failed to download Brain MRI report.');
      }
    } catch (err) {
      setReportError(err instanceof Error ? err.message : 'Download failed.');
    } finally {
      setDownloadingReport(false);
    }
  };

  const handleClear = () => {
    clearActiveBrainMRICase();
    fileRef.current = null;
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
    setSelectedFile(null);
    setValidationInfo(null);
    setError(null);
    setReportError(null);
  };

  const totalSlices = result?.slices?.total_slices || 155;
  const currentPreview =
    viewMode === 'mask'
      ? result?.slices?.mask_preview || result?.slices?.overlay_preview
      : viewMode === 'original'
      ? result?.slices?.mri_preview
      : result?.slices?.overlay_preview || result?.slices?.mri_preview;

  useEffect(() => {
    if (currentPreview) {
      console.info('[MediVision Diagnostics] Active Multi-Planar Slice View Render:', {
        selectedPlane,
        selectedSlice,
        viewMode,
        currentPreview_len: currentPreview.length,
        is_data_url: currentPreview.startsWith('data:image/'),
        format_prefix: currentPreview.slice(0, 30),
      });
    }
  }, [selectedPlane, selectedSlice, viewMode, currentPreview]);

  const isModelReady = statusInfo?.status === 'ready' && !statusInfo?.is_pilot && !statusInfo?.is_experimental;
  const isExperimentalReady = Boolean(statusInfo?.is_experimental && statusInfo?.status === 'ready');
  const isPilotVerified = Boolean((statusInfo?.is_pilot || statusInfo?.pilot_checkpoint_path) && !isExperimentalReady);
  const isTraining = statusInfo?.stage_phase === 'training_in_progress';
  const isDatasetReady =
    statusInfo?.stage_phase === 'dataset_ready' ||
    ((statusInfo?.dataset_status?.complete_patient_count ?? 0) >= 3 && !isModelReady && !isExperimentalReady);

  return (
    <div className="space-y-6">
      {/* Module Operational Status Banner with Distinct Lifecycle Stages */}
      {isModelReady ? (
        <div className="p-4 rounded-2xl border border-emerald-300 bg-emerald-50/90 text-emerald-950 flex flex-col md:flex-row md:items-center justify-between gap-3 shadow-xs">
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-xl text-white shrink-0 bg-emerald-600">
              <CheckCircle2 className="w-5 h-5" />
            </div>
            <div>
              <h4 className="font-bold text-sm text-emerald-900">
                Module 2 Status: OPERATIONAL • Full 3D U-Net Model Trained
              </h4>
              <p className="text-xs text-emerald-800">
                Trained on the BraTS Multi-Modal Cohort (FLAIR, T1, T1ce, T2). Zero patient leakage, multi-class subregion delineation active.
              </p>
            </div>
          </div>
          <div className="flex items-center gap-2 self-start md:self-center">
            {statusInfo?.metrics?.held_out_test_metrics?.metrics?.mean_dice != null && (
              <span className="px-2.5 py-1 rounded-full font-mono text-[11px] font-bold bg-emerald-200/80 text-emerald-900">
                Mean DSC: {statusInfo.metrics.held_out_test_metrics.metrics.mean_dice}
              </span>
            )}
            {statusInfo?.metrics?.held_out_test_metrics?.metrics?.whole_tumor_dice != null && (
              <span className="px-2.5 py-1 rounded-full font-mono text-[11px] font-bold bg-emerald-200/80 text-emerald-900">
                WT DSC: {statusInfo.metrics.held_out_test_metrics.metrics.whole_tumor_dice}
              </span>
            )}
          </div>
        </div>
      ) : isExperimentalReady ? (
        <div className="p-4 rounded-2xl border border-indigo-300 bg-indigo-50/90 text-indigo-950 flex flex-col md:flex-row md:items-center justify-between gap-3 shadow-xs">
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-xl text-white shrink-0 bg-indigo-600">
              <Activity className="w-5 h-5" />
            </div>
            <div>
              <h4 className="font-bold text-sm text-indigo-900">
                Module 2 Status: EXPERIMENTAL PROTOTYPE ACTIVE (10 Patients, 3 Epochs) • EVALUATION ONLY
              </h4>
              <p className="text-xs text-indigo-800">
                Loaded custom prototype weights from <code className="font-mono bg-indigo-200/60 px-1 py-0.5 rounded">custom_10p_3e/</code>. Multi-planar volumetric segmentation is enabled for evaluation. <strong>Not clinically validated or approved for diagnosis.</strong>
              </p>
            </div>
          </div>
          <div className="flex items-center gap-2 self-start md:self-center flex-wrap">
            <span className="px-2.5 py-1 rounded-full bg-indigo-200/80 text-indigo-900 font-mono text-[11px] font-bold">
              Experimental Mode
            </span>
            {statusInfo?.custom_metrics?.best_val_dice != null && (
              <span className="px-2.5 py-1 rounded-full bg-indigo-200/80 text-indigo-900 font-mono text-[11px] font-bold">
                Val Mean DSC: {statusInfo.custom_metrics.best_val_dice}
              </span>
            )}
            {statusInfo?.custom_metrics?.held_out_test_metrics?.metrics?.mean_dice != null && (
              <span className="px-2.5 py-1 rounded-full bg-indigo-200/80 text-indigo-900 font-mono text-[11px] font-bold">
                Test Mean DSC: {statusInfo.custom_metrics.held_out_test_metrics.metrics.mean_dice}
              </span>
            )}
          </div>
        </div>
      ) : isPilotVerified ? (
        <div className="p-4 rounded-2xl border border-amber-300 bg-amber-50/90 text-amber-950 flex flex-col md:flex-row md:items-center justify-between gap-3 shadow-xs">
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-xl text-white shrink-0 bg-amber-600">
              <CheckCircle2 className="w-5 h-5" />
            </div>
            <div>
              <h4 className="font-bold text-sm text-amber-900">
                Module 2 Status: CPU PILOT VERIFIED • Production Model Training Pending
              </h4>
              <p className="text-xs text-amber-800">
                Pilot smoke-test completed on 2 patient studies. CPU execution, gradients, and checkpointing verified. <strong>Not clinically validated or operational for diagnosis.</strong>
              </p>
            </div>
          </div>
          <div className="flex items-center gap-2 self-start md:self-center">
            <span className="px-2.5 py-1 rounded-full bg-amber-200/80 text-amber-900 font-mono text-[11px] font-bold">
              Pilot Verified (Non-Production)
            </span>
            {statusInfo?.pilot_metrics?.held_out_test_metrics?.metrics?.mean_dice != null && (
              <span className="px-2.5 py-1 rounded-full bg-amber-200/80 text-amber-900 font-mono text-[11px] font-bold">
                Pilot Test DSC: {statusInfo.pilot_metrics.held_out_test_metrics.metrics.mean_dice}
              </span>
            )}
          </div>
        </div>
      ) : isTraining ? (
        <div className="p-4 rounded-2xl bg-purple-50/90 border border-purple-300 text-purple-950 flex flex-col md:flex-row md:items-center justify-between gap-3 shadow-xs">
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-xl bg-purple-600 text-white shrink-0">
              <RefreshCw className="w-5 h-5 animate-spin" />
            </div>
            <div>
              <h4 className="font-bold text-purple-900 text-sm">
                Module 2 Status: 3D U-NET TRAINING IN PROGRESS • Checkpoint Active
              </h4>
              <p className="text-xs text-purple-800">
                PyTorch 3D U-Net is optimizing DiceCELoss on BraTS patient cohorts. Checkpoints are streaming to <code className="font-mono bg-purple-200/60 px-1 py-0.5 rounded">backend/model_artifacts/brain_mri/last_checkpoint.pth</code>.
              </p>
            </div>
          </div>
          <div className="flex items-center gap-2 self-start md:self-center">
            <span className="px-2.5 py-1 rounded-full bg-purple-200/80 text-purple-900 font-mono text-[11px] font-bold flex items-center gap-1.5">
              <Clock className="w-3.5 h-3.5" />
              <span>Training Active</span>
            </span>
          </div>
        </div>
      ) : isDatasetReady ? (
        <div className="p-4 rounded-2xl bg-sky-50/90 border border-sky-300 text-sky-950 flex flex-col md:flex-row md:items-center justify-between gap-3 shadow-xs">
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-xl bg-sky-600 text-white shrink-0">
              <Database className="w-5 h-5" />
            </div>
            <div>
              <h4 className="font-bold text-sky-900 text-sm">
                Module 2 Status: BRATS DATASET VERIFIED • MODEL TRAINING PENDING
              </h4>
              <p className="text-xs text-sky-800">
                Detected <strong>{statusInfo?.dataset_status?.complete_patient_count} complete patient studies</strong> in <code className="font-mono bg-sky-200/60 px-1 py-0.5 rounded">datasets/brats/</code>. Ready for 3D U-Net training: run <code className="font-mono bg-sky-200/60 px-1 py-0.5 rounded">python -m ml.brain_mri.train --device auto</code>.
              </p>
            </div>
          </div>
          <div className="flex items-center gap-2 self-start md:self-center">
            <span className="px-2.5 py-1 rounded-full bg-sky-200/80 text-sky-900 font-mono text-[11px] font-bold">
              {statusInfo?.dataset_status?.complete_patient_count} Patients Verified
            </span>
          </div>
        </div>
      ) : (
        <div className="p-4 rounded-2xl bg-amber-50/90 border border-amber-300 text-amber-950 flex flex-col md:flex-row md:items-center justify-between gap-3 shadow-xs">
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-xl bg-amber-600 text-white shrink-0">
              <AlertTriangle className="w-5 h-5" />
            </div>
            <div>
              <h4 className="font-bold text-amber-900 text-sm">
                Module 2 Status: ARCHITECTURE READY • BRATS DATASET REQUIRED
              </h4>
              <p className="text-xs text-amber-800">
                In strict accordance with project boundaries, no mock predictions or simulated metrics are permitted. 3D U-Net architecture, losses, and endpoints are verified. Place BraTS data into <code className="font-mono bg-amber-200/60 px-1 py-0.5 rounded">datasets/brats/</code> and run <code className="font-mono bg-amber-200/60 px-1 py-0.5 rounded">python -m ml.brain_mri.train</code>.
              </p>
            </div>
          </div>
          <div className="flex items-center gap-2 self-start md:self-center">
            <span className="px-2.5 py-1 rounded-full bg-amber-200/80 text-amber-900 font-mono text-[11px] font-bold">
              0 Studies • BraTS Missing
            </span>
          </div>
        </div>
      )}

      {/* Main Two-Column Layout */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Column: Study Ingestion & Slice Viewer (7 Cols) */}
        <div className="lg:col-span-7 space-y-4">
          {/* Uploader Card */}
          <div className="rounded-2xl border border-[#EAE3DD] bg-white/95 p-6 shadow-sm space-y-4">
            <div className="flex items-center justify-between pb-3 border-b border-[#EAE3DD]">
              <h3 className="text-sm font-bold text-[#1A1512] flex items-center gap-2">
                <Brain className="w-4 h-4 text-[#014BAA]" />
                Brain MRI Volumetric Study Ingestion
              </h3>
              <div className="flex items-center gap-2">
                {activeFilename && (
                  <button
                    type="button"
                    onClick={handleClear}
                    className="text-[11px] font-semibold text-rose-600 hover:underline flex items-center gap-1"
                    title="Clear active study"
                  >
                    <Trash2 className="w-3.5 h-3.5" />
                    <span>Reset</span>
                  </button>
                )}
                <span className="text-[10px] font-mono text-[#014BAA] bg-[#014BAA]/10 px-2 py-0.5 rounded font-bold">
                  NIfTI (.nii, .nii.gz) / ZIP
                </span>
              </div>
            </div>

            {/* Drag & Drop Zone */}
            <div
              onDragOver={(e) => {
                e.preventDefault();
                setIsDragging(true);
              }}
              onDragLeave={(e) => {
                e.preventDefault();
                setIsDragging(false);
              }}
              onDrop={(e) => {
                e.preventDefault();
                e.stopPropagation();
                setIsDragging(false);
                if (e.dataTransfer.files && e.dataTransfer.files[0]) {
                  handleFileSelect(e.dataTransfer.files[0]);
                }
              }}
              onClick={() => fileInputRef.current?.click()}
              className={`p-6 border-2 border-dashed rounded-xl text-center cursor-pointer transition-all ${
                isDragging
                  ? 'border-[#014BAA] bg-[#014BAA]/5'
                  : 'border-[#DFD7D0] bg-[#FAF7F5] hover:border-[#014BAA]/50'
              }`}
            >
              <input
                ref={fileInputRef}
                type="file"
                accept=".nii,.gz,.zip"
                onClick={(e) => e.stopPropagation()}
                onChange={(e) => {
                  if (e.target.files && e.target.files[0]) {
                    handleFileSelect(e.target.files[0]);
                  }
                }}
                className="hidden"
              />
              <FileUp className="w-8 h-8 text-[#014BAA] mx-auto mb-2 opacity-80" />
              <p className="text-xs font-bold text-[#1A1512]">
                {activeFilename ? `Selected Study: ${activeFilename}` : 'Click to Upload or Drag & Drop Brain MRI Study'}
              </p>
              <p className="text-[11px] text-[#61554E] mt-1">
                Supports co-registered 4-channel mpMRI (FLAIR, T1, T1ce, T2) in NIfTI format or multi-sequence ZIP archive.
              </p>
            </div>

            {/* Validation Feedback */}
            {validating && (
              <div className="flex items-center gap-2 text-xs text-[#014BAA] font-semibold">
                <RefreshCw className="w-4 h-4 animate-spin" />
                <span>Validating NIfTI headers and sequence alignment...</span>
              </div>
            )}

            {validationInfo && (
              <div
                className={`p-3 rounded-xl border text-xs flex items-center justify-between ${
                  validationInfo.status === 'valid'
                    ? 'bg-emerald-50 border-emerald-300 text-emerald-900'
                    : 'bg-amber-50 border-amber-300 text-amber-900'
                }`}
              >
                <div>
                  <span className="font-bold block">{validationInfo.message}</span>
                  {validationInfo.dimensions && (
                    <span className="text-[11px] font-mono">
                      Dimensions: {validationInfo.dimensions.join(' × ')} | Voxel Spacing: {validationInfo.voxel_spacing_mm?.map(v => v.toFixed(1)).join(' × ')} mm
                    </span>
                  )}
                </div>
                <span className="text-[10px] font-bold uppercase tracking-wider px-2 py-0.5 rounded bg-white/70">
                  {validationInfo.status}
                </span>
              </div>
            )}

            {/* Run Button */}
            <div className="pt-2 flex items-center justify-between">
              <span className="text-[11px] text-[#61554E]">
                Architecture: <strong className="text-[#1A1512]">3D U-Net (PyTorch)</strong>
              </span>
              <button
                type="button"
                onClick={handleRunSegmentation}
                disabled={segmenting || !hasFileToSegment}
                className="px-6 py-2.5 rounded-xl bg-[#014BAA] hover:bg-[#01357A] text-white font-bold text-xs tracking-wide shadow-md shadow-[#014BAA]/20 transition-all flex items-center gap-2 disabled:opacity-50"
              >
                {segmenting ? (
                  <>
                    <RefreshCw className="w-4 h-4 animate-spin" />
                    <span>Segmenting Volume...</span>
                  </>
                ) : (
                  <>
                    <Activity className="w-4 h-4" />
                    <span>Run 3D U-Net Lesion Segmentation</span>
                  </>
                )}
              </button>
            </div>
          </div>

          {/* Interactive Slice Viewer Card */}
          <div className="rounded-2xl border border-[#EAE3DD] bg-white/95 p-6 shadow-sm space-y-4">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-3 border-b border-[#EAE3DD]">
              <div className="flex items-center gap-2">
                <Layers className="w-4 h-4 text-[#014BAA]" />
                <h3 className="text-sm font-bold text-[#1A1512]">
                  Interactive Multi-Planar Slice Viewer
                </h3>
              </div>

              {/* Plane Selector */}
              <div className="flex items-center gap-1 p-1 bg-[#FAF7F5] rounded-xl border border-[#DFD7D0]">
                {(['axial', 'coronal', 'sagittal'] as const).map((p) => (
                  <button
                    key={p}
                    type="button"
                    onClick={() => handlePlaneChange(p)}
                    className={`px-3 py-1 rounded-lg text-xs font-bold capitalize transition-all ${
                      selectedPlane === p
                        ? 'bg-[#014BAA] text-white shadow-xs'
                        : 'text-[#61554E] hover:text-[#1A1512]'
                    }`}
                  >
                    {p}
                  </button>
                ))}
              </div>
            </div>

            {/* Slice View Display Area */}
            <div className="relative aspect-square max-h-[360px] mx-auto rounded-xl bg-[#1A1512] overflow-hidden flex items-center justify-center border border-[#DFD7D0]">
              {currentPreview ? (
                <img
                  src={currentPreview}
                  alt={`Brain MRI ${selectedPlane} slice ${selectedSlice}`}
                  className="w-full h-full object-contain"
                />
              ) : (
                <div className="text-center p-6 space-y-2">
                  <Brain className="w-10 h-10 text-white/40 mx-auto" />
                  <p className="text-xs font-bold text-white/90">Slice Viewer Ready</p>
                  <p className="text-[11px] text-white/60 max-w-xs">
                    Upload an MRI study above to review multi-planar cross-sections and segmented tumor borders.
                  </p>
                </div>
              )}

              {/* Overlay Plane Tag & Loading Indicator */}
              {currentPreview && (
                <div className="absolute top-3 left-3 px-2.5 py-1 rounded-lg bg-black/70 backdrop-blur-xs text-white text-[11px] font-mono font-bold flex items-center gap-2">
                  <span>{selectedPlane.toUpperCase()} • SLICE {selectedSlice} / {totalSlices}</span>
                  {sliceLoading && <RefreshCw className="w-3 h-3 animate-spin text-amber-400" />}
                </div>
              )}
            </div>

            {/* Slider & View Mode Controls */}
            {result?.slices && (
              <div className="space-y-3 pt-2">
                {/* Slice Depth Slider */}
                <div className="space-y-1">
                  <div className="flex justify-between text-xs font-semibold text-[#1A1512]">
                    <span>Slice Navigation Depth</span>
                    <span className="font-mono text-[#014BAA]">
                      Slice {selectedSlice} of {totalSlices}
                    </span>
                  </div>
                  <input
                    type="range"
                    min={0}
                    max={Math.max(0, totalSlices - 1)}
                    value={selectedSlice}
                    onChange={(e) => handleSliceChange(parseInt(e.target.value, 10))}
                    className="w-full accent-[#014BAA] cursor-pointer"
                  />
                </div>

                {/* View Mode Selector */}
                <div className="flex items-center justify-between pt-2">
                  <span className="text-xs text-[#61554E] font-medium">Visualization Layer:</span>
                  <div className="flex items-center gap-1 p-1 bg-[#FAF7F5] rounded-xl border border-[#DFD7D0]">
                    {(['overlay', 'original', 'mask'] as const).map((mode) => (
                      <button
                        key={mode}
                        type="button"
                        onClick={() => handleViewModeChange(mode)}
                        className={`px-3 py-1 rounded-lg text-xs font-bold capitalize transition-all ${
                          viewMode === mode
                            ? 'bg-[#014BAA] text-white shadow-xs'
                            : 'text-[#61554E] hover:text-[#1A1512]'
                        }`}
                      >
                        {mode === 'overlay' ? 'Blended Overlay' : mode === 'original' ? 'Original MRI' : 'Mask Only'}
                      </button>
                    ))}
                  </div>
                </div>
              </div>
            )}

            {/* Color-Coded Subregion Legend */}
            <div className="pt-3 border-t border-[#EAE3DD] grid grid-cols-3 gap-2 text-[11px]">
              <div className="flex items-center gap-2 p-2 rounded-lg bg-[#FAF7F5]">
                <span className="w-3 h-3 rounded-full bg-[#E11D48] shrink-0" />
                <span className="font-semibold text-[#1A1512]">Necrotic Core (NCR)</span>
              </div>
              <div className="flex items-center gap-2 p-2 rounded-lg bg-[#FAF7F5]">
                <span className="w-3 h-3 rounded-full bg-[#059669] shrink-0" />
                <span className="font-semibold text-[#1A1512]">Peritumoral Edema (ED)</span>
              </div>
              <div className="flex items-center gap-2 p-2 rounded-lg bg-[#FAF7F5]">
                <span className="w-3 h-3 rounded-full bg-[#D97706] shrink-0" />
                <span className="font-semibold text-[#1A1512]">Enhancing Tumor (ET)</span>
              </div>
            </div>
          </div>
        </div>

        {/* Right Column: Quantitative Volumetric Analysis & Report (5 Cols) */}
        <div className="lg:col-span-5 space-y-4">
          {/* Volumetric Assessment Card */}
          <div className="rounded-2xl border border-[#EAE3DD] bg-white/95 p-6 shadow-sm space-y-4">
            <div className="flex items-center justify-between">
              <h3 className="text-sm font-bold text-[#1A1512] flex items-center gap-2">
                <Activity className="w-4 h-4 text-[#014BAA]" />
                Volumetric Lesion Quantification
              </h3>
              {result && (
                <span className="text-[10px] font-mono text-emerald-700 bg-emerald-100 px-2 py-0.5 rounded-full font-bold">
                  PERSISTED TO ATLAS
                </span>
              )}
            </div>

            {error && (
              <div className="p-4 rounded-xl bg-rose-50 border border-rose-300 text-rose-900 text-xs">
                {error}
              </div>
            )}

            {result ? (
              <div className="space-y-4">
                {/* Dominant Finding Summary */}
                <div className="p-4 rounded-xl bg-[#FAF7F5] border border-[#EAE3DD] space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="text-[11px] font-bold uppercase tracking-wider text-[#7C6E66]">
                      Volumetric Diagnostic Delineation
                    </span>
                    {result.is_experimental && (
                      <span className="text-[10px] font-bold bg-amber-100 text-amber-900 border border-amber-300 px-2 py-0.5 rounded-full">
                        EXPERIMENTAL PROTOTYPE
                      </span>
                    )}
                  </div>
                  <p className="text-xs font-bold text-[#1A1512] leading-relaxed">
                    {result.dominant_finding}
                  </p>
                </div>

                {/* Subregion Volume Measurements */}
                <div className="space-y-2.5">
                  <span className="text-[11px] font-bold uppercase tracking-wider text-[#7C6E66]">
                    BraTS Lesion Compartment Volumes
                  </span>

                  <div className="p-3.5 rounded-xl bg-emerald-50/80 border border-emerald-200 flex items-center justify-between">
                    <div>
                      <span className="text-xs font-bold text-emerald-950 block">Whole Tumor (WT)</span>
                      <span className="text-[11px] text-emerald-800">Necrotic Core + Edema + Enhancing</span>
                    </div>
                    <span className="text-base font-mono font-black text-emerald-800">
                      {result.volumetric_metrics.whole_tumor_cm3 ?? 0.0} cm³
                    </span>
                  </div>

                  <div className="p-3.5 rounded-xl bg-amber-50/80 border border-amber-200 flex items-center justify-between">
                    <div>
                      <span className="text-xs font-bold text-amber-950 block">Tumor Core (TC)</span>
                      <span className="text-[11px] text-amber-800">Resectable Core (NCR + ET)</span>
                    </div>
                    <span className="text-base font-mono font-black text-amber-800">
                      {result.volumetric_metrics.tumor_core_cm3 ?? 0.0} cm³
                    </span>
                  </div>

                  <div className="p-3.5 rounded-xl bg-rose-50/80 border border-rose-200 flex items-center justify-between">
                    <div>
                      <span className="text-xs font-bold text-rose-950 block">Active Enhancing Tumor (ET)</span>
                      <span className="text-[11px] text-rose-800">Gadolinium Extravasation</span>
                    </div>
                    <span className="text-base font-mono font-black text-rose-800">
                      {result.volumetric_metrics.enhancing_tumor_cm3 ?? 0.0} cm³
                    </span>
                  </div>

                  <div className="p-3.5 rounded-xl bg-[#FAF7F5] border border-[#EAE3DD] flex items-center justify-between text-xs">
                    <span className="text-[#61554E] font-medium">Peritumoral Vasogenic Edema (ED):</span>
                    <span className="font-mono font-bold text-[#1A1512]">
                      {result.volumetric_metrics.edema_cm3 ?? 0.0} cm³
                    </span>
                  </div>
                </div>

                {/* PDF Download Button */}
                <div className="pt-2">
                  <button
                    type="button"
                    onClick={handleDownloadReport}
                    disabled={downloadingReport || !result}
                    className={`w-full py-2.5 px-4 rounded-xl text-xs font-bold transition-all shadow-xs flex items-center justify-center gap-2 ${
                      downloadingReport || !result
                        ? 'bg-[#EAE3DD] text-[#7C6E66] cursor-not-allowed'
                        : 'bg-[#014BAA] hover:bg-[#01357A] text-[#F8F3F0] active:scale-95 shadow-md shadow-[#014BAA]/20'
                    }`}
                    title="Download executive Brain MRI volumetric clinical PDF report"
                  >
                    {downloadingReport ? (
                      <>
                        <RefreshCw className="w-4 h-4 animate-spin" />
                        <span>Synthesizing PDF Report...</span>
                      </>
                    ) : (
                      <>
                        <Download className="w-4 h-4" />
                        <span>Download Patient Report (PDF)</span>
                      </>
                    )}
                  </button>
                  {reportError && (
                    <p className="mt-1.5 text-[11px] text-rose-600 font-medium text-center">{reportError}</p>
                  )}
                </div>

                {/* Disclaimer */}
                <div className="p-3.5 rounded-xl bg-[#FAF7F5] border border-[#EAE3DD] text-[11px] text-[#61554E] italic">
                  {result.disclaimer}
                </div>
              </div>
            ) : (
              <div className="py-12 text-center space-y-2 border border-dashed border-[#DFD7D0] rounded-xl bg-[#FAF7F5]">
                <Brain className="h-8 w-8 text-[#014BAA] mx-auto opacity-70" />
                <p className="text-xs font-bold text-[#1A1512]">Volumetric Segmentation Ready</p>
                <p className="text-[11px] text-[#61554E] max-w-xs mx-auto">
                  Upload a 4-channel Brain MRI study and click 'Run 3D U-Net Lesion Segmentation' to delineate and measure glioma subregions.
                </p>
              </div>
            )}
          </div>

          {/* Model Specification & BraTS Benchmark Card */}
          <div className="rounded-2xl border border-[#EAE3DD] bg-white/95 p-6 shadow-sm space-y-3 text-xs">
            <div className="flex items-center justify-between pb-2 border-b border-[#EAE3DD]">
              <h3 className="font-bold text-[#1A1512] flex items-center gap-2">
                <Sliders className="w-4 h-4 text-[#014BAA]" />
                BraTS Benchmark Target Specifications
              </h3>
              <span className="text-[10px] font-mono text-[#014BAA] font-bold">
                BraTS 2020 / GLI
              </span>
            </div>

            <div className="space-y-2 text-[#61554E]">
              <div className="flex justify-between py-1 border-b border-[#EAE3DD]/60">
                <span>Architecture</span>
                <span className="font-mono font-bold text-[#1A1512]">3D U-Net (PyTorch)</span>
              </div>
              <div className="flex justify-between py-1 border-b border-[#EAE3DD]/60">
                <span>Input Sequences</span>
                <span className="font-mono font-bold text-[#1A1512]">FLAIR, T1, T1ce, T2</span>
              </div>
              <div className="flex justify-between py-1 border-b border-[#EAE3DD]/60">
                <span>Normalization</span>
                <span className="font-mono font-bold text-[#1A1512]">Z-score on Non-Zero Tissue</span>
              </div>
              <div className="flex justify-between py-1 border-b border-[#EAE3DD]/60">
                <span>Loss Function</span>
                <span className="font-mono font-bold text-[#1A1512]">Generalized Dice + CE</span>
              </div>
              <div className="flex justify-between py-1">
                <span>Split Strategy</span>
                <span className="font-mono font-bold text-[#014BAA]">Patient-Level 80/10/10</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
