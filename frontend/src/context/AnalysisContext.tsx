import React, { useState, useEffect, useCallback } from 'react';
import type {
  ActiveAnalysisCase,
  ActiveBrainMRICase,
  ActiveClinicalCase,
  ActiveYOLOCase,
  BrainMRISegmentResponse,
  ClinicalPatientInput,
  ClinicalRiskResponse,
  PredictionHistoryItem,
  YOLODetectionResponse,
} from '../types';
import { fetchHistoryRecord } from '../services/api';
import {
  saveCaseToIndexedDB,
  loadCaseFromIndexedDB,
  clearCaseFromIndexedDB,
} from '../utils/storage';
import { AnalysisContext } from './AnalysisContextDefinition';

const STORAGE_KEY = 'medivision_active_case_meta';
const CLINICAL_STORAGE_KEY = 'medivision_active_clinical_case';
const BRAIN_MRI_STORAGE_KEY = 'medivision_active_brain_mri_case';
const YOLO_STORAGE_KEY = 'medivision_active_yolo_case';


export const AnalysisProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [activeCase, setActiveCase] = useState<ActiveAnalysisCase | null>(() => {
    try {
      const stored = sessionStorage.getItem(STORAGE_KEY);
      if (stored) {
        const parsed = JSON.parse(stored);
        if (parsed && typeof parsed === 'object' && parsed.filename) {
          return parsed as ActiveAnalysisCase;
        }
      }
    } catch (err) {
      console.warn('Failed to parse active case from sessionStorage:', err);
    }
    return null;
  });

  const [activeClinicalCase, setActiveClinicalCase] = useState<ActiveClinicalCase | null>(() => {
    try {
      const stored = sessionStorage.getItem(CLINICAL_STORAGE_KEY);
      if (stored) {
        const parsed = JSON.parse(stored);
        if (parsed && typeof parsed === 'object' && parsed.formData) {
          return parsed as ActiveClinicalCase;
        }
      }
    } catch (err) {
      console.warn('Failed to parse active clinical case from sessionStorage:', err);
    }
    return null;
  });

  const saveActiveClinicalCase = useCallback((data: ActiveClinicalCase) => {
    setActiveClinicalCase(data);
    try {
      sessionStorage.setItem(CLINICAL_STORAGE_KEY, JSON.stringify(data));
    } catch (err) {
      console.warn('Failed to persist active clinical case to sessionStorage:', err);
    }
  }, []);

  const clearActiveClinicalCase = useCallback(() => {
    setActiveClinicalCase(null);
    try {
      sessionStorage.removeItem(CLINICAL_STORAGE_KEY);
    } catch (err) {
      console.warn('Failed to clear active clinical case from sessionStorage:', err);
    }
  }, []);

  const [activeBrainMRICase, setActiveBrainMRICase] = useState<ActiveBrainMRICase | null>(() => {
    try {
      const stored = sessionStorage.getItem(BRAIN_MRI_STORAGE_KEY);
      if (stored) {
        const parsed = JSON.parse(stored);
        if (parsed && typeof parsed === 'object' && parsed.filename) {
          return parsed as ActiveBrainMRICase;
        }
      }
    } catch (err) {
      console.warn('Failed to parse active brain mri case from sessionStorage:', err);
    }
    return null;
  });

  const saveActiveBrainMRICase = useCallback((data: ActiveBrainMRICase) => {
    setActiveBrainMRICase(data);
    try {
      sessionStorage.setItem(BRAIN_MRI_STORAGE_KEY, JSON.stringify(data));
    } catch (err) {
      console.warn('Failed to persist active brain mri case to sessionStorage:', err);
    }
  }, []);

  const clearActiveBrainMRICase = useCallback(() => {
    setActiveBrainMRICase(null);
    try {
      sessionStorage.removeItem(BRAIN_MRI_STORAGE_KEY);
    } catch (err) {
      console.warn('Failed to clear active brain mri case from sessionStorage:', err);
    }
  }, []);

  const [activeYOLOCase, setActiveYOLOCase] = useState<ActiveYOLOCase | null>(() => {
    try {
      const stored = sessionStorage.getItem(YOLO_STORAGE_KEY);
      if (stored) {
        const parsed = JSON.parse(stored);
        if (parsed && typeof parsed === 'object' && parsed.filename) {
          return parsed as ActiveYOLOCase;
        }
      }
    } catch (err) {
      console.warn('Failed to parse active yolo case from sessionStorage:', err);
    }
    return null;
  });

  const saveActiveYOLOCase = useCallback((data: ActiveYOLOCase) => {
    setActiveYOLOCase(data);
    try {
      sessionStorage.setItem(YOLO_STORAGE_KEY, JSON.stringify(data));
    } catch {
      // sessionStorage quota exceeded fallback: save stripped metadata without large base64 preview
      try {
        const stripped: ActiveYOLOCase = {
          ...data,
          imagePreview: null,
          result: data.result ? { ...data.result, preview_image: null } : null,
        };
        sessionStorage.setItem(YOLO_STORAGE_KEY, JSON.stringify(stripped));
      } catch (err) {
        console.warn('Failed to persist active yolo case to sessionStorage:', err);
      }
    }
  }, []);

  const clearActiveYOLOCase = useCallback(() => {
    setActiveYOLOCase(null);
    try {
      sessionStorage.removeItem(YOLO_STORAGE_KEY);
    } catch (err) {
      console.warn('Failed to clear active yolo case from sessionStorage:', err);
    }
  }, []);

  const [isRestored, setIsRestored] = useState<boolean>(false);

  // Initial rehydration: check IndexedDB for large image data and latest analysis
  useEffect(() => {
    let isMounted = true;
    async function rehydrate() {
      try {
        const fromDb = await loadCaseFromIndexedDB();
        if (isMounted && fromDb && fromDb.filename) {
          setActiveCase(fromDb);
        }
      } catch (err) {
        console.warn('IndexedDB initial rehydration skipped:', err);
      } finally {
        if (isMounted) setIsRestored(true);
      }
    }
    rehydrate();
    return () => {
      isMounted = false;
    };
  }, []);

  // Save case to both state, IndexedDB, and sessionStorage metadata
  const saveActiveCase = useCallback(async (data: ActiveAnalysisCase) => {
    setActiveCase(data);
    try {
      sessionStorage.setItem(STORAGE_KEY, JSON.stringify(data));
    } catch {
      // sessionStorage quota exceeded fallback: save stripped metadata
      try {
        const stripped = { ...data, imagePreview: null, gradcamPreview: null, heatmapPreview: null };
        sessionStorage.setItem(STORAGE_KEY, JSON.stringify(stripped));
      } catch {
        // ignore
      }
    }
    await saveCaseToIndexedDB(data);
  }, []);

  const updateViewMode = useCallback((mode: 'overlay' | 'heatmap' | 'original' | 'split') => {
    setActiveCase((prev) => {
      if (!prev) return prev;
      const updated: ActiveAnalysisCase = { ...prev, viewMode: mode };
      saveCaseToIndexedDB(updated).catch(() => {});
      try {
        sessionStorage.setItem(STORAGE_KEY, JSON.stringify(updated));
      } catch {
        // ignore
      }
      return updated;
    });
  }, []);

  const clearActiveCase = useCallback(async () => {
    setActiveCase(null);
    try {
      sessionStorage.removeItem(STORAGE_KEY);
    } catch (err) {
      console.warn('Failed to clear active case from sessionStorage:', err);
    }
    await clearCaseFromIndexedDB();
  }, []);

  const reopenHistoricalCase = useCallback(
    async (item: PredictionHistoryItem) => {
      let preview = item.image_preview || null;
      let gradcam = item.gradcam_preview || null;
      let heatmap = item.heatmap_preview || null;

      // If image or gradcam preview is missing in the item, fetch full record
      if ((!preview || !gradcam) && item.id) {
        try {
          const full = await fetchHistoryRecord(item.id);
          if (full.data) {
            preview = full.data.image_preview || preview;
            gradcam = full.data.gradcam_preview || gradcam;
            heatmap = full.data.heatmap_preview || heatmap;
          }
        } catch {
          // fallback to item properties
        }
      }

      const isCxr = item.modality === 'chest_xray' || !item.modality;
      const reconstructed: ActiveAnalysisCase = {
        recordId: item.id,
        filename: item.filename,
        fileType: item.file_type || (isCxr ? 'standard_image' : 'tabular_ehr'),
        isDicom: item.file_type === 'dicom' || item.filename.toLowerCase().endsWith('.dcm'),
        imagePreview: preview,
        gradcamPreview: gradcam,
        heatmapPreview: heatmap,
        prediction: item.prediction,
        predictedClassId: item.predicted_class_id ?? (item.confidence && item.confidence >= item.threshold ? 1 : 0),
        confidence: item.confidence,
        threshold: item.threshold ?? 0.5,
        modelName: item.model_name || (isCxr ? 'Chest X-ray ResNet50 Classifier' : 'Clinical Tabular Risk Predictor'),
        status: item.status || 'success',
        message: `Historical record retrieved from MongoDB Atlas (ID: ${item.id})`,
        disclaimer: item.disclaimer || 'Academic research prototype. Results are not clinically validated.',
        timestamp: item.timestamp || new Date().toISOString(),
        viewMode: 'overlay',
        modality: item.modality || 'chest_xray',
      };

      if (item.modality === 'clinical_ml') {
        let pData = item.patient_data;
        if (!pData && item.id) {
          try {
            const full = await fetchHistoryRecord(item.id);
            if (full.data && full.data.patient_data) {
              pData = full.data.patient_data;
            }
          } catch {
            // fallback
          }
        }

        const clinicalFormData: ClinicalPatientInput = (pData as ClinicalPatientInput) || {
          age: 50,
          sex: 1,
          cp: 1,
          trestbps: 120,
          chol: 200,
          fbs: 0,
          restecg: 0,
          thalach: 150,
          exang: 0,
          oldpeak: 0,
          slope: 1,
          ca: 0,
          thal: 3,
          patient_id: item.filename ? item.filename.replace(/^patient_|\.json$/g, '') : item.id,
        };

        const clinicalResult: ClinicalRiskResponse = {
          id: item.id,
          status: 'success',
          risk_probability: item.risk_probability ?? item.confidence ?? null,
          predicted_class_id: item.predicted_class_id ?? null,
          risk_tier: (item.risk_tier as any) || (
            (item.risk_probability ?? item.confidence ?? 0) >= 0.65
              ? 'Elevated Risk'
              : (item.risk_probability ?? item.confidence ?? 0) >= 0.35
              ? 'Moderate Risk'
              : 'Low Risk'
          ),
          prediction: item.prediction || 'Clinical Risk Evaluation',
          model_name: item.model_name || 'Random Forest Cardiovascular Risk Predictor',
          dataset_source: 'UCI Cleveland Heart Disease Cohort',
          top_contributing_factors: item.top_contributing_factors || [],
          message: `Historical record retrieved from MongoDB Atlas (ID: ${item.id})`,
          disclaimer: item.disclaimer || 'Academic research prototype. Results are not clinically validated.',
        };

        saveActiveClinicalCase({
          formData: clinicalFormData,
          result: clinicalResult,
          activePreset: 'custom',
        });
      }

      if (item.modality === 'brain_mri') {
        const brainMriResult: BrainMRISegmentResponse = {
          id: item.id,
          status: 'success',
          model_name: item.model_name || '3D U-Net Volumetric Brain Tumor Segmenter',
          filename: item.filename,
          patient_id: item.filename ? item.filename.split('_')[0] : 'BraTS_Patient',
          volumetric_metrics: item.volumetric_metrics || {
            whole_tumor_cm3: 0.0,
            tumor_core_cm3: 0.0,
            enhancing_tumor_cm3: 0.0,
            edema_cm3: 0.0,
          },
          dominant_finding: item.prediction || 'Volumetric Brain Lesion Delineation',
          slices: {
            plane: (item.slices?.plane as 'axial' | 'coronal' | 'sagittal') || 'axial',
            slice_idx: item.slices?.slice_idx ?? 0,
            total_slices: item.slices?.total_slices ?? 64,
            mri_preview: item.slices?.mri_preview || preview,
            mask_preview: item.slices?.mask_preview || heatmap,
            overlay_preview: item.slices?.overlay_preview || gradcam,
          },
          message: `Historical record retrieved from MongoDB Atlas (ID: ${item.id})`,
          disclaimer: item.disclaimer || 'Academic research prototype. Results are not clinically validated.',
        };

        saveActiveBrainMRICase({
          filename: item.filename,
          recordId: item.id,
          patientId: brainMriResult.patient_id,
          result: brainMriResult,
          selectedPlane: (item.slices?.plane as 'axial' | 'coronal' | 'sagittal') || 'axial',
          selectedSlice: item.slices?.slice_idx ?? 0,
          viewMode: 'overlay',
        });
      }

      if (item.modality === 'yolo_detection') {
        let yoloPreview = item.image_preview || preview;
        let detectionsList = item.detections || [];
        let numDets = item.num_detections ?? detectionsList.length;

        if (item.id && (!yoloPreview || detectionsList.length === 0)) {
          try {
            const full = await fetchHistoryRecord(item.id);
            if (full.data) {
              yoloPreview = full.data.image_preview || yoloPreview;
              detectionsList = full.data.detections || detectionsList;
              numDets = full.data.num_detections ?? detectionsList.length;
            }
          } catch {
            // fallback
          }
        }

        const yoloResult: YOLODetectionResponse = {
          id: item.id,
          status: (item.status as 'success' | 'not_trained' | 'error') || 'success',
          message: item.prediction || `Historical YOLO localization case restored (ID: ${item.id})`,
          filename: item.filename,
          detections: (detectionsList as any) || [],
          num_detections: numDets,
          image_dimensions: null,
          conf_threshold: item.threshold ?? 0.25,
          iou_threshold: 0.45,
          preview_image: yoloPreview,
          model_status: {
            model_architecture: 'YOLOv8n',
            trained: true,
            status: 'ready',
            checkpoint_path: 'yolov8_best.pt',
            classes: ['lung_opacity'],
            metrics: null,
          },
          persisted: true,
          analysis_id: item.id,
          disclaimer: item.disclaimer || 'Academic research prototype. Results are not clinically validated.',
        };

        saveActiveYOLOCase({
          recordId: item.id,
          filename: item.filename,
          imagePreview: yoloPreview,
          result: yoloResult,
          confThreshold: item.threshold ?? 0.25,
          iouThreshold: 0.45,
          timestamp: item.timestamp || new Date().toISOString(),
        });
      }

      await saveActiveCase(reconstructed);
    },
    [saveActiveCase, saveActiveClinicalCase, saveActiveBrainMRICase, saveActiveYOLOCase]
  );

  return (
    <AnalysisContext.Provider
      value={{
        activeCase,
        saveActiveCase,
        updateViewMode,
        clearActiveCase,
        reopenHistoricalCase,
        isRestored,
        activeClinicalCase,
        saveActiveClinicalCase,
        clearActiveClinicalCase,
        activeBrainMRICase,
        saveActiveBrainMRICase,
        clearActiveBrainMRICase,
        activeYOLOCase,
        saveActiveYOLOCase,
        clearActiveYOLOCase,
      }}
    >
      {children}
    </AnalysisContext.Provider>
  );
};

