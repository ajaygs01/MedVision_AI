import type {
  HealthResponse,
  ModelsStatusResponse,
  ChestXRayClassifyResponse,
  PredictionHistoryResponse,
  YOLODetectionResponse,
  YOLODetectionStatusResponse,
} from '../types';

const RAW_API_BASE = (import.meta.env.VITE_API_BASE_URL || '').replace(/\/+$/, '');

export function getApiUrl(endpoint: string): string {
  const cleanEndpoint = endpoint.startsWith('/') ? endpoint : `/${endpoint}`;
  if (!RAW_API_BASE) {
    return cleanEndpoint;
  }
  if (RAW_API_BASE.endsWith('/api') && cleanEndpoint.startsWith('/api')) {
    return `${RAW_API_BASE}${cleanEndpoint.slice(4)}`;
  }
  return `${RAW_API_BASE}${cleanEndpoint}`;
}

export async function fetchHealth(): Promise<{ data: HealthResponse | null; error: string | null }> {
  try {
    const res = await fetch(getApiUrl('/api/health'), {
      headers: {
        Accept: 'application/json',
      },
    });

    if (!res.ok) {
      return {
        data: null,
        error: `HTTP ${res.status}: ${res.statusText}`,
      };
    }

    const data: HealthResponse = await res.json();
    return { data, error: null };
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : 'Unknown network error';
    return {
      data: null,
      error: `Backend unreachable (${message}). Verify FastAPI server is running on port 8001.`,
    };
  }
}

export async function fetchModelsStatus(): Promise<{ data: ModelsStatusResponse | null; error: string | null }> {
  try {
    const res = await fetch(getApiUrl('/api/models/status'), {
      headers: {
        Accept: 'application/json',
      },
    });

    if (!res.ok) {
      return {
        data: null,
        error: `HTTP ${res.status}: ${res.statusText}`,
      };
    }

    const data: ModelsStatusResponse = await res.json();
    return { data, error: null };
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : 'Unknown network error';
    return {
      data: null,
      error: `Unable to load model statuses (${message}). Ensure backend server is running on port 8001.`,
    };
  }
}

export async function classifyChestXRay(
  file: File
): Promise<{ data: ChestXRayClassifyResponse | null; error: string | null }> {
  try {
    const formData = new FormData();
    formData.append('file', file);

    const res = await fetch(getApiUrl('/api/imaging/chest-xray/classify'), {
      method: 'POST',
      body: formData,
    });

    if (!res.ok) {
      let errorDetail = `HTTP ${res.status}: ${res.statusText}`;
      try {
        const errorJson = await res.json();
        if (errorJson.detail) {
          errorDetail = typeof errorJson.detail === 'string'
            ? errorJson.detail
            : JSON.stringify(errorJson.detail);
        }
      } catch {
        // ignore json parsing error
      }
      return { data: null, error: errorDetail };
    }

    const data: ChestXRayClassifyResponse = await res.json();
    return { data, error: null };
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : 'Unknown network error';
    return {
      data: null,
      error: `Classification request failed (${message}). Ensure backend server is running on port 8001.`,
    };
  }
}

export async function fetchPredictionHistory(
  limit: number = 50
): Promise<{ data: PredictionHistoryResponse | null; error: string | null }> {
  try {
    const res = await fetch(getApiUrl(`/api/imaging/history?limit=${limit}`), {
      headers: {
        Accept: 'application/json',
      },
    });

    if (!res.ok) {
      return {
        data: null,
        error: `HTTP ${res.status}: ${res.statusText}`,
      };
    }

    const data: PredictionHistoryResponse = await res.json();
    return { data, error: null };
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : 'Unknown network error';
    return {
      data: null,
      error: `Unable to fetch prediction history (${message}). Ensure backend server is running on port 8001.`,
    };
  }
}

export async function fetchHistoryRecord(
  recordId: string
): Promise<{ data: import('../types').PredictionHistoryItem | null; error: string | null }> {
  try {
    const res = await fetch(getApiUrl(`/api/imaging/history/${recordId}`), {
      headers: {
        Accept: 'application/json',
      },
    });

    if (!res.ok) {
      return {
        data: null,
        error: `HTTP ${res.status}: ${res.statusText}`,
      };
    }

    const data: import('../types').PredictionHistoryItem = await res.json();
    return { data, error: null };
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : 'Unknown network error';
    return {
      data: null,
      error: `Unable to fetch case record (${message}).`,
    };
  }
}

export async function predictClinicalRisk(
  patient: import('../types').ClinicalPatientInput
): Promise<{ data: import('../types').ClinicalRiskResponse | null; error: string | null }> {
  try {
    const res = await fetch(getApiUrl('/api/clinical-ml/predict'), {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Accept: 'application/json',
      },
      body: JSON.stringify(patient),
    });

    if (!res.ok) {
      let errorDetail = `HTTP ${res.status}: ${res.statusText}`;
      try {
        const errorJson = await res.json();
        if (errorJson.detail) {
          errorDetail = typeof errorJson.detail === 'string'
            ? errorJson.detail
            : JSON.stringify(errorJson.detail);
        }
      } catch {
        // ignore
      }
      return { data: null, error: errorDetail };
    }

    const data: import('../types').ClinicalRiskResponse = await res.json();
    return { data, error: null };
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : 'Unknown network error';
    return {
      data: null,
      error: `Clinical ML prediction request failed: ${message}`,
    };
  }
}

export async function fetchClinicalModelInfo(): Promise<{
  data: import('../types').ClinicalModelInfoResponse | null;
  error: string | null;
}> {
  try {
    const res = await fetch(getApiUrl('/api/clinical-ml/model-info'), {
      headers: {
        Accept: 'application/json',
      },
    });

    if (!res.ok) {
      return { data: null, error: `HTTP ${res.status}: ${res.statusText}` };
    }

    const data: import('../types').ClinicalModelInfoResponse = await res.json();
    return { data, error: null };
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : 'Unknown network error';
    return { data: null, error: `Unable to fetch model metrics: ${message}` };
  }
}

export async function generateGradCAM(
  file: File
): Promise<{ data: import('../types').GradCAMResponse | null; error: string | null }> {
  try {
    const formData = new FormData();
    formData.append('file', file);

    const res = await fetch(getApiUrl('/api/xai/gradcam'), {
      method: 'POST',
      body: formData,
    });

    if (!res.ok) {
      let errorDetail = `HTTP ${res.status}: ${res.statusText}`;
      try {
        const errorJson = await res.json();
        if (errorJson.detail) {
          errorDetail = typeof errorJson.detail === 'string'
            ? errorJson.detail
            : JSON.stringify(errorJson.detail);
        }
      } catch {
        // ignore
      }
      return { data: null, error: errorDetail };
    }

    const data: import('../types').GradCAMResponse = await res.json();
    return { data, error: null };
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : 'Unknown network error';
    return { data: null, error: `Grad-CAM generation request failed: ${message}` };
  }
}

export async function fetchXAIStatus(): Promise<{
  data: { gradcam_ready: boolean; engine: string } | null;
  error: string | null;
}> {
  try {
    const res = await fetch(getApiUrl('/api/xai/status'), {
      headers: {
        Accept: 'application/json',
      },
    });

    if (!res.ok) {
      return { data: null, error: `HTTP ${res.status}: ${res.statusText}` };
    }

    const data = await res.json();
    return { data, error: null };
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : 'Unknown network error';
    return { data: null, error: `Unable to fetch XAI status: ${message}` };
  }
}

export async function downloadAnalysisReport(
  recordId?: string | null,
  customFilename?: string,
  fallbackPayload?: Record<string, unknown>
): Promise<{ success: boolean; error: string | null }> {
  // 1. Try to fetch from persisted history if recordId is provided
  if (recordId) {
    try {
      const res = await fetch(getApiUrl(`/api/imaging/history/${recordId}/report`));
      if (res.ok) {
        const blob = await res.blob();
        const downloadUrl = window.URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = downloadUrl;
        link.download = customFilename || `MediVision_Report_${recordId}.pdf`;
        document.body.appendChild(link);
        link.click();
        window.URL.revokeObjectURL(downloadUrl);
        document.body.removeChild(link);
        return { success: true, error: null };
      }
    } catch {
      // Continue to direct generation fallback if available
    }
  }

  // 2. Direct generation fallback: synthesizes PDF immediately from current analysis payload
  if (fallbackPayload) {
    try {
      const payloadToSend = {
        ...fallbackPayload,
        id: recordId || fallbackPayload.id || fallbackPayload.recordId || `CASE_${Date.now()}`,
      };
      const res = await fetch(getApiUrl('/api/reports/generate'), {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Accept: 'application/pdf',
        },
        body: JSON.stringify(payloadToSend),
      });

      if (res.ok) {
        const blob = await res.blob();
        const downloadUrl = window.URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = downloadUrl;
        const fname =
          customFilename ||
          `MediVision_Report_${recordId || fallbackPayload.filename || 'analysis'}.pdf`;
        link.download = fname;
        document.body.appendChild(link);
        link.click();
        window.URL.revokeObjectURL(downloadUrl);
        document.body.removeChild(link);
        return { success: true, error: null };
      }

      let errorMsg = `HTTP ${res.status}: ${res.statusText}`;
      try {
        const errJson = await res.json();
        if (errJson.detail) errorMsg = errJson.detail;
      } catch {
        // ignore
      }
      return { success: false, error: errorMsg };
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Unknown network error';
      return { success: false, error: `Direct report generation failed: ${message}` };
    }
  }

  return {
    success: false,
    error: 'Analysis record ID or telemetry payload is required to download report.',
  };
}

export async function fetchBrainMRIStatus(): Promise<{
  data: import('../types').BrainMRIStatusResponse | null;
  error: string | null;
}> {
  try {
    const res = await fetch(getApiUrl('/api/brain-mri/status'), {
      headers: {
        Accept: 'application/json',
      },
    });

    if (!res.ok) {
      return { data: null, error: `HTTP ${res.status}: ${res.statusText}` };
    }

    const data: import('../types').BrainMRIStatusResponse = await res.json();
    return { data, error: null };
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : 'Unknown network error';
    return { data: null, error: `Unable to fetch Brain MRI status: ${message}` };
  }
}

export async function validateBrainMRIStudy(
  file: File
): Promise<{ data: import('../types').BrainMRIValidationResponse | null; error: string | null }> {
  try {
    const formData = new FormData();
    formData.append('file', file);

    const res = await fetch(getApiUrl('/api/brain-mri/validate'), {
      method: 'POST',
      body: formData,
    });

    if (!res.ok) {
      let errorDetail = `HTTP ${res.status}: ${res.statusText}`;
      try {
        const errorJson = await res.json();
        if (errorJson.detail) errorDetail = errorJson.detail;
      } catch {
        // ignore
      }
      return { data: null, error: errorDetail };
    }

    const data: import('../types').BrainMRIValidationResponse = await res.json();
    return { data, error: null };
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : 'Unknown network error';
    return { data: null, error: `Validation request failed: ${message}` };
  }
}

export async function segmentBrainMRIStudy(
  file: File,
  sliceIdx?: number,
  plane: string = 'axial'
): Promise<{ data: import('../types').BrainMRISegmentResponse | null; error: string | null }> {
  try {
    const formData = new FormData();
    formData.append('file', file);
    if (sliceIdx !== undefined && sliceIdx !== null) {
      formData.append('slice_idx', sliceIdx.toString());
    }
    formData.append('plane', plane);

    const res = await fetch(getApiUrl('/api/brain-mri/segment'), {
      method: 'POST',
      body: formData,
    });

    if (!res.ok) {
      let errorDetail = `HTTP ${res.status}: ${res.statusText}`;
      try {
        const errorJson = await res.json();
        if (errorJson.detail) errorDetail = errorJson.detail;
      } catch {
        // ignore
      }
      return { data: null, error: errorDetail };
    }

    const data: import('../types').BrainMRISegmentResponse = await res.json();
    return { data, error: null };
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : 'Unknown network error';
    return { data: null, error: `Segmentation request failed: ${message}` };
  }
}

export async function fetchBrainMRISlice(
  sliceIdx?: number,
  plane: string = 'axial'
): Promise<{ data: import('../types').BrainMRISliceResponse | null; error: string | null }> {
  try {
    const params = new URLSearchParams();
    if (sliceIdx !== undefined && sliceIdx !== null) {
      params.append('slice_idx', sliceIdx.toString());
    }
    params.append('plane', plane);

    const res = await fetch(getApiUrl(`/api/brain-mri/slice?${params.toString()}`), {
      headers: {
        Accept: 'application/json',
      },
    });

    if (!res.ok) {
      let errorDetail = `HTTP ${res.status}: ${res.statusText}`;
      try {
        const errorJson = await res.json();
        if (errorJson.detail) errorDetail = errorJson.detail;
      } catch {
        // ignore
      }
      return { data: null, error: errorDetail };
    }

    const data: import('../types').BrainMRISliceResponse = await res.json();
    return { data, error: null };
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : 'Unknown network error';
    return { data: null, error: `Slice extraction failed: ${message}` };
  }
}

export async function fetchYOLOStatus(): Promise<{ data: YOLODetectionStatusResponse | null; error: string | null }> {
  try {
    const res = await fetch(getApiUrl('/api/yolo/status'), {
      headers: {
        Accept: 'application/json',
      },
    });

    if (!res.ok) {
      return {
        data: null,
        error: `HTTP ${res.status}: ${res.statusText}`,
      };
    }

    const data: YOLODetectionStatusResponse = await res.json();
    return { data, error: null };
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : 'Unknown network error';
    return {
      data: null,
      error: `Failed to fetch YOLO status (${message})`,
    };
  }
}

export async function detectYOLO(
  file: File,
  confThreshold: number = 0.25,
  iouThreshold: number = 0.45
): Promise<{ data: YOLODetectionResponse | null; error: string | null }> {
  try {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('conf_threshold', confThreshold.toString());
    formData.append('iou_threshold', iouThreshold.toString());

    const res = await fetch(getApiUrl('/api/yolo/detect'), {
      method: 'POST',
      body: formData,
    });

    if (!res.ok) {
      let errorDetail = `HTTP ${res.status}: ${res.statusText}`;
      try {
        const errorJson = await res.json();
        if (errorJson.detail) {
          errorDetail = typeof errorJson.detail === 'string' ? errorJson.detail : JSON.stringify(errorJson.detail);
        }
      } catch {
        // ignore
      }
      return { data: null, error: errorDetail };
    }

    const data: YOLODetectionResponse = await res.json();
    return { data, error: null };
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : 'Unknown network error';
    return { data: null, error: `YOLO detection request failed: ${message}` };
  }
}

