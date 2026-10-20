import type { HealthResponse, ModelsStatusResponse, ChestXRayClassifyResponse } from '../types';

const API_BASE = import.meta.env.VITE_API_BASE_URL || '';

export async function fetchHealth(): Promise<{ data: HealthResponse | null; error: string | null }> {
  try {
    const res = await fetch(`${API_BASE}/api/health`, {
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
      error: `Backend unreachable (${message}). Verify FastAPI server is running on port 8000.`,
    };
  }
}

export async function fetchModelsStatus(): Promise<{ data: ModelsStatusResponse | null; error: string | null }> {
  try {
    const res = await fetch(`${API_BASE}/api/models/status`, {
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
      error: `Unable to load model statuses (${message})`,
    };
  }
}

export async function classifyChestXRay(
  file: File
): Promise<{ data: ChestXRayClassifyResponse | null; error: string | null }> {
  try {
    const formData = new FormData();
    formData.append('file', file);

    const res = await fetch(`${API_BASE}/api/imaging/chest-xray/classify`, {
      method: 'POST',
      body: formData,
    });

    if (!res.ok) {
      let errorDetail = `HTTP ${res.status}: ${res.statusText}`;
      try {
        const errorJson = await res.json();
        if (errorJson.detail) {
          errorDetail = errorJson.detail;
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
      error: `Classification request failed (${message}). Ensure backend server is running.`,
    };
  }
}
