import type { HealthResponse, ModelsStatusResponse, ChestXRayClassifyResponse } from '../types';

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
