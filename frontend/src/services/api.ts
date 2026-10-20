import type { HealthResponse, ModelsStatusResponse } from '../types';

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
