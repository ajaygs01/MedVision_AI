import type { ActiveAnalysisCase } from '../types';

const DB_NAME = 'MediVisionDB';
const DB_VERSION = 1;
const STORE_NAME = 'analysis_cases';
const ACTIVE_CASE_KEY = 'active_case';

function openDB(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    if (typeof window === 'undefined' || !window.indexedDB) {
      reject(new Error('IndexedDB not supported in this environment'));
      return;
    }

    const request = window.indexedDB.open(DB_NAME, DB_VERSION);

    request.onupgradeneeded = (event) => {
      const db = (event.target as IDBOpenDBRequest).result;
      if (!db.objectStoreNames.contains(STORE_NAME)) {
        db.createObjectStore(STORE_NAME);
      }
    };

    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error || new Error('Failed to open IndexedDB'));
  });
}

export async function saveCaseToIndexedDB(caseData: ActiveAnalysisCase): Promise<void> {
  try {
    const db = await openDB();
    return new Promise((resolve, reject) => {
      const tx = db.transaction(STORE_NAME, 'readwrite');
      const store = tx.objectStore(STORE_NAME);
      const req = store.put(caseData, ACTIVE_CASE_KEY);
      req.onsuccess = () => resolve();
      req.onerror = () => reject(req.error || new Error('IndexedDB put failed'));
      tx.oncomplete = () => db.close();
    });
  } catch (err) {
    console.warn('[Storage] Failed to save case to IndexedDB fallback:', err);
  }
}

export async function loadCaseFromIndexedDB(): Promise<ActiveAnalysisCase | null> {
  try {
    const db = await openDB();
    return new Promise((resolve) => {
      const tx = db.transaction(STORE_NAME, 'readonly');
      const store = tx.objectStore(STORE_NAME);
      const req = store.get(ACTIVE_CASE_KEY);
      req.onsuccess = () => {
        resolve(req.result || null);
      };
      req.onerror = () => resolve(null);
      tx.oncomplete = () => db.close();
    });
  } catch (err) {
    console.warn('[Storage] Failed to load case from IndexedDB fallback:', err);
    return null;
  }
}

export async function clearCaseFromIndexedDB(): Promise<void> {
  try {
    const db = await openDB();
    return new Promise((resolve) => {
      const tx = db.transaction(STORE_NAME, 'readwrite');
      const store = tx.objectStore(STORE_NAME);
      const req = store.delete(ACTIVE_CASE_KEY);
      req.onsuccess = () => resolve();
      req.onerror = () => resolve();
      tx.oncomplete = () => db.close();
    });
  } catch (err) {
    console.warn('[Storage] Failed to clear case from IndexedDB:', err);
  }
}
