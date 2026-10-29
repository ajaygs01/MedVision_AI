import { createContext } from 'react';
import type {
  ActiveAnalysisCase,
  ActiveBrainMRICase,
  ActiveClinicalCase,
  ActiveYOLOCase,
  PredictionHistoryItem,
} from '../types';

export interface AnalysisContextType {
  activeCase: ActiveAnalysisCase | null;
  saveActiveCase: (data: ActiveAnalysisCase) => Promise<void>;
  updateViewMode: (mode: 'overlay' | 'heatmap' | 'original' | 'split') => void;
  clearActiveCase: () => Promise<void>;
  reopenHistoricalCase: (item: PredictionHistoryItem) => Promise<void>;
  isRestored: boolean;

  // Clinical ML state persistence across navigation
  activeClinicalCase: ActiveClinicalCase | null;
  saveActiveClinicalCase: (data: ActiveClinicalCase) => void;
  clearActiveClinicalCase: () => void;

  // Brain MRI state persistence across navigation
  activeBrainMRICase: ActiveBrainMRICase | null;
  saveActiveBrainMRICase: (data: ActiveBrainMRICase) => void;
  clearActiveBrainMRICase: () => void;

  // YOLO Detection state persistence across navigation
  activeYOLOCase: ActiveYOLOCase | null;
  saveActiveYOLOCase: (data: ActiveYOLOCase) => void;
  clearActiveYOLOCase: () => void;
}

export const AnalysisContext = createContext<AnalysisContextType | undefined>(undefined);


