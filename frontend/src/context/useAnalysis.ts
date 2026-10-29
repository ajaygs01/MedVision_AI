import { useContext } from 'react';
import { AnalysisContext, type AnalysisContextType } from './AnalysisContextDefinition';

export function useAnalysis(): AnalysisContextType {
  const context = useContext(AnalysisContext);
  if (!context) {
    throw new Error('useAnalysis must be used within an AnalysisProvider');
  }
  return context;
}
