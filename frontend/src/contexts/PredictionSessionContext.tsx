import type { ReactNode } from 'react';

import { usePrediction } from '../hooks/usePrediction';
import { PredictionSessionContext } from './predictionSessionStore';


export function PredictionSessionProvider({ children }: { children: ReactNode }) {
  const session = usePrediction();

  return (
    <PredictionSessionContext.Provider value={session}>
      {children}
    </PredictionSessionContext.Provider>
  );
}
