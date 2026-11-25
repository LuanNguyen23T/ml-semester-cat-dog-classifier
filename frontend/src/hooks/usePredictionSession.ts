import { useContext } from 'react';

import { PredictionSessionContext } from '../contexts/predictionSessionStore';
import type { PredictionSessionValue } from '../contexts/predictionSessionStore';


export function usePredictionSession(): PredictionSessionValue {
  const session = useContext(PredictionSessionContext);
  if (!session) {
    throw new Error('usePredictionSession must be used within PredictionSessionProvider');
  }
  return session;
}
