import { createContext } from 'react';

import type { usePrediction } from '../hooks/usePrediction';


export type PredictionSessionValue = ReturnType<typeof usePrediction>;

export const PredictionSessionContext = createContext<PredictionSessionValue | null>(null);
