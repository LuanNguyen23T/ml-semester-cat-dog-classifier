import { useState, useEffect, useRef, useCallback } from "react";
import axios from "axios";
import { formatApiError, predictionApi } from "../services/predictionApi";
import type { PredictionResponse, PredictionStatus } from "../types/prediction";

const POLL_INTERVAL_MS = 2000;
const STORAGE_KEY = "pet-classifier:prediction-ids";
const TERMINAL_STATUSES: PredictionStatus[] = ["COMPLETED", "FAILED"];

function readStoredPredictionIds(): string[] {
  try {
    const value = localStorage.getItem(STORAGE_KEY);
    if (!value) return [];
    const parsed: unknown = JSON.parse(value);
    return Array.isArray(parsed) && parsed.every((id) => typeof id === "string")
      ? parsed
      : [];
  } catch {
    return [];
  }
}

function storePredictionIds(ids: string[]): void {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(ids));
  } catch {
    // Polling still works for this tab when browser storage is unavailable.
  }
}

function clearStoredPredictionIds(): void {
  try {
    localStorage.removeItem(STORAGE_KEY);
  } catch {
    // Nothing else is required when browser storage is unavailable.
  }
}

interface UsePredictionReturn {
  predictions: PredictionResponse[];
  activeIds: string[];
  isPolling: boolean;
  error: string | null;
  startPolling: (ids: string[]) => void;
  stopPolling: () => void;
  reset: () => void;
}

/**
 * Hook that polls GET /predictions/{id} every 2s for multiple IDs until
 * all statuses become COMPLETED or FAILED.
 */
export function usePrediction(): UsePredictionReturn {
  const [predictions, setPredictions] = useState<PredictionResponse[]>([]);
  const [activeIds, setActiveIds] = useState<string[]>(readStoredPredictionIds);
  const [isPolling, setIsPolling] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const predictionIdsRef = useRef<string[]>(activeIds);

  const stopPolling = useCallback(() => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
    setIsPolling(false);
  }, []);

  const fetchPredictions = useCallback(
    async (ids: string[]) => {
      try {
        const results = await Promise.all(
          ids.map((id) => predictionApi.getPrediction(id)),
        );
        setPredictions(results);
        setError(null);

        // Check if all are in terminal status
        const allTerminal = results.every((r) =>
          TERMINAL_STATUSES.includes(r.status),
        );
        if (allTerminal && results.length > 0) {
          stopPolling();
        }
      } catch (err: unknown) {
        const message = formatApiError(err, "Không thể lấy kết quả dự đoán từ máy chủ.");
        setError(message);
        if (axios.isAxiosError(err) && err.response?.status === 404) {
          stopPolling();
        }
      }
    },
    [stopPolling],
  );

  const startPolling = useCallback(
    (ids: string[]) => {
      // Stop any existing poll
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }

      predictionIdsRef.current = ids;
      setActiveIds(ids);
      setPredictions([]);
      setError(null);
      setIsPolling(true);
      storePredictionIds(ids);

      // Fetch immediately
      fetchPredictions(ids);

      // Then poll every 2s
      intervalRef.current = setInterval(() => {
        fetchPredictions(ids);
      }, POLL_INTERVAL_MS);
    },
    [fetchPredictions],
  );

  const reset = useCallback(() => {
    stopPolling();
    setPredictions([]);
    setError(null);
    predictionIdsRef.current = [];
    setActiveIds([]);
    clearStoredPredictionIds();
  }, [stopPolling]);

  // Resume an unfinished or completed session after a full page refresh.
  useEffect(() => {
    if (predictionIdsRef.current.length > 0) {
      startPolling(predictionIdsRef.current);
    }

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, [startPolling]);

  return {
    predictions,
    activeIds,
    isPolling,
    error,
    startPolling,
    stopPolling,
    reset,
  };
}
