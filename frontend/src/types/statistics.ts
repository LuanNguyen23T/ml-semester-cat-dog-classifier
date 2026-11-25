import type { PredictionResponse } from './prediction';

export interface StatisticsOverview {
  total_predictions: number;
  completed_predictions: number;
  failed_predictions: number;
  pending_predictions: number;
  success_rate: number;
  average_confidence: number | null;
  average_processing_time: number | null;
}

export interface DailyStatistics {
  date: string;
  total: number;
  completed: number;
  failed: number;
  pending: number;
}

export interface MethodStatistics {
  method: string;
  total: number;
  completed: number;
  failed: number;
  success_rate: number;
  average_confidence: number | null;
  average_processing_time: number | null;
}

export interface CategoryStatistics {
  name: string;
  count: number;
}

export interface BreedSpeciesStatistics extends CategoryStatistics {
  species: string;
}

export interface StatisticsResponse {
  overview: StatisticsOverview;
  daily: DailyStatistics[];
  by_method: MethodStatistics[];
  by_species: CategoryStatistics[];
  by_breed_species: BreedSpeciesStatistics[];
  top_breeds: CategoryStatistics[];
  recent_predictions: PredictionResponse[];
  recent_total: number;
  recent_page: number;
  recent_limit: number;
}

export interface StatisticsFilters {
  date_from: string;
  date_to: string;
  method?: string;
  recent_page?: number;
  recent_limit?: number;
}
