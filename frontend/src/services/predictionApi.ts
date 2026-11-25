import axios from 'axios';
import {
  DEFAULT_MODEL_METHOD,
  type BreedResponse,
  type ModelInfo,
  type PredictionCreateResponse,
  type PredictionHistoryResponse,
  type PredictionResponse,
  type SingleModelMethod,
  type SpeciesResponse,
} from '../types/prediction';

const api = axios.create({
  baseURL: '/api/v1',
  timeout: 30000,
});

export function formatApiError(
  error: unknown,
  fallback = 'Đã xảy ra lỗi khi kết nối với hệ thống.',
): string {
  if (!axios.isAxiosError(error)) {
    return error instanceof Error ? error.message : fallback;
  }

  if (!error.response) {
    if (error.code === 'ECONNABORTED') {
      return 'Kết nối tới máy chủ quá thời gian chờ. Vui lòng thử lại.';
    }
    return 'Mất kết nối tới máy chủ backend. Vui lòng kiểm tra backend đang chạy và thử lại.';
  }

  const status = error.response.status;
  const rawData = error.response.data;
  const detail =
    typeof rawData === 'object' && rawData !== null && 'detail' in rawData
      ? String((rawData as { detail: unknown }).detail ?? '')
      : '';

  if (status === 413 || detail.includes('10 MB')) {
    return 'Ảnh vượt quá dung lượng tối đa 10 MB. Vui lòng chọn ảnh nhỏ hơn.';
  }

  if (status === 400) {
    if (detail.includes('Unsupported file type')) {
      return 'Sai loại file. Chỉ hỗ trợ ảnh JPG, PNG hoặc WebP.';
    }
    if (detail.includes('dimensions must not exceed')) {
      return 'Kích thước ảnh quá lớn (cạnh dài tối đa 8000 px). Vui lòng chọn ảnh nhỏ hơn.';
    }
    if (detail.includes('Invalid or corrupted') || detail.includes('Empty image')) {
      return 'File không phải ảnh hợp lệ hoặc bị hỏng. Vui lòng chọn ảnh JPG, PNG hoặc WebP hợp lệ.';
    }
    return detail ? `Yêu cầu không hợp lệ: ${detail}` : 'Ảnh tải lên không hợp lệ.';
  }

  if (status === 503) {
    return 'Mô hình AI được chọn hiện chưa sẵn sàng (thiếu file trọng số trên máy chủ). Vui lòng chọn mô hình khác.';
  }

  if (status === 502 || status === 504 || (status === 500 && !detail)) {
    return 'Mất kết nối tới máy chủ backend. Vui lòng kiểm tra backend đang chạy và thử lại.';
  }

  if (detail) {
    return `Lỗi hệ thống (${status}): ${detail}`;
  }

  return fallback;
}

export const predictionApi = {
  /**
   * Upload an image and create a new prediction record.
   * Returns immediately with id + status=UPLOADED.
   */
  createPrediction: async (
    file: File,
    method: SingleModelMethod = DEFAULT_MODEL_METHOD,
    tta = false,
  ): Promise<PredictionCreateResponse> => {
    const formData = new FormData();
    formData.append('file', file);
    const { data } = await api.post<PredictionCreateResponse>('/predictions', formData, {
      params: { method, tta },
      headers: { 'Content-Type': 'multipart/form-data' },
    });
    return data;
  },

  /**
   * Upload an image and create multiple prediction records (one for each model).
   */
  createPredictionBatch: async (file: File, tta = false): Promise<PredictionCreateResponse[]> => {
    const formData = new FormData();
    formData.append('file', file);
    const { data } = await api.post<PredictionCreateResponse[]>('/predictions/batch', formData, {
      params: { tta },
      headers: { 'Content-Type': 'multipart/form-data' },
    });
    return data;
  },

  /**
   * Get a single prediction by its UUID.
   */
  getPrediction: async (id: string): Promise<PredictionResponse> => {
    const { data } = await api.get<PredictionResponse>(`/predictions/${id}`);
    return data;
  },

  /**
   * List predictions with optional filters and pagination.
   */
  listPredictions: async (params?: {
    breed_id?: number;
    status?: string;
    method?: string;
    skip?: number;
    limit?: number;
  }): Promise<PredictionResponse[]> => {
    const { data } = await api.get<PredictionResponse[]>('/predictions', { params });
    return data;
  },

  listPredictionHistory: async (params?: {
    status?: string;
    method?: string;
    page?: number;
    limit?: number;
  }): Promise<PredictionHistoryResponse> => {
    const { data } = await api.get<PredictionHistoryResponse>('/prediction-history', { params });
    return data;
  },

  /**
   * List available classification models
   */
  listModels: async (): Promise<ModelInfo[]> => {
    const { data } = await api.get<ModelInfo[]>('/models');
    return data;
  },
};

export const breedApi = {
  /**
   * List all breeds, optionally filtered by species name.
   */
  listBreeds: async (species?: string): Promise<BreedResponse[]> => {
    const { data } = await api.get<BreedResponse[]>('/breeds', {
      params: species ? { species } : undefined,
    });
    return data;
  },

  /**
   * List all species (Cat / Dog).
   */
  listSpecies: async (): Promise<SpeciesResponse[]> => {
    const { data } = await api.get<SpeciesResponse[]>('/species');
    return data;
  },
};
