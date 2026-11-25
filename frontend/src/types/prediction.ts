// TypeScript interfaces matching the backend API schemas

export type PredictionStatus = 'UPLOADED' | 'PROCESSING' | 'COMPLETED' | 'FAILED';

export type SingleModelMethod =
  | 'resnet18'
  | 'seresnet18'
  | 'densenet121'
  | 'efficientnet-b0'
  | 'dual-stream-efficientnet';

export type ModelMethod = 'all' | SingleModelMethod;
export type ModelStreamGroup = 'dual-stream' | 'single-stream';

export const DEFAULT_MODEL_METHOD: SingleModelMethod = 'dual-stream-efficientnet';

export interface ModelInfo {
  id: SingleModelMethod;
  name: string;
  description: string;
  accuracy: number;
  accuracy_single?: number;
  accuracy_tta?: number;
  params: number;
  recommended?: boolean;
  is_default?: boolean;
  tag: string;
  available: boolean;
}

export const FALLBACK_MODELS: ModelInfo[] = [
  {
    id: 'dual-stream-efficientnet',
    name: 'Dual-Stream EfficientNet-B0',
    description: 'Mô hình 2 luồng (Thân + Mặt) với cơ chế cổng Gated Fusion, đạt độ chính xác cao nhất.',
    accuracy: 85.72,
    accuracy_single: 85.72,
    accuracy_tta: 86.54,
    params: 5042081,
    recommended: true,
    is_default: true,
    tag: 'Recommended',
    available: true,
  },
  {
    id: 'efficientnet-b0',
    name: 'EfficientNet-B0',
    description: 'Mạng MBConv + Squeeze-and-Excitation gọn nhẹ, cân bằng tốt giữa tốc độ và độ chính xác.',
    accuracy: 83.32,
    accuracy_single: 83.32,
    accuracy_tta: 83.78,
    params: 4045605,
    recommended: false,
    is_default: false,
    tag: 'Fast & Accurate',
    available: true,
  },
  {
    id: 'densenet121',
    name: 'DenseNet-121',
    description: 'Mạng kết nối dày đặc tái sử dụng đặc trưng đa tầng hiệu quả.',
    accuracy: 74.54,
    accuracy_single: 74.54,
    accuracy_tta: 75.33,
    params: 6991781,
    recommended: false,
    is_default: false,
    tag: 'Dense Features',
    available: true,
  },
  {
    id: 'seresnet18',
    name: 'SEResNet-18',
    description: 'ResNet-18 tích hợp khối chú ý kênh Squeeze-and-Excitation.',
    accuracy: 70.62,
    accuracy_single: 70.62,
    accuracy_tta: 71.55,
    params: 11282533,
    recommended: false,
    is_default: false,
    tag: 'Channel Attention',
    available: true,
  },
  {
    id: 'resnet18',
    name: 'ResNet-18',
    description: 'Mạng tích chập phần dư cổ điển làm mốc đối chiếu cơ sở.',
    accuracy: 67.84,
    accuracy_single: 67.84,
    accuracy_tta: 68.47,
    params: 11195493,
    recommended: false,
    is_default: false,
    tag: 'Baseline',
    available: true,
  },
];

export const MODEL_LABEL_MAP: Record<SingleModelMethod, string> = {
  'dual-stream-efficientnet': 'Dual-Stream EfficientNet-B0',
  'efficientnet-b0': 'EfficientNet-B0',
  'densenet121': 'DenseNet-121',
  'seresnet18': 'SEResNet-18',
  'resnet18': 'ResNet-18',
};

export function getModelStreamGroup(method: SingleModelMethod | string): ModelStreamGroup {
  return method === 'dual-stream-efficientnet' ? 'dual-stream' : 'single-stream';
}

export function getModelDisplayName(
  method?: string | null,
  modelNames?: Map<string, string>,
): string {
  if (!method) return 'Không rõ';
  if (modelNames?.has(method)) return modelNames.get(method)!;
  if (method in MODEL_LABEL_MAP) {
    return MODEL_LABEL_MAP[method as SingleModelMethod];
  }
  return method
    .split('-')
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(' ');
}

export function hasTtaEnabled(modelVersion?: string | null): boolean {
  return Boolean(modelVersion && modelVersion.includes('+tta'));
}

export function formatBreedName(name?: string | null): string {
  if (!name) return '—';
  return name.replace(/_/g, ' ').replace(/\b\w/g, (char) => char.toUpperCase());
}

export interface SpeciesResponse {
  id: number;
  name: string;
}

export interface BreedResponse {
  id: number;
  name: string;
  species_id: number;
  species: SpeciesResponse;
}

export interface TopKResult {
  breed_id: number;
  breed_name: string;
  species: string;
  score: number;
}

export interface PredictionCreateResponse {
  id: string;
  batch_id: string | null;
  status: PredictionStatus;
  method: ModelMethod | null;
  note?: string | null;
}

export interface PredictionResponse {
  id: string;
  batch_id: string | null;
  image_key: string;
  predicted_breed: BreedResponse | null;
  score: number | null;
  top_k_results: TopKResult[] | null;
  model_version: string | null;
  method: SingleModelMethod | string | null;
  processing_time?: number;
  status: PredictionStatus;
  note?: string | null;
  error_message: string | null;
  created_at: string;
  is_unknown_breed?: boolean;
  unknown_score?: number;
  breed_status?: string;
  unknown_assessment?: {
    is_unknown: boolean;
    breed_status: string;
    status_label_vi: string;
    unknown_score: number;
    known_confidence: number;
    summary: string;
  } | null;
}

export interface PredictionHistoryGroup {
  group_id: string;
  batch_id: string | null;
  image_key: string;
  note?: string | null;
  created_at: string;
  predictions: PredictionResponse[];
}

export interface PredictionHistoryResponse {
  items: PredictionHistoryGroup[];
  total: number;
  page: number;
  limit: number;
}

export interface PredictionsListResponse {
  items: PredictionResponse[];
  total: number;
  page: number;
  limit: number;
}
