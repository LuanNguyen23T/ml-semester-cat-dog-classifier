import { useCallback, useEffect, useMemo, useState } from 'react';
import {
  Activity,
  BarChart3,
  CheckCircle2,
  Clock3,
  ChevronLeft,
  ChevronRight,
  Gauge,
  RefreshCw,
  X,
  Trash2,
  Loader2,
} from 'lucide-react';
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';

import { formatApiError, predictionApi } from '../../services/predictionApi';
import { statisticsApi } from '../../services/statisticsApi';
import {
  FALLBACK_MODELS,
  formatBreedName,
  getModelDisplayName,
  hasTtaEnabled,
  type ModelInfo,
  type PredictionResponse,
} from '../../types/prediction';
import type { StatisticsFilters, StatisticsResponse } from '../../types/statistics';
import './statistics.css';


const CHART_COLORS = ['#2563eb', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6'];
const CAT_COLORS = ['#bfdbfe', '#bae6fd', '#a7f3d0', '#fde68a', '#fecdd3'];
const DOG_COLORS = ['#1e3a8a', '#075985', '#065f46', '#92400e', '#9f1239'];
const RECENT_PAGE_SIZE = 10;

function toInputDate(date: Date): string {
  const year = date.getFullYear();
  const month = String(date.getMonth() + 1).padStart(2, '0');
  const day = String(date.getDate()).padStart(2, '0');
  return `${year}-${month}-${day}`;
}

function getInitialFilters(): StatisticsFilters {
  const end = new Date();
  const start = new Date(end);
  start.setDate(start.getDate() - 29);
  return { date_from: toInputDate(start), date_to: toInputDate(end) };
}

function formatPercent(value: number | null): string {
  return value === null ? 'No data' : `${(value * 100).toFixed(1)}%`;
}

function formatDuration(value?: number | null): string {
  if (value === null || value === undefined) return 'No data';
  return value < 1 ? `${Math.round(value * 1000)} ms` : `${value.toFixed(2)} s`;
}

function formatDateTime(value: string): string {
  return new Intl.DateTimeFormat('en-US', {
    month: 'short',
    day: 'numeric',
    year: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  }).format(new Date(value));
}

function displayMethod(method: string | null, modelNames: Map<string, string>): string {
  if (!method) return 'Legacy record';
  return getModelDisplayName(method, modelNames);
}

function ChartEmptyState() {
  return (
    <div className="statistics-chart-empty">
      <BarChart3 size={28} />
      <span>No data for this period</span>
    </div>
  );
}

interface PredictionDetailProps {
  prediction: PredictionResponse;
  modelName: string;
  onClose: () => void;
  onDeleted?: (id: string) => void;
}

function PredictionDetail({ prediction, modelName, onClose, onDeleted }: PredictionDetailProps) {
  const [confirmDelete, setConfirmDelete] = useState(false);
  const [isDeleting, setIsDeleting] = useState(false);
  const [deleteError, setDeleteError] = useState<string | null>(null);

  const handleDelete = async () => {
    setIsDeleting(true);
    setDeleteError(null);
    try {
      await predictionApi.deletePrediction(prediction.id, true);
      onDeleted?.(prediction.id);
    } catch (err: unknown) {
      setDeleteError(formatApiError(err, 'Xóa thất bại. Vui lòng thử lại.'));
      setIsDeleting(false);
    }
  };

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [onClose]);

  return (
    <div className="statistics-modal-backdrop" onMouseDown={onClose} role="presentation">
      <section
        className="statistics-modal"
        role="dialog"
        aria-modal="true"
        aria-labelledby="prediction-detail-title"
        onMouseDown={(event) => event.stopPropagation()}
      >
        <header className="statistics-modal-header">
          <div>
            <span className="statistics-eyebrow">Prediction details</span>
            <h2 id="prediction-detail-title" style={{ color: prediction.is_unknown_breed ? '#f59e0b' : 'inherit' }}>
              {prediction.is_unknown_breed ? 'Chưa xác định' : (prediction.predicted_breed?.name ? formatBreedName(prediction.predicted_breed.name) : 'Pending result')}
            </h2>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            {confirmDelete ? (
              <div
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: 6,
                  background: 'rgba(239,68,68,0.1)',
                  padding: '4px 8px',
                  borderRadius: 8,
                  border: '1px solid rgba(239,68,68,0.3)',
                }}
              >
                <span style={{ fontSize: 12, color: 'var(--color-error)', fontWeight: 600 }}>Xác nhận xóa?</span>
                <button
                  className="btn btn-ghost"
                  onClick={handleDelete}
                  disabled={isDeleting}
                  style={{
                    padding: '4px 10px',
                    fontSize: 12,
                    borderRadius: 6,
                    background: 'var(--color-error)',
                    color: '#ffffff',
                    fontWeight: 700,
                    display: 'inline-flex',
                    alignItems: 'center',
                    gap: 4,
                  }}
                >
                  {isDeleting ? <Loader2 size={12} className="animate-spin" /> : 'Xóa'}
                </button>
                <button
                  className="btn btn-ghost"
                  onClick={() => setConfirmDelete(false)}
                  disabled={isDeleting}
                  style={{ padding: '4px 8px', fontSize: 12, borderRadius: 6 }}
                >
                  Hủy
                </button>
              </div>
            ) : (
              <button
                className="btn btn-ghost"
                onClick={() => setConfirmDelete(true)}
                style={{
                  padding: '6px 12px',
                  fontSize: 12,
                  borderRadius: 8,
                  color: 'var(--color-error)',
                  border: '1px solid rgba(239,68,68,0.25)',
                  display: 'inline-flex',
                  alignItems: 'center',
                  gap: 6,
                  fontWeight: 600,
                }}
                title="Xóa mẫu này"
              >
                <Trash2 size={14} />
                Xóa
              </button>
            )}
            <button className="statistics-icon-button" onClick={onClose} aria-label="Close details">
              <X size={19} />
            </button>
          </div>
        </header>

        {deleteError && (
          <div style={{ padding: '8px 16px', color: 'var(--color-error)', fontSize: 13, fontWeight: 600 }}>
            ⚠ {deleteError}
          </div>
        )}

        <div className="statistics-modal-content">
          <img src={prediction.image_key} alt="Classified pet" className="statistics-detail-image" />
          <dl className="statistics-detail-grid">
            <div><dt>Status</dt><dd><span className={`badge badge-${prediction.status.toLowerCase()}`}>{prediction.status}</span></dd></div>
            <div><dt>Model</dt><dd>{modelName}</dd></div>
            <div><dt>TTA Mode</dt><dd>{hasTtaEnabled(prediction.model_version) ? 'Bật (4-View)' : 'Tắt'}</dd></div>
            <div><dt>Species</dt><dd>{prediction.is_unknown_breed ? 'Không xác định' : (prediction.predicted_breed?.species.name ?? 'Not available')}</dd></div>
            <div><dt>Confidence</dt><dd>{formatPercent(prediction.score)}</dd></div>
            <div><dt>Processing time</dt><dd>{formatDuration(prediction.processing_time)}</dd></div>
            <div><dt>Created</dt><dd>{formatDateTime(prediction.created_at)}</dd></div>
            {prediction.is_unknown_breed && (
              <div><dt>OOD Alert</dt><dd style={{ color: '#f59e0b', fontWeight: 700 }}>Chưa xác định</dd></div>
            )}
          </dl>
        </div>

        {prediction.top_k_results && prediction.top_k_results.length > 0 && (
          <div className="statistics-ranked-results">
            <h3>Top predictions {prediction.is_unknown_breed && <small style={{ color: 'var(--color-text-muted)', fontSize: 12, fontWeight: 400 }}>(Tương đồng gần nhất)</small>}</h3>
            {prediction.top_k_results.filter(r => r && r.breed_name).slice(0, 5).map((result, index) => (
              <div className="statistics-ranked-row" key={`${result.breed_id}-${index}`}>
                <span>{index + 1}</span>
                <strong>{formatBreedName(result.breed_name)}</strong>
                <div><i style={{ width: `${Math.max(2, result.score * 100)}%` }} /></div>
                <b>{(result.score * 100).toFixed(1)}%</b>
              </div>
            ))}
          </div>
        )}

        {prediction.error_message && (
          <div className="statistics-error-message">
            <strong>Error details</strong>
            <span>{prediction.error_message}</span>
          </div>
        )}
      </section>
    </div>
  );
}

export function StatisticsPage() {
  const initialFilters = useMemo(() => getInitialFilters(), []);
  const [filters, setFilters] = useState<StatisticsFilters>(initialFilters);
  const [recentPage, setRecentPage] = useState(1);
  const [data, setData] = useState<StatisticsResponse | null>(null);
  const [models, setModels] = useState<ModelInfo[]>(FALLBACK_MODELS);
  const [selectedPrediction, setSelectedPrediction] = useState<PredictionResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const modelNames = useMemo(
    () => new Map(models.map((model) => [model.id, model.name])),
    [models],
  );

  const fetchStatistics = useCallback(async () => {
    if (filters.date_from > filters.date_to) {
      setError('Ngày bắt đầu phải nhỏ hơn hoặc bằng ngày kết thúc.');
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const response = await statisticsApi.getStatistics({
        ...filters,
        method: filters.method || undefined,
        recent_page: recentPage,
        recent_limit: RECENT_PAGE_SIZE,
      });
      setData(response);
    } catch (requestError: unknown) {
      setError(formatApiError(requestError, 'Không thể tải dữ liệu thống kê.'));
    } finally {
      setLoading(false);
    }
  }, [filters, recentPage]);

  useEffect(() => {
    predictionApi
      .listModels()
      .then((res) => {
        if (res.length > 0) setModels(res);
      })
      .catch(() => setModels(FALLBACK_MODELS));
  }, []);

  useEffect(() => {
    // Synchronize the dashboard with the current server-side filters.
    // oxlint-disable-next-line react/set-state-in-effect
    fetchStatistics();
  }, [fetchStatistics]);

  const tooltipStyle = {
    background: 'var(--color-surface)',
    border: '1px solid var(--color-border)',
    borderRadius: 8,
    color: 'var(--color-text)',
    boxShadow: 'var(--shadow-md)',
  };

  const hasSpeciesData = (data?.by_species.length ?? 0) > 0;
  const hasBreedSpeciesData = (data?.by_breed_species.length ?? 0) > 0;
  const hasBreedData = (data?.top_breeds.length ?? 0) > 0;
  const recentPageCount = data ? Math.max(1, Math.ceil(data.recent_total / data.recent_limit)) : 1;

  return (
    <div className="statistics-page">
      <header className="statistics-page-header">
        <div>
          <div className="statistics-title-row">
            <BarChart3 size={24} />
            <h1>Classification Statistics</h1>
          </div>
          <p>Monitor model-run volume, reliability, confidence, and inference speed.</p>
        </div>
        <button className="btn btn-ghost" onClick={fetchStatistics} disabled={loading}>
          <RefreshCw size={15} className={loading ? 'animate-spin' : ''} />
          Refresh
        </button>
      </header>

      <div className="statistics-filters">
        <label>
          <span>From</span>
          <input
            type="date"
            value={filters.date_from}
            max={filters.date_to}
            onChange={(event) => { setRecentPage(1); setFilters((current) => ({ ...current, date_from: event.target.value })); }}
          />
        </label>
        <label>
          <span>To</span>
          <input
            type="date"
            value={filters.date_to}
            min={filters.date_from}
            onChange={(event) => { setRecentPage(1); setFilters((current) => ({ ...current, date_to: event.target.value })); }}
          />
        </label>
        <label className="statistics-model-filter">
          <span>Model</span>
          <select value={filters.method ?? ''} onChange={(event) => {
            setRecentPage(1);
            setFilters((current) => ({ ...current, method: event.target.value || undefined }));
          }}>
            <option value="">Tất cả mô hình (5)</option>
            {models.map((model) => <option value={model.id} key={model.id}>{model.name}</option>)}
          </select>
        </label>
      </div>

      {error && (
        <div className="statistics-alert" role="alert">
          <span>{error}</span>
          <button onClick={fetchStatistics}>Try again</button>
        </div>
      )}

      {loading && !data ? (
        <div className="statistics-loading-grid">
          {Array.from({ length: 8 }).map((_, index) => <div className="skeleton" key={index} />)}
        </div>
      ) : data && (
        <>
          <section className="statistics-kpi-grid" aria-label="Overview">
            <article className="statistics-kpi-card kpi-blue">
              <div className="statistics-kpi-icon"><Activity size={20} /></div>
              <div><span>Total model runs</span><strong>{data.overview.total_predictions.toLocaleString()}</strong><small>{data.overview.completed_predictions.toLocaleString()} completed, {data.overview.pending_predictions.toLocaleString()} pending</small></div>
            </article>
            <article className="statistics-kpi-card kpi-green">
              <div className="statistics-kpi-icon"><CheckCircle2 size={20} /></div>
              <div><span>Run success rate</span><strong>{data.overview.success_rate.toFixed(1)}%</strong><small>{data.overview.failed_predictions.toLocaleString()} failed among finished runs</small></div>
            </article>
            <article className="statistics-kpi-card kpi-amber">
              <div className="statistics-kpi-icon"><Gauge size={20} /></div>
              <div><span>Average confidence</span><strong>{formatPercent(data.overview.average_confidence)}</strong><small>Completed model runs</small></div>
            </article>
            <article className="statistics-kpi-card kpi-violet">
              <div className="statistics-kpi-icon"><Clock3 size={20} /></div>
              <div><span>Average inference time</span><strong>{formatDuration(data.overview.average_processing_time)}</strong><small>Classifier execution only</small></div>
            </article>
          </section>

          <section className="statistics-chart-grid">
            <article className="statistics-panel statistics-panel-wide">
              <div className="statistics-panel-heading"><div><h2>Species distribution</h2><p>Completed model outputs by species</p></div></div>
              <div className="statistics-species-layout">
                <div className="statistics-chart statistics-species-total">
                {!hasSpeciesData ? <ChartEmptyState /> : (
                  <ResponsiveContainer width="100%" height="100%">
                    <PieChart>
                      <Pie data={data.by_species} dataKey="count" nameKey="name" innerRadius={58} outerRadius={88} paddingAngle={3}>
                        {data.by_species.map((entry, index) => <Cell key={entry.name} fill={CHART_COLORS[index % CHART_COLORS.length]} />)}
                      </Pie>
                      <Tooltip contentStyle={tooltipStyle} />
                      <Legend iconType="circle" wrapperStyle={{ fontSize: 12 }} />
                    </PieChart>
                  </ResponsiveContainer>
                )}
                </div>
                <div className="statistics-species-breakdown">
                  <div className="statistics-breakdown-heading">
                    <h3>Breed breakdown</h3>
                  </div>
                  <div className="statistics-breakdown-scroll">
                  <div className="statistics-breakdown-chart" style={{ height: Math.max(310, data.by_breed_species.length * 27) }}>
                    {!hasBreedSpeciesData ? <ChartEmptyState /> : (
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart data={data.by_breed_species} layout="vertical" margin={{ top: 4, right: 18, left: 18, bottom: 4 }}>
                          <CartesianGrid stroke="var(--color-border)" strokeDasharray="3 3" horizontal={false} />
                          <XAxis type="number" allowDecimals={false} tick={{ fill: 'var(--color-text-muted)', fontSize: 11 }} axisLine={false} tickLine={false} />
                          <YAxis type="category" dataKey="name" width={125} tick={{ fill: 'var(--color-text-muted)', fontSize: 10 }} axisLine={false} tickLine={false} />
                          <Tooltip contentStyle={tooltipStyle} formatter={(value, _name, item) => [value, item.payload.species]} />
                          <Bar dataKey="count" name="Predictions" radius={[0, 4, 4, 0]} maxBarSize={18}>
                            {data.by_breed_species.map((entry, index) => (
                              <Cell
                                key={`${entry.species}-${entry.name}`}
                                fill={(entry.species.toLowerCase() === 'cat' ? CAT_COLORS : DOG_COLORS)[index % 5]}
                              />
                            ))}
                          </Bar>
                        </BarChart>
                      </ResponsiveContainer>
                    )}
                  </div>
                  </div>
                </div>
              </div>
            </article>

            <article className="statistics-panel">
              <div className="statistics-panel-heading"><div><h2>Run reliability</h2><p>Successful completion rate by classification method</p></div></div>
              <div className="statistics-chart">
                {data.by_method.length === 0 ? <ChartEmptyState /> : (
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={data.by_method.map((item) => ({ ...item, label: displayMethod(item.method, modelNames) }))} layout="vertical" margin={{ top: 4, right: 18, left: 8, bottom: 0 }}>
                      <CartesianGrid stroke="var(--color-border)" strokeDasharray="3 3" horizontal={false} />
                      <XAxis type="number" domain={[0, 100]} tick={{ fill: 'var(--color-text-muted)', fontSize: 11 }} axisLine={false} tickLine={false} />
                      <YAxis type="category" dataKey="label" width={112} tick={{ fill: 'var(--color-text-muted)', fontSize: 11 }} axisLine={false} tickLine={false} />
                      <Tooltip contentStyle={tooltipStyle} />
                      <Bar dataKey="success_rate" name="Run success rate (%)" fill="#10b981" radius={[0, 4, 4, 0]} maxBarSize={20} />
                    </BarChart>
                  </ResponsiveContainer>
                )}
              </div>
            </article>

            <article className="statistics-panel">
              <div className="statistics-panel-heading"><div><h2>Top predicted breeds</h2><p>Most frequent completed model outputs</p></div></div>
              <div className="statistics-chart">
                {!hasBreedData ? <ChartEmptyState /> : (
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={data.top_breeds} margin={{ top: 4, right: 8, left: -20, bottom: 20 }}>
                      <CartesianGrid stroke="var(--color-border)" strokeDasharray="3 3" vertical={false} />
                      <XAxis dataKey="name" angle={-28} textAnchor="end" interval={0} height={64} tick={{ fill: 'var(--color-text-muted)', fontSize: 10 }} axisLine={false} tickLine={false} />
                      <YAxis allowDecimals={false} tick={{ fill: 'var(--color-text-muted)', fontSize: 11 }} axisLine={false} tickLine={false} />
                      <Tooltip contentStyle={tooltipStyle} />
                      <Bar dataKey="count" name="Predictions" fill="#2563eb" radius={[4, 4, 0, 0]} maxBarSize={34} />
                    </BarChart>
                  </ResponsiveContainer>
                )}
              </div>
            </article>
          </section>

          <section className="statistics-panel statistics-activity-panel">
            <div className="statistics-panel-heading"><div><h2>Recent activity</h2><p>Latest model runs within the selected period</p></div></div>
            {data.recent_predictions.length === 0 ? <ChartEmptyState /> : (
              <div className="statistics-table-wrap">
                <table className="statistics-table">
                  <colgroup>
                    <col className="statistics-col-time" />
                    <col className="statistics-col-model" />
                    <col className="statistics-col-result" />
                    <col className="statistics-col-confidence" />
                    <col className="statistics-col-duration" />
                    <col className="statistics-col-status" />
                  </colgroup>
                  <thead><tr><th>Time</th><th>Model</th><th>Result</th><th>Confidence</th><th>Duration</th><th>Status</th></tr></thead>
                  <tbody>
                    {data.recent_predictions.map((prediction) => (
                      <tr
                        key={prediction.id}
                        tabIndex={0}
                        onClick={() => setSelectedPrediction(prediction)}
                        onKeyDown={(event) => { if (event.key === 'Enter') setSelectedPrediction(prediction); }}
                      >
                        <td>{formatDateTime(prediction.created_at)}</td>
                        <td>
                          {displayMethod(prediction.method, modelNames)}
                          {hasTtaEnabled(prediction.model_version) && <small>TTA: Bật</small>}
                        </td>
                        <td>
                          {prediction.is_unknown_breed ? (
                            <strong style={{ color: '#f59e0b' }}>Chưa xác định</strong>
                          ) : (
                            <strong>{prediction.predicted_breed ? formatBreedName(prediction.predicted_breed.name) : 'Pending'}</strong>
                          )}
                          <small>
                            {prediction.is_unknown_breed
                              ? `(Gần: ${prediction.predicted_breed ? formatBreedName(prediction.predicted_breed.name) : '-'})`
                              : (prediction.predicted_breed?.species.name ?? '')}
                          </small>
                        </td>
                        <td>{formatPercent(prediction.score)}</td>
                        <td>{formatDuration(prediction.processing_time)}</td>
                        <td><span className={`badge badge-${prediction.status.toLowerCase()}`}>{prediction.status}</span></td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
            {data.recent_total > data.recent_limit && (
              <div className="statistics-pagination">
                <button className="statistics-icon-button" onClick={() => setRecentPage((value) => Math.max(1, value - 1))} disabled={recentPage === 1} aria-label="Previous activity page"><ChevronLeft size={18} /></button>
                <span>Page {recentPage} of {recentPageCount} · {data.recent_total.toLocaleString()} runs</span>
                <button className="statistics-icon-button" onClick={() => setRecentPage((value) => Math.min(recentPageCount, value + 1))} disabled={recentPage >= recentPageCount} aria-label="Next activity page"><ChevronRight size={18} /></button>
              </div>
            )}
          </section>
        </>
      )}

      {selectedPrediction && (
        <PredictionDetail
          prediction={selectedPrediction}
          modelName={displayMethod(selectedPrediction.method, modelNames)}
          onClose={() => setSelectedPrediction(null)}
          onDeleted={() => {
            setSelectedPrediction(null);
            fetchStatistics();
          }}
        />
      )}
    </div>
  );
}
