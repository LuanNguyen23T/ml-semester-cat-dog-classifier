import { useCallback, useEffect, useState } from 'react';
import { ChevronLeft, ChevronRight, Filter, History, RefreshCw, X } from 'lucide-react';

import { BatchComparisonTable } from '../../components/BatchComparisonTable';
import { PredictionCard } from '../../components/PredictionCard';
import { PredictionResult } from '../../components/PredictionResult';
import { formatApiError, predictionApi } from '../../services/predictionApi';
import {
  FALLBACK_MODELS,
  type ModelInfo,
  type PredictionHistoryGroup,
  type PredictionResponse,
  type PredictionStatus,
} from '../../types/prediction';

const STATUS_OPTIONS = [
  { value: '', label: 'Tất cả trạng thái' },
  { value: 'COMPLETED', label: 'Completed' },
  { value: 'PROCESSING', label: 'Processing' },
  { value: 'FAILED', label: 'Failed' },
];
const LIMIT = 12;

function getGroupStatus(predictions: PredictionResponse[]): PredictionStatus {
  if (predictions.some((item) => item.status === 'PROCESSING')) return 'PROCESSING';
  if (predictions.some((item) => item.status === 'UPLOADED')) return 'UPLOADED';
  if (predictions.every((item) => item.status === 'FAILED')) return 'FAILED';
  return 'COMPLETED';
}

function HistoryDetail({
  group,
  onClose,
}: {
  group: PredictionHistoryGroup;
  onClose: () => void;
}) {
  const [selectedIndex, setSelectedIndex] = useState(0);
  const selectedResult = group.predictions[selectedIndex] ?? group.predictions[0];

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [onClose]);

  if (!selectedResult) return null;
  return (
    <div
      role="presentation"
      onMouseDown={onClose}
      style={{
        position: 'fixed',
        inset: 0,
        zIndex: 200,
        display: 'grid',
        placeItems: 'center',
        padding: 16,
        background: 'rgba(15, 23, 42, .62)',
        backdropFilter: 'blur(4px)',
      }}
    >
      <section
        role="dialog"
        aria-modal="true"
        aria-label="Chi tiết dự đoán"
        onMouseDown={(event) => event.stopPropagation()}
        style={{
          width: 'min(820px, 100%)',
          maxHeight: 'calc(100vh - 32px)',
          overflowY: 'auto',
          padding: 18,
          border: '1px solid var(--color-border)',
          borderRadius: 12,
          background: 'var(--color-bg)',
          boxShadow: 'var(--shadow-lg)',
          display: 'flex',
          flexDirection: 'column',
          gap: 16,
        }}
      >
        <header style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 16 }}>
          <div>
            <span style={{ color: 'var(--color-text-muted)', fontSize: 11, fontWeight: 700, textTransform: 'uppercase' }}>
              Chi tiết phân loại
            </span>
            <h2 style={{ marginTop: 3, fontSize: 20 }}>
              {group.batch_id ? 'So sánh All Models' : 'Kết quả phân loại'}
            </h2>
          </div>
          <button
            className="btn btn-ghost"
            style={{ width: 38, height: 38, padding: 0 }}
            onClick={onClose}
            aria-label="Đóng chi tiết"
          >
            <X size={18} />
          </button>
        </header>

        <div
          style={{
            width: '100%',
            height: 'min(40vh, 360px)',
            padding: 8,
            borderRadius: 8,
            background: 'var(--color-surface-2)',
          }}
        >
          <img
            src={group.image_key}
            alt="Classified pet"
            style={{ display: 'block', width: '100%', height: '100%', objectFit: 'contain', borderRadius: 6 }}
          />
        </div>

        {group.batch_id && group.predictions.length > 1 && (
          <BatchComparisonTable
            predictions={group.predictions}
            selectedIndex={selectedIndex}
            onSelectIndex={setSelectedIndex}
          />
        )}

        <PredictionResult prediction={selectedResult} />
      </section>
    </div>
  );
}

export function HistoryPage() {
  const [groups, setGroups] = useState<PredictionHistoryGroup[]>([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedStatus, setSelectedStatus] = useState('');
  const [selectedMethod, setSelectedMethod] = useState('');
  const [page, setPage] = useState(1);
  const [models, setModels] = useState<ModelInfo[]>(FALLBACK_MODELS);
  const [selectedGroup, setSelectedGroup] = useState<PredictionHistoryGroup | null>(null);

  const fetchPredictions = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await predictionApi.listPredictionHistory({
        status: selectedStatus || undefined,
        method: selectedMethod || undefined,
        page,
        limit: LIMIT,
      });
      setGroups(data.items);
      setTotal(data.total);
    } catch (requestError: unknown) {
      setError(formatApiError(requestError, 'Không thể tải lịch sử phân loại.'));
    } finally {
      setLoading(false);
    }
  }, [selectedStatus, selectedMethod, page]);

  useEffect(() => {
    predictionApi
      .listModels()
      .then((data) => {
        if (data.length > 0) setModels(data);
      })
      .catch(() => setModels(FALLBACK_MODELS));
  }, []);

  useEffect(() => {
    // Synchronize history with the current server-side filters and page.
    // oxlint-disable-next-line react/set-state-in-effect
    fetchPredictions();
  }, [fetchPredictions]);

  const pageCount = Math.max(1, Math.ceil(total / LIMIT));
  const filterStyle: React.CSSProperties = {
    minWidth: 160,
    flex: '1 1 180px',
    maxWidth: 260,
    padding: '10px 14px',
    color: 'var(--color-text)',
    font: '500 14px var(--font-sans)',
    border: '1px solid var(--color-border)',
    borderRadius: 8,
    outline: 'none',
    background: 'var(--color-surface-2)',
    cursor: 'pointer',
  };

  return (
    <div style={{ minHeight: '100vh', width: '100%', maxWidth: 1100, margin: '0 auto', padding: '36px 16px 60px' }}>
      <div style={{ marginBottom: 28 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 8 }}>
          <History size={22} color="var(--color-accent)" />
          <h1 style={{ fontSize: 28, fontWeight: 800 }}>Classification History</h1>
        </div>
        <p style={{ color: 'var(--color-text-muted)', fontSize: 15 }}>
          {loading ? 'Đang tải...' : `${total} lượt phân loại`}
        </p>
      </div>

      <div style={{ display: 'flex', gap: 12, alignItems: 'center', marginBottom: 28, flexWrap: 'wrap' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <Filter size={14} color="var(--color-text-muted)" />
          <span style={{ fontSize: 13, color: 'var(--color-text-muted)', fontWeight: 600 }}>Lọc:</span>
        </div>
        <select
          value={selectedStatus}
          onChange={(event) => {
            setSelectedStatus(event.target.value);
            setPage(1);
          }}
          style={filterStyle}
        >
          {STATUS_OPTIONS.map((option) => (
            <option key={option.value} value={option.value}>
              {option.label}
            </option>
          ))}
        </select>
        <select
          value={selectedMethod}
          onChange={(event) => {
            setSelectedMethod(event.target.value);
            setPage(1);
          }}
          style={filterStyle}
        >
          <option value="">Tất cả mô hình (5)</option>
          {models.map((model) => (
            <option key={model.id} value={model.id}>
              {model.name}
            </option>
          ))}
        </select>
        <button className="btn btn-ghost" onClick={fetchPredictions} style={{ marginLeft: 'auto' }}>
          <RefreshCw size={14} className={loading ? 'animate-spin' : ''} /> Làm mới
        </button>
      </div>

      {loading ? (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(min(100%, 280px), 1fr))', gap: 16 }}>
          {Array.from({ length: 6 }).map((_, index) => (
            <div key={index} className="skeleton" style={{ height: 340, borderRadius: 8 }} />
          ))}
        </div>
      ) : error ? (
        <div style={{ padding: '60px 20px', color: 'var(--color-error)', textAlign: 'center' }}>
          <p>{error}</p>
          <button className="btn btn-primary" onClick={fetchPredictions} style={{ marginTop: 20 }}>
            Thử lại
          </button>
        </div>
      ) : groups.length === 0 ? (
        <div style={{ padding: '80px 20px', color: 'var(--color-text-muted)', textAlign: 'center' }}>
          <History size={48} color="rgba(139,92,246,0.3)" style={{ margin: '0 auto 16px' }} />
          <p style={{ marginBottom: 8 }}>Chưa có lịch sử nhận diện</p>
          <p style={{ fontSize: 14, color: 'var(--color-text-faint)' }}>
            Hãy tải lên một bức ảnh ở trang chủ để bắt đầu
          </p>
        </div>
      ) : (
        <>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(min(100%, 280px), 1fr))', gap: 16 }}>
            {groups.map((group) => {
              const cardPrediction = { ...group.predictions[0], status: getGroupStatus(group.predictions) };
              return (
                <PredictionCard
                  key={group.group_id}
                  prediction={cardPrediction}
                  batchSize={group.batch_id ? group.predictions.length : 1}
                  batchResults={group.predictions}
                  onClick={() => setSelectedGroup(group)}
                />
              );
            })}
          </div>
          <div style={{ display: 'flex', justifyContent: 'center', gap: 12, marginTop: 32, alignItems: 'center' }}>
            <button
              className="btn btn-ghost"
              onClick={() => setPage((value) => Math.max(1, value - 1))}
              disabled={page === 1}
            >
              <ChevronLeft size={16} /> Trước
            </button>
            <span style={{ color: 'var(--color-text-muted)', fontSize: 14 }}>
              Trang {page} / {pageCount}
            </span>
            <button
              className="btn btn-ghost"
              onClick={() => setPage((value) => value + 1)}
              disabled={page >= pageCount}
            >
              Sau <ChevronRight size={16} />
            </button>
          </div>
        </>
      )}

      {selectedGroup && <HistoryDetail group={selectedGroup} onClose={() => setSelectedGroup(null)} />}
    </div>
  );
}
