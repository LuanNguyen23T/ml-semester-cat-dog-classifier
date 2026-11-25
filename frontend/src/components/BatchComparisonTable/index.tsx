import { AlertTriangle, CheckCircle2, Clock, Loader2, Sparkles, XCircle, Tag } from 'lucide-react';
import {
  formatBreedName,
  getModelDisplayName,
  hasTtaEnabled,
  type PredictionResponse,
} from '../../types/prediction';

interface BatchComparisonTableProps {
  predictions: PredictionResponse[];
  selectedIndex: number;
  onSelectIndex: (index: number) => void;
}

function getEffectiveVerdict(p: PredictionResponse): string {
  if (p.is_unknown_breed) return '__UNKNOWN__';
  return p.predicted_breed?.name ?? '__UNKNOWN__';
}

export function BatchComparisonTable({
  predictions,
  selectedIndex,
  onSelectIndex,
}: BatchComparisonTableProps) {
  const batchNote = predictions.find((p) => p.note)?.note;
  const completedWithBreed = predictions.filter(
    (p) => p.status === 'COMPLETED' && Boolean(p.predicted_breed?.name),
  );
  const runningCount = predictions.filter(
    (p) => p.status === 'UPLOADED' || p.status === 'PROCESSING',
  ).length;
  const failedCount = predictions.filter((p) => p.status === 'FAILED').length;

  const verdictCounts = new Map<string, number>();
  for (const item of completedWithBreed) {
    const verdict = getEffectiveVerdict(item);
    verdictCounts.set(verdict, (verdictCounts.get(verdict) ?? 0) + 1);
  }

  const sortedCounts = Array.from(verdictCounts.entries()).sort((a, b) => b[1] - a[1]);
  const topVerdict = sortedCounts[0]?.[0] ?? null;
  const topVerdictCount = sortedCounts[0]?.[1] ?? 0;
  const hasMultipleCompleted = completedWithBreed.length > 1;
  const allCompletedAgree = hasMultipleCompleted && sortedCounts.length === 1;
  const hasAnyDisagreement = hasMultipleCompleted && sortedCounts.length > 1;
  const consensusVerdict = topVerdictCount >= 2 ? topVerdict : null;

  return (
    <div className="glass-card" style={{ padding: 18, minWidth: 0 }}>
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          gap: 10,
          marginBottom: 12,
          flexWrap: 'wrap',
        }}
      >
        <div>
          <h4 style={{ fontSize: 15, fontWeight: 700, color: 'var(--color-text)' }}>
            So sánh 5 mô hình ({predictions.length})
          </h4>
          <p style={{ fontSize: 12, color: 'var(--color-text-muted)', marginTop: 2 }}>
            {runningCount > 0
              ? `Đang chạy ${runningCount}/${predictions.length} mô hình — bấm vào dòng để xem Top-5 chi tiết.`
              : 'Bấm vào từng dòng để xem biểu đồ Top-5 chi tiết của mô hình đó.'}
          </p>
        </div>

        <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', alignItems: 'center' }}>
          {allCompletedAgree && topVerdict && (
            <span
              style={{
                display: 'inline-flex',
                alignItems: 'center',
                gap: 5,
                padding: '4px 10px',
                borderRadius: 999,
                fontSize: 11,
                fontWeight: 700,
                background: topVerdict === '__UNKNOWN__' ? 'rgba(245,158,11,0.14)' : 'rgba(16,185,129,0.14)',
                color: topVerdict === '__UNKNOWN__' ? '#d97706' : '#10b981',
                border: topVerdict === '__UNKNOWN__' ? '1px solid rgba(245,158,11,0.35)' : '1px solid rgba(16,185,129,0.35)',
              }}
            >
              {topVerdict === '__UNKNOWN__' ? <AlertTriangle size={12} /> : <CheckCircle2 size={12} />}
              {topVerdict === '__UNKNOWN__'
                ? `Chưa xác định (${topVerdictCount}/${completedWithBreed.length})`
                : `Đồng thuận: ${formatBreedName(topVerdict)} (${topVerdictCount}/${completedWithBreed.length})`}
            </span>
          )}
          {hasAnyDisagreement && topVerdict && (
            <span
              style={{
                display: 'inline-flex',
                alignItems: 'center',
                gap: 5,
                padding: '4px 10px',
                borderRadius: 999,
                fontSize: 11,
                fontWeight: 700,
                background: 'rgba(245,158,11,0.14)',
                color: '#d97706',
                border: '1px solid rgba(245,158,11,0.35)',
              }}
            >
              <AlertTriangle size={12} />
              {consensusVerdict
                ? consensusVerdict === '__UNKNOWN__'
                  ? `Chưa xác định (${topVerdictCount}/${completedWithBreed.length}) · Có phân hóa`
                  : `Đa số: ${formatBreedName(consensusVerdict)} (${topVerdictCount}/${completedWithBreed.length}) · Có bất đồng`
                : 'Các mô hình đang phân hóa Top-1'}
            </span>
          )}
          {failedCount > 0 && (
            <span
              style={{
                display: 'inline-flex',
                alignItems: 'center',
                gap: 5,
                padding: '4px 10px',
                borderRadius: 999,
                fontSize: 11,
                fontWeight: 700,
                background: 'rgba(239,68,68,0.12)',
                color: 'var(--color-error)',
                border: '1px solid rgba(239,68,68,0.3)',
              }}
            >
              <XCircle size={12} />
              {failedCount} mô hình lỗi
            </span>
          )}
        </div>
      </div>

      {batchNote && (
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: 6,
            padding: '6px 12px',
            marginBottom: 12,
            borderRadius: 8,
            background: 'rgba(139, 92, 246, 0.08)',
            border: '1px solid rgba(139, 92, 246, 0.22)',
            color: 'var(--color-text)',
            fontSize: 12,
          }}
        >
          <Tag size={13} color="var(--color-accent)" style={{ flexShrink: 0 }} />
          <span>Nhãn thực tế của ảnh: <strong style={{ color: 'var(--color-accent)' }}>{batchNote}</strong></span>
        </div>
      )}

      <div style={{ width: '100%', overflowX: 'auto', WebkitOverflowScrolling: 'touch' }}>
        <table
          style={{
            width: '100%',
            minWidth: 560,
            borderCollapse: 'collapse',
            fontSize: 13,
          }}
        >
          <thead>
            <tr
              style={{
                borderBottom: '1px solid var(--color-border)',
                color: 'var(--color-text-muted)',
                fontSize: 11,
                textTransform: 'uppercase',
                letterSpacing: '0.05em',
                textAlign: 'left',
              }}
            >
              <th style={{ padding: '8px 10px' }}>Mô hình</th>
              <th style={{ padding: '8px 10px' }}>Top-1 dự đoán</th>
              <th style={{ padding: '8px 10px' }}>Tin cậy</th>
              <th style={{ padding: '8px 10px' }}>Thời gian</th>
            </tr>
          </thead>
          <tbody>
            {predictions.map((p, idx) => {
              const isSelected = idx === selectedIndex;
              const breedName = p.predicted_breed?.name ?? null;
              const ttaOn = hasTtaEnabled(p.model_version);
              const isPending = p.status === 'UPLOADED' || p.status === 'PROCESSING';
              const isFailed = p.status === 'FAILED';

              const rowBg = isSelected
                ? 'rgba(139,92,246,0.12)'
                : isFailed
                  ? 'rgba(239,68,68,0.06)'
                  : 'transparent';

              return (
                <tr
                  key={p.id}
                  onClick={() => onSelectIndex(idx)}
                  style={{
                    cursor: 'pointer',
                    background: rowBg,
                    borderBottom: '1px solid var(--color-border)',
                    outline: isSelected ? '1px solid rgba(139,92,246,0.45)' : 'none',
                    transition: 'background 0.15s ease',
                  }}
                >
                  <td style={{ padding: '10px' }}>
                    <div style={{ fontWeight: 700, color: 'var(--color-text)' }}>
                      {getModelDisplayName(p.method)}
                    </div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginTop: 2 }}>
                      <span
                        style={{
                          fontSize: 10,
                          color: ttaOn ? 'var(--color-accent)' : 'var(--color-text-faint)',
                          fontWeight: 600,
                          display: 'inline-flex',
                          alignItems: 'center',
                          gap: 3,
                        }}
                      >
                        <Sparkles size={9} />
                        {ttaOn ? 'TTA: Bật' : 'TTA: Tắt'}
                      </span>
                    </div>
                  </td>

                  <td style={{ padding: '10px', fontWeight: 600, color: 'var(--color-text)' }}>
                    {isPending ? (
                      <span style={{ display: 'inline-flex', alignItems: 'center', gap: 6, color: 'var(--color-text-muted)', fontWeight: 500 }}>
                        <Loader2 size={13} className="animate-spin" />
                        Đang suy luận...
                      </span>
                    ) : isFailed ? (
                      <span style={{ color: 'var(--color-error)', fontWeight: 600 }}>
                        Thất bại
                      </span>
                    ) : p.is_unknown_breed ? (
                      <div>
                        <span style={{ color: '#d97706', display: 'inline-flex', alignItems: 'center', gap: 4, fontWeight: 700 }}>
                          <AlertTriangle size={12} />
                          Chưa xác định
                        </span>
                        <div style={{ fontSize: 11, color: 'var(--color-text-muted)', fontWeight: 400 }}>
                          (Gần: {formatBreedName(breedName)})
                        </div>
                      </div>
                    ) : (
                      <span>{formatBreedName(breedName)}</span>
                    )}
                  </td>

                  <td style={{ padding: '10px', fontWeight: 700, color: 'var(--color-text)' }}>
                    {p.status === 'COMPLETED' && p.score != null
                      ? `${(p.score * 100).toFixed(1)}%`
                      : '—'}
                  </td>

                  <td style={{ padding: '10px', color: 'var(--color-text-muted)', whiteSpace: 'nowrap' }}>
                    {p.processing_time != null ? (
                      <span style={{ display: 'inline-flex', alignItems: 'center', gap: 4 }}>
                        <Clock size={11} />
                        {p.processing_time.toFixed(2)}s
                      </span>
                    ) : (
                      '—'
                    )}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
