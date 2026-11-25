import { AlertTriangle, CheckCircle2, Clock, Loader2, Sparkles, XCircle } from 'lucide-react';
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

export function BatchComparisonTable({
  predictions,
  selectedIndex,
  onSelectIndex,
}: BatchComparisonTableProps) {
  const completedWithBreed = predictions.filter(
    (p) => p.status === 'COMPLETED' && Boolean(p.predicted_breed?.name),
  );
  const runningCount = predictions.filter(
    (p) => p.status === 'UPLOADED' || p.status === 'PROCESSING',
  ).length;
  const failedCount = predictions.filter((p) => p.status === 'FAILED').length;

  const breedCounts = new Map<string, number>();
  for (const item of completedWithBreed) {
    const breed = item.predicted_breed!.name;
    breedCounts.set(breed, (breedCounts.get(breed) ?? 0) + 1);
  }

  const sortedCounts = Array.from(breedCounts.entries()).sort((a, b) => b[1] - a[1]);
  const topBreed = sortedCounts[0]?.[0] ?? null;
  const topBreedCount = sortedCounts[0]?.[1] ?? 0;
  const hasMultipleCompleted = completedWithBreed.length > 1;
  const allCompletedAgree = hasMultipleCompleted && sortedCounts.length === 1;
  const hasAnyDisagreement = hasMultipleCompleted && sortedCounts.length > 1;
  const consensusBreed = topBreedCount >= 2 ? topBreed : null;

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
          {allCompletedAgree && topBreed && (
            <span
              style={{
                display: 'inline-flex',
                alignItems: 'center',
                gap: 5,
                padding: '4px 10px',
                borderRadius: 999,
                fontSize: 11,
                fontWeight: 700,
                background: 'rgba(16,185,129,0.14)',
                color: '#10b981',
                border: '1px solid rgba(16,185,129,0.35)',
              }}
            >
              <CheckCircle2 size={12} />
              Đồng thuận: {formatBreedName(topBreed)} ({topBreedCount}/{completedWithBreed.length})
            </span>
          )}
          {hasAnyDisagreement && topBreed && (
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
              {consensusBreed
                ? `Đa số: ${formatBreedName(consensusBreed)} (${topBreedCount}/${completedWithBreed.length}) · Có bất đồng`
                : 'Các mô hình đang bất đồng Top-1'}
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
              <th style={{ padding: '8px 10px' }}>Đối chiếu</th>
            </tr>
          </thead>
          <tbody>
            {predictions.map((p, idx) => {
              const isSelected = idx === selectedIndex;
              const breedName = p.predicted_breed?.name ?? null;
              const ttaOn = hasTtaEnabled(p.model_version);
              const isPending = p.status === 'UPLOADED' || p.status === 'PROCESSING';
              const isFailed = p.status === 'FAILED';

              let agreementKind: 'agree' | 'disagree' | 'neutral' = 'neutral';
              if (p.status === 'COMPLETED' && breedName && hasMultipleCompleted) {
                if (allCompletedAgree) {
                  agreementKind = 'agree';
                } else if (consensusBreed && breedName === consensusBreed) {
                  agreementKind = 'agree';
                } else {
                  agreementKind = 'disagree';
                }
              }

              const rowBg = isSelected
                ? 'rgba(139,92,246,0.12)'
                : isFailed
                  ? 'rgba(239,68,68,0.06)'
                  : agreementKind === 'agree'
                    ? 'rgba(16,185,129,0.06)'
                    : agreementKind === 'disagree'
                      ? 'rgba(245,158,11,0.08)'
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

                  <td style={{ padding: '10px', whiteSpace: 'nowrap' }}>
                    {isPending ? (
                      <span className="badge badge-processing" style={{ fontSize: 10, padding: '2px 8px' }}>
                        Đang chạy
                      </span>
                    ) : isFailed ? (
                      <span
                        className="badge badge-failed"
                        style={{ fontSize: 10, padding: '2px 8px' }}
                        title={p.error_message ?? 'Mô hình gặp lỗi'}
                      >
                        Lỗi
                      </span>
                    ) : agreementKind === 'agree' ? (
                      <span
                        style={{
                          padding: '2px 8px',
                          borderRadius: 999,
                          fontSize: 10,
                          fontWeight: 700,
                          background: 'rgba(16,185,129,0.16)',
                          color: '#10b981',
                        }}
                      >
                        Đồng thuận
                      </span>
                    ) : agreementKind === 'disagree' ? (
                      <span
                        style={{
                          padding: '2px 8px',
                          borderRadius: 999,
                          fontSize: 10,
                          fontWeight: 700,
                          background: 'rgba(245,158,11,0.18)',
                          color: '#d97706',
                        }}
                      >
                        Bất đồng
                      </span>
                    ) : (
                      <span style={{ fontSize: 11, color: 'var(--color-text-faint)' }}>Hoàn tất</span>
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
