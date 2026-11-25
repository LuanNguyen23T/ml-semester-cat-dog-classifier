import { CheckCircle, Clock, XCircle, Loader2, Cat, Dog, Sparkles, AlertTriangle, Tag } from 'lucide-react';
import {
  formatBreedName,
  getModelDisplayName,
  hasTtaEnabled,
  type PredictionResponse,
} from '../../types/prediction';

interface PredictionCardProps {
  prediction: PredictionResponse;
  onClick?: () => void;
  batchSize?: number;
  batchResults?: PredictionResponse[];
}

const STATUS_CONFIG = {
  UPLOADED: { icon: <Clock size={12} />, className: 'badge badge-uploaded', label: 'Uploaded' },
  PROCESSING: { icon: <Loader2 size={12} />, className: 'badge badge-processing', label: 'Processing' },
  COMPLETED: { icon: <CheckCircle size={12} />, className: 'badge badge-completed', label: 'Completed' },
  FAILED: { icon: <XCircle size={12} />, className: 'badge badge-failed', label: 'Failed' },
};

function formatDate(dateStr: string): string {
  const date = new Date(dateStr);
  return date.toLocaleDateString('vi-VN', { day: '2-digit', month: '2-digit', hour: '2-digit', minute: '2-digit' });
}

export function PredictionCard({ prediction, onClick, batchSize = 1, batchResults = [] }: PredictionCardProps) {
  const { status, predicted_breed, score, created_at } = prediction;
  const cfg = STATUS_CONFIG[status] ?? STATUS_CONFIG.UPLOADED;
  const isBatch = batchSize > 1;
  const isUnknown = Boolean(prediction.is_unknown_breed);
  const breedName = predicted_breed?.name ? formatBreedName(predicted_breed.name) : '-';
  const speciesName = predicted_breed?.species?.name ?? '';
  const hasTta = isBatch
    ? batchResults.some((item) => hasTtaEnabled(item.model_version))
    : hasTtaEnabled(prediction.model_version);
  const displayNote = prediction.note || batchResults.find((b) => b.note)?.note;

  const batchBreeds = Array.from(
    batchResults
      .reduce((summary, result) => {
        const breed = result.status === 'COMPLETED' ? result.predicted_breed : null;
        if (!breed) return summary;
        const current = summary.get(breed.name);
        summary.set(breed.name, {
          name: formatBreedName(breed.name),
          species: breed.species.name,
          count: (current?.count ?? 0) + 1,
        });
        return summary;
      }, new Map<string, { name: string; species: string; count: number }>())
      .values(),
  ).sort((left, right) => right.count - left.count || left.name.localeCompare(right.name));
  const primaryBatchBreed = batchBreeds[0];

  return (
    <div
      className="glass-card"
      onClick={onClick}
      style={{
        cursor: onClick ? 'pointer' : 'default',
        display: 'flex',
        flexDirection: 'column',
        overflow: 'hidden',
        minWidth: 0,
      }}
      id={`prediction-card-${prediction.id}`}
    >
      {/* Image Banner */}
      <div
        style={{
          width: '100%',
          height: 180,
          background: 'var(--color-surface-2)',
          position: 'relative',
          overflow: 'hidden',
        }}
      >
        {prediction.image_key && prediction.image_key.startsWith('http') ? (
          <img
            src={prediction.image_key}
            alt="Uploaded pet"
            style={{ width: '100%', height: '100%', objectFit: 'cover' }}
          />
        ) : (
          <div
            style={{
              width: '100%',
              height: '100%',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              color: 'var(--color-text-faint)',
              fontSize: 12,
            }}
          >
            No Image
          </div>
        )}
          <div
            style={{
              position: 'absolute',
              top: 12,
              right: 12,
              background: isUnknown
                ? 'rgba(254, 243, 199, 0.95)'
                : status === 'COMPLETED'
                ? 'rgba(236, 253, 245, 0.95)'
                : 'rgba(255,255,255,0.92)',
              border: isUnknown
                ? '1px solid rgba(245, 158, 11, 0.35)'
                : status === 'COMPLETED'
                ? '1px solid rgba(16, 185, 129, 0.35)'
                : '1px solid var(--color-border)',
              backdropFilter: 'blur(4px)',
              padding: '4px 10px',
              borderRadius: 999,
              display: 'flex',
              alignItems: 'center',
              boxShadow: '0 2px 8px rgba(0,0,0,0.08)',
            }}
          >
            {isUnknown ? (
              <span
                style={{
                  display: 'inline-flex',
                  alignItems: 'center',
                  gap: 4,
                  fontSize: 11,
                  fontWeight: 700,
                  color: '#d97706',
                  textTransform: 'none',
                }}
              >
                <AlertTriangle size={12} color="#d97706" />
                Chưa xác định
              </span>
            ) : status === 'COMPLETED' ? (
              <span
                style={{
                  display: 'inline-flex',
                  alignItems: 'center',
                  gap: 4,
                  fontSize: 11,
                  fontWeight: 700,
                  color: '#059669',
                  textTransform: 'none',
                }}
              >
                <CheckCircle size={12} color="#059669" />
                Completed
              </span>
            ) : (
              <span
                className={cfg.className}
                style={{
                  margin: 0,
                  padding: 0,
                  background: 'transparent',
                  border: 'none',
                  textTransform: 'none',
                  fontSize: 11,
                  fontWeight: 700,
                }}
              >
                {cfg.icon}
                <span style={{ marginLeft: 4 }}>{cfg.label}</span>
              </span>
            )}
          </div>
      </div>

      <div style={{ padding: 20, display: 'flex', flexDirection: 'column', gap: 12, minWidth: 0 }}>
        {/* Date, Method, TTA, Time */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 6, flexWrap: 'wrap' }}>
          <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', alignItems: 'center' }}>
            {(isBatch || prediction.method) && (
              <span
                style={{
                  fontSize: 10,
                  color: 'var(--color-text-muted)',
                  background: 'var(--color-surface)',
                  padding: '2px 6px',
                  borderRadius: 4,
                  border: '1px solid var(--color-border)',
                  fontWeight: 600,
                }}
              >
                {isBatch ? `${batchSize} models` : getModelDisplayName(prediction.method)}
              </span>
            )}
            {hasTta && (
              <span
                style={{
                  fontSize: 10,
                  color: 'var(--color-accent)',
                  background: 'rgba(139,92,246,0.12)',
                  padding: '2px 6px',
                  borderRadius: 4,
                  border: '1px solid rgba(139,92,246,0.3)',
                  fontWeight: 700,
                  display: 'inline-flex',
                  alignItems: 'center',
                  gap: 3,
                }}
              >
                <Sparkles size={9} />
                TTA
              </span>
            )}
            {!isBatch && prediction.processing_time != null && (
              <span
                style={{
                  fontSize: 10,
                  color: 'var(--color-text-faint)',
                  background: 'var(--color-surface)',
                  padding: '2px 6px',
                  borderRadius: 4,
                  border: '1px solid var(--color-border)',
                  display: 'flex',
                  alignItems: 'center',
                  gap: 3,
                }}
              >
                <Clock size={9} />
                {prediction.processing_time.toFixed(2)}s
              </span>
            )}
          </div>
          <span style={{ fontSize: 11, color: 'var(--color-text-faint)' }}>
            {formatDate(created_at)}
          </span>
        </div>

        {/* Note / Ground Truth */}
        {displayNote && (
          <div
            style={{
              display: 'inline-flex',
              alignItems: 'center',
              gap: 5,
              padding: '3px 8px',
              borderRadius: 6,
              background: 'rgba(139, 92, 246, 0.08)',
              border: '1px solid rgba(139, 92, 246, 0.22)',
              color: 'var(--color-accent)',
              fontSize: 11,
              fontWeight: 600,
              maxWidth: '100%',
              overflow: 'hidden',
              textOverflow: 'ellipsis',
              whiteSpace: 'nowrap',
            }}
            title={displayNote}
          >
            <Tag size={11} style={{ flexShrink: 0 }} />
            <span style={{ overflow: 'hidden', textOverflow: 'ellipsis' }}>{displayNote}</span>
          </div>
        )}

        {/* Breed */}
        {isBatch ? (
          <div style={{ display: 'grid', gap: 7 }}>
            {primaryBatchBreed ? (
              <>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                  {primaryBatchBreed.species === 'Cat' ? (
                    <Cat size={16} color="var(--color-accent)" />
                  ) : (
                    <Dog size={16} color="#6366f1" />
                  )}
                  <span style={{ fontSize: 16, fontWeight: 700, color: 'var(--color-text)' }}>
                    {primaryBatchBreed.name}
                  </span>
                  <span style={{ marginLeft: 'auto', fontSize: 11, color: 'var(--color-text-muted)' }}>
                    {primaryBatchBreed.count} models
                  </span>
                </div>
                {batchBreeds.slice(1).map((breed) => (
                  <div
                    key={breed.name}
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: 6,
                      paddingLeft: 24,
                      color: 'var(--color-text-muted)',
                      fontSize: 12,
                    }}
                  >
                    <span>{breed.name}</span>
                    <span style={{ marginLeft: 'auto', color: 'var(--color-text-faint)', fontSize: 10 }}>
                      {breed.count} models
                    </span>
                  </div>
                ))}
              </>
            ) : (
              <span style={{ color: 'var(--color-text-muted)', fontSize: 13 }}>
                Đang chờ kết quả từ các mô hình
              </span>
            )}
          </div>
        ) : (
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              {isUnknown ? (
                <AlertTriangle size={16} color="#d97706" />
              ) : speciesName === 'Cat' ? (
                <Cat size={16} color="var(--color-accent)" />
              ) : speciesName === 'Dog' ? (
                <Dog size={16} color="#6366f1" />
              ) : null}
              <span
                style={{
                  fontSize: 16,
                  fontWeight: 700,
                  color: isUnknown ? '#d97706' : 'var(--color-text)',
                }}
              >
                {isUnknown ? 'Chưa xác định' : breedName}
              </span>
            </div>
            <span style={{ fontSize: 12, color: 'var(--color-text-muted)', marginLeft: 24 }}>
              {isUnknown ? `(Gần: ${breedName})` : speciesName}
            </span>
          </div>
        )}

        {/* Score bar */}
        {!isBatch && score !== null && score !== undefined && status === 'COMPLETED' && (
          <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 6 }}>
              <span style={{ fontSize: 11, color: 'var(--color-text-faint)' }}>Độ tin cậy</span>
              <span style={{ fontSize: 12, fontWeight: 700, color: 'var(--color-accent)' }}>
                {Math.round(score * 100)}%
              </span>
            </div>
            <div
              style={{
                height: 4,
                background: 'rgba(139,92,246,0.15)',
                borderRadius: 2,
                overflow: 'hidden',
              }}
            >
              <div
                style={{
                  height: '100%',
                  width: `${score * 100}%`,
                  background: 'var(--gradient-brand)',
                  borderRadius: 2,
                  transition: 'width 0.8s ease',
                }}
              />
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
