import { CheckCircle, XCircle, Clock, Loader2, Award, Cpu, TrendingUp, Sparkles, AlertTriangle, Tag } from 'lucide-react';
import {
  formatBreedName,
  getModelDisplayName,
  hasTtaEnabled,
  type PredictionResponse,
} from '../../types/prediction';
import { PredictionChart } from '../PredictionChart';

interface PredictionResultProps {
  prediction: PredictionResponse;
}

function StatusBadge({ status }: { status: string }) {
  const configs = {
    UPLOADED: { label: 'Uploaded', className: 'badge badge-uploaded', icon: <Clock size={11} /> },
    PROCESSING: { label: 'Processing', className: 'badge badge-processing', icon: <Loader2 size={11} className="animate-spin" /> },
    COMPLETED: { label: 'Completed', className: 'badge badge-completed', icon: <CheckCircle size={11} /> },
    FAILED: { label: 'Failed', className: 'badge badge-failed', icon: <XCircle size={11} /> },
  };

  const config = configs[status as keyof typeof configs] ?? configs.UPLOADED;

  return (
    <span className={config.className}>
      {config.icon}
      {config.label}
    </span>
  );
}

function ScoreRing({ score }: { score: number }) {
  const pct = Math.round(score * 100);
  const radius = 42;
  const circumference = 2 * Math.PI * radius;
  const strokeDashoffset = circumference - (pct / 100) * circumference;
  const color = pct >= 75 ? '#10b981' : pct >= 50 ? '#f59e0b' : '#ef4444';

  return (
    <div style={{ position: 'relative', width: 110, height: 110, flexShrink: 0 }}>
      <svg width="110" height="110" style={{ transform: 'rotate(-90deg)' }}>
        <circle cx="55" cy="55" r={radius} fill="none" stroke="rgba(139,92,246,0.15)" strokeWidth="8" />
        <circle
          cx="55"
          cy="55"
          r={radius}
          fill="none"
          stroke={color}
          strokeWidth="8"
          strokeLinecap="round"
          strokeDasharray={circumference}
          strokeDashoffset={strokeDashoffset}
          style={{ transition: 'stroke-dashoffset 1s ease' }}
        />
      </svg>
      <div
        style={{
          position: 'absolute',
          inset: 0,
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
        }}
      >
        <span style={{ fontSize: 22, fontWeight: 800, color, lineHeight: 1 }}>{pct}%</span>
        <span style={{ fontSize: 10, color: 'var(--color-text-muted)', marginTop: 2 }}>tin cậy</span>
      </div>
    </div>
  );
}

export function PredictionResult({ prediction }: PredictionResultProps) {
  const { status, predicted_breed, score, top_k_results, model_version, method, error_message } = prediction;
  const ttaEnabled = hasTtaEnabled(model_version);
  const modelLabel = getModelDisplayName(method);

  if (status === 'UPLOADED' || status === 'PROCESSING') {
    return (
      <div className="glass-card animate-fade-in" style={{ padding: 40, textAlign: 'center' }}>
        <div
          style={{
            width: 64,
            height: 64,
            borderRadius: '50%',
            border: '3px solid rgba(139,92,246,0.2)',
            borderTopColor: 'var(--color-accent)',
            animation: 'spin 1s linear infinite',
            margin: '0 auto 20px',
          }}
        />
        <h3 style={{ fontSize: 18, fontWeight: 600, marginBottom: 8 }}>
          Đang phân tích ({modelLabel})...
        </h3>
        <p style={{ color: 'var(--color-text-muted)', fontSize: 14 }}>
          AI đang nhận diện giống thú cưng của bạn
        </p>
      </div>
    );
  }

  if (status === 'FAILED') {
    return (
      <div
        className="glass-card animate-fade-in"
        style={{
          padding: 32,
          textAlign: 'center',
          borderColor: 'rgba(239,68,68,0.3)',
        }}
      >
        <XCircle size={48} color="var(--color-error)" style={{ margin: '0 auto 16px' }} />
        <h3 style={{ fontSize: 18, fontWeight: 600, color: 'var(--color-error)', marginBottom: 8 }}>
          Nhận diện thất bại ({modelLabel})
        </h3>
        {error_message && (
          <p
            style={{
              color: 'var(--color-text-muted)',
              fontSize: 13,
              background: 'rgba(239,68,68,0.05)',
              padding: '12px 16px',
              borderRadius: 8,
              marginTop: 12,
              textAlign: 'left',
              overflowWrap: 'anywhere',
            }}
          >
            {error_message}
          </p>
        )}
      </div>
    );
  }

  const breedDisplayName = formatBreedName(predicted_breed?.name);
  const speciesName = predicted_breed?.species?.name ?? '—';

  return (
    <div className="animate-fade-in" style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      <div className="glass-card" style={{ padding: 24 }}>
        <div
          style={{
            display: 'flex',
            alignItems: 'flex-start',
            justifyContent: 'space-between',
            marginBottom: 24,
            flexWrap: 'wrap',
            gap: 12,
          }}
        >
          <div style={{ minWidth: 0 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 10, flexWrap: 'wrap' }}>
              <StatusBadge status={status} />
              {(method || model_version) && (
                <span
                  style={{
                    display: 'inline-flex',
                    alignItems: 'center',
                    gap: 4,
                    fontSize: 11,
                    color: 'var(--color-text-muted)',
                    padding: '3px 8px',
                    background: 'rgba(139,92,246,0.08)',
                    border: '1px solid var(--color-border)',
                    borderRadius: 6,
                    fontWeight: 600,
                  }}
                >
                  <Cpu size={11} />
                  {method ? modelLabel : model_version}
                </span>
              )}
              <span
                style={{
                  display: 'inline-flex',
                  alignItems: 'center',
                  gap: 4,
                  fontSize: 11,
                  fontWeight: 700,
                  padding: '3px 8px',
                  borderRadius: 6,
                  background: ttaEnabled ? 'rgba(139,92,246,0.16)' : 'var(--color-surface-2)',
                  color: ttaEnabled ? 'var(--color-accent)' : 'var(--color-text-faint)',
                  border: ttaEnabled ? '1px solid rgba(139,92,246,0.35)' : '1px solid var(--color-border)',
                }}
              >
                <Sparkles size={10} />
                {ttaEnabled ? 'TTA: Bật (4-View)' : 'TTA: Tắt'}
              </span>
              {prediction.processing_time != null && (
                <span
                  style={{
                    display: 'inline-flex',
                    alignItems: 'center',
                    gap: 4,
                    fontSize: 11,
                    color: 'var(--color-text-muted)',
                    padding: '3px 8px',
                    background: 'rgba(16, 185, 129, 0.08)',
                    border: '1px solid var(--color-border)',
                    borderRadius: 6,
                    fontWeight: 600,
                  }}
                >
                  <Clock size={11} />
                  {prediction.processing_time.toFixed(2)}s
                </span>
              )}
              {prediction.is_unknown_breed && (
                <span
                  style={{
                    display: 'inline-flex',
                    alignItems: 'center',
                    gap: 4,
                    fontSize: 11,
                    fontWeight: 700,
                    padding: '3px 8px',
                    borderRadius: 6,
                    background: 'rgba(245, 158, 11, 0.15)',
                    color: '#f59e0b',
                    border: '1px solid rgba(245, 158, 11, 0.4)',
                  }}
                >
                  <AlertTriangle size={11} />
                  Chưa xác định
                </span>
              )}
            </div>

            <h2
              style={{
                fontSize: 28,
                fontWeight: 800,
                letterSpacing: '-0.02em',
                lineHeight: 1.1,
                color: prediction.is_unknown_breed ? '#f59e0b' : 'inherit',
              }}
            >
              {prediction.is_unknown_breed ? 'Chưa xác định' : breedDisplayName}
            </h2>
            <p style={{ color: 'var(--color-text-muted)', fontSize: 14, marginTop: 6 }}>
              <Award size={13} style={{ display: 'inline', marginRight: 4 }} />
              {prediction.is_unknown_breed
                ? `Tương đồng gần nhất: ${breedDisplayName} (${speciesName})`
                : speciesName}
            </p>
            {prediction.note && (
              <div
                style={{
                  display: 'inline-flex',
                  alignItems: 'center',
                  gap: 5,
                  marginTop: 8,
                  padding: '3px 8px',
                  borderRadius: 6,
                  background: 'rgba(139, 92, 246, 0.08)',
                  border: '1px solid rgba(139, 92, 246, 0.22)',
                  color: 'var(--color-accent)',
                  fontSize: 12,
                  fontWeight: 600,
                }}
              >
                <Tag size={12} />
                <span>Nhãn thực tế: <strong>{prediction.note}</strong></span>
              </div>
            )}
          </div>

          {score !== null && score !== undefined && <ScoreRing score={score} />}
        </div>

        {top_k_results && top_k_results.length > 0 && (
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 16 }}>
              <TrendingUp size={14} color="var(--color-accent)" />
              <span style={{ fontSize: 13, fontWeight: 600, color: 'var(--color-text-muted)' }}>
                Top {top_k_results.length} giống có khả năng cao nhất
              </span>
            </div>
            <PredictionChart topKResults={top_k_results} />
          </div>
        )}
      </div>
    </div>
  );
}
