import React, { useEffect, useState } from 'react';
import {
  DEFAULT_MODEL_METHOD,
  FALLBACK_MODELS,
  getModelStreamGroup,
  type ModelInfo,
  type ModelMethod,
  type SingleModelMethod,
} from '../../types/prediction';
import { predictionApi } from '../../services/predictionApi';
import { Cpu, Sparkles, ImageIcon, Layers, Star } from 'lucide-react';

interface ConfigPanelProps {
  selectedMethod: ModelMethod;
  onSelectMethod: (method: ModelMethod) => void;
  useTta: boolean;
  onToggleTta: (enabled: boolean) => void;
  onClassify: () => void;
}

function ModelOptionCard({
  model,
  selected,
  onSelect,
}: {
  model: ModelInfo;
  selected: boolean;
  onSelect: () => void;
}) {
  const singleAcc = model.accuracy_single ?? model.accuracy;
  const ttaAcc = model.accuracy_tta;

  return (
    <button
      type="button"
      disabled={!model.available}
      onClick={onSelect}
      style={{
        width: '100%',
        textAlign: 'left',
        padding: '12px 14px',
        borderRadius: 12,
        border: selected
          ? '2px solid var(--color-accent)'
          : '1px solid var(--color-border)',
        background: selected
          ? 'rgba(139,92,246,0.10)'
          : 'var(--color-bg)',
        opacity: model.available ? 1 : 0.48,
        cursor: model.available ? 'pointer' : 'not-allowed',
        transition: 'all 0.18s ease',
        display: 'flex',
        flexDirection: 'column',
        gap: 6,
      }}
    >
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 8, flexWrap: 'wrap' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, minWidth: 0 }}>
          <span
            style={{
              width: 14,
              height: 14,
              borderRadius: '50%',
              flexShrink: 0,
              border: selected ? '4px solid var(--color-accent)' : '2px solid var(--color-text-faint)',
              background: 'var(--color-surface)',
            }}
          />
          <span style={{ fontSize: 14, fontWeight: 700, color: 'var(--color-text)' }}>
            {model.name}
          </span>
          {model.recommended && (
            <span
              style={{
                display: 'inline-flex',
                alignItems: 'center',
                gap: 4,
                padding: '2px 8px',
                borderRadius: 999,
                fontSize: 10,
                fontWeight: 700,
                background: 'rgba(139,92,246,0.16)',
                color: 'var(--color-accent)',
                border: '1px solid rgba(139,92,246,0.35)',
              }}
            >
              <Star size={10} /> Recommended
            </span>
          )}
          {!model.available && (
            <span
              style={{
                padding: '2px 8px',
                borderRadius: 999,
                fontSize: 10,
                fontWeight: 700,
                background: 'rgba(239,68,68,0.12)',
                color: 'var(--color-error)',
                border: '1px solid rgba(239,68,68,0.28)',
              }}
            >
              Không khả dụng
            </span>
          )}
        </div>

        <span
          style={{
            fontSize: 11,
            fontWeight: 700,
            color: 'var(--color-accent)',
            whiteSpace: 'nowrap',
          }}
        >
          Top-1: {singleAcc.toFixed(2)}%
          {ttaAcc != null ? ` · TTA ${ttaAcc.toFixed(2)}%` : ''}
        </span>
      </div>

      <p style={{ fontSize: 12, color: 'var(--color-text-muted)', lineHeight: 1.45, paddingLeft: 22 }}>
        {model.description}
      </p>
    </button>
  );
}

export const ConfigPanel: React.FC<ConfigPanelProps> = ({
  selectedMethod,
  onSelectMethod,
  useTta,
  onToggleTta,
  onClassify,
}) => {
  const [models, setModels] = useState<ModelInfo[]>(FALLBACK_MODELS);
  const [lastSingleMethod, setLastSingleMethod] = useState<SingleModelMethod>(DEFAULT_MODEL_METHOD);

  useEffect(() => {
    predictionApi
      .listModels()
      .then((data) => {
        if (data.length > 0) {
          setModels(data);
        }
      })
      .catch((err) => console.error('Failed to fetch models', err));
  }, []);

  const handleSelectSingle = (method: SingleModelMethod) => {
    setLastSingleMethod(method);
    onSelectMethod(method);
  };

  const singleStreamModels = models.filter((m) => getModelStreamGroup(m.id) === 'single-stream');
  const dualStreamModels = models.filter((m) => getModelStreamGroup(m.id) === 'dual-stream');

  const selectedModelInfo =
    selectedMethod !== 'all' ? models.find((m) => m.id === selectedMethod) : undefined;
  const isSingleSelectionUnavailable =
    selectedMethod !== 'all' && selectedModelInfo !== undefined && !selectedModelInfo.available;

  const ensureAvailableSingleMethod = (): SingleModelMethod => {
    const preferred = models.find((m) => m.id === lastSingleMethod && m.available);
    if (preferred) return preferred.id;
    const firstAvailable = models.find((m) => m.available);
    return firstAvailable?.id ?? DEFAULT_MODEL_METHOD;
  };

  return (
    <div
      style={{
        background: 'var(--color-surface)',
        borderRadius: 24,
        border: '1px solid var(--color-border)',
        boxShadow: '0 10px 40px -10px rgba(0,0,0,0.05)',
        height: '100%',
        display: 'flex',
        flexDirection: 'column',
        padding: 24,
      }}
    >
      <h2
        style={{
          fontSize: 20,
          fontWeight: 700,
          color: 'var(--color-text)',
          marginBottom: 20,
          letterSpacing: '-0.01em',
          display: 'flex',
          alignItems: 'center',
          gap: 8,
        }}
      >
        <Sparkles size={20} className="text-indigo-500" />
        AI Configuration
      </h2>

      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: 20 }}>
        {/* Classification Mode */}
        <div>
          <label
            style={{
              display: 'block',
              fontSize: 12,
              fontWeight: 700,
              color: 'var(--color-text-muted)',
              marginBottom: 8,
              textTransform: 'uppercase',
              letterSpacing: '0.05em',
            }}
          >
            Chế độ nhận diện
          </label>
          <div style={{ display: 'flex', gap: 10 }}>
            <button
              type="button"
              onClick={() => onSelectMethod(ensureAvailableSingleMethod())}
              className={`flex-1 px-4 py-3 border-2 rounded-xl font-semibold text-sm flex items-center justify-center gap-2 transition-all ${
                selectedMethod !== 'all'
                  ? 'bg-indigo-50 dark:bg-indigo-900/20 border-indigo-500 text-indigo-700 dark:text-indigo-300'
                  : 'bg-slate-50 dark:bg-slate-800/50 border-transparent text-slate-500 hover:bg-slate-100 dark:hover:bg-slate-800'
              }`}
            >
              <Cpu size={16} /> Single Model
            </button>
            <button
              type="button"
              onClick={() => onSelectMethod('all')}
              className={`flex-1 px-4 py-3 border-2 rounded-xl font-semibold text-sm flex items-center justify-center gap-2 transition-all ${
                selectedMethod === 'all'
                  ? 'bg-indigo-50 dark:bg-indigo-900/20 border-indigo-500 text-indigo-700 dark:text-indigo-300'
                  : 'bg-slate-50 dark:bg-slate-800/50 border-transparent text-slate-500 hover:bg-slate-100 dark:hover:bg-slate-800'
              }`}
            >
              <Layers size={16} /> All Models
            </button>
          </div>
        </div>

        {/* 5 Models List grouped by Dual-Stream & Single-Stream */}
        {selectedMethod !== 'all' ? (
          <div className="animate-fade-in" style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
            {dualStreamModels.length > 0 && (
              <div>
                <div
                  style={{
                    fontSize: 11,
                    fontWeight: 700,
                    color: 'var(--color-text-muted)',
                    textTransform: 'uppercase',
                    letterSpacing: '0.06em',
                    marginBottom: 8,
                  }}
                >
                  Hai luồng (Dual-Stream)
                </div>
                <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                  {dualStreamModels.map((model) => (
                    <ModelOptionCard
                      key={model.id}
                      model={model}
                      selected={selectedMethod === model.id}
                      onSelect={() => handleSelectSingle(model.id)}
                    />
                  ))}
                </div>
              </div>
            )}

            {singleStreamModels.length > 0 && (
              <div>
                <div
                  style={{
                    fontSize: 11,
                    fontWeight: 700,
                    color: 'var(--color-text-muted)',
                    textTransform: 'uppercase',
                    letterSpacing: '0.06em',
                    marginBottom: 8,
                  }}
                >
                  Một luồng (Single-Stream)
                </div>
                <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                  {singleStreamModels.map((model) => (
                    <ModelOptionCard
                      key={model.id}
                      model={model}
                      selected={selectedMethod === model.id}
                      onSelect={() => handleSelectSingle(model.id)}
                    />
                  ))}
                </div>
              </div>
            )}
          </div>
        ) : (
          <div
            className="animate-fade-in"
            style={{
              padding: '14px 16px',
              borderRadius: 12,
              background: 'rgba(139,92,246,0.06)',
              border: '1px solid var(--color-border)',
              fontSize: 13,
              color: 'var(--color-text-muted)',
              lineHeight: 1.55,
            }}
          >
            Chế độ <strong style={{ color: 'var(--color-text)' }}>All Models</strong> sẽ chạy đồng thời cả 5 mô hình
            trên cùng một ảnh (tách nền 1 lần dùng chung) và hiển thị bảng đối chiếu độ đồng thuận Top-1.
          </div>
        )}

        {/* TTA Switch */}
        <div
          style={{
            padding: '14px 16px',
            borderRadius: 14,
            border: '1px solid var(--color-border)',
            background: 'var(--color-surface-2)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            gap: 14,
          }}
        >
          <div style={{ minWidth: 0 }}>
            <div style={{ fontSize: 14, fontWeight: 700, color: 'var(--color-text)' }}>
              Độ chính xác cao (TTA)
            </div>
            <p style={{ fontSize: 12, color: 'var(--color-text-muted)', marginTop: 3, lineHeight: 1.45 }}>
              Chính xác hơn khoảng 0,5–0,9% nhưng chậm hơn khoảng 3–4 lần
            </p>
          </div>
          <button
            type="button"
            role="switch"
            aria-checked={useTta}
            aria-label="Độ chính xác cao (TTA)"
            onClick={() => onToggleTta(!useTta)}
            style={{
              width: 48,
              height: 28,
              borderRadius: 999,
              border: 'none',
              padding: 3,
              flexShrink: 0,
              cursor: 'pointer',
              background: useTta ? 'var(--color-accent)' : 'rgba(148,163,184,0.45)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: useTta ? 'flex-end' : 'flex-start',
              transition: 'background 0.2s ease',
            }}
          >
            <span
              style={{
                width: 22,
                height: 22,
                borderRadius: '50%',
                background: '#ffffff',
                boxShadow: '0 1px 4px rgba(0,0,0,0.2)',
              }}
            />
          </button>
        </div>
      </div>

      <button
        type="button"
        disabled={isSingleSelectionUnavailable}
        onClick={onClassify}
        style={{
          width: '100%',
          padding: '16px',
          background: isSingleSelectionUnavailable
            ? 'rgba(148,163,184,0.4)'
            : 'linear-gradient(135deg, #8b5cf6 0%, #6366f1 100%)',
          color: 'white',
          border: 'none',
          borderRadius: 14,
          fontSize: 16,
          fontWeight: 700,
          cursor: isSingleSelectionUnavailable ? 'not-allowed' : 'pointer',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          gap: 8,
          boxShadow: isSingleSelectionUnavailable ? 'none' : '0 4px 14px rgba(99,102,241,0.3)',
          marginTop: 20,
          transition: 'all 0.2s ease',
        }}
      >
        <ImageIcon size={20} />
        Classify Now
      </button>
    </div>
  );
};
