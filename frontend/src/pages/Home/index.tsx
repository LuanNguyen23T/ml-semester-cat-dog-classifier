import { useState, useCallback } from 'react';
import { Cat, Zap, Sparkles } from 'lucide-react';
import { ImageUploader } from '../../components/ImageUploader';
import { PredictionResult } from '../../components/PredictionResult';
import { BatchComparisonTable } from '../../components/BatchComparisonTable';
import { ConfigPanel } from '../../components/ConfigPanel';
import { AnalyzingSpinner } from '../../components/AnalyzingSpinner';
import { formatApiError, predictionApi } from '../../services/predictionApi';
import { usePredictionSession } from '../../hooks/usePredictionSession';
import {
  DEFAULT_MODEL_METHOD,
  getModelDisplayName,
  type ModelMethod,
} from '../../types/prediction';

type Step = 'UPLOAD' | 'CONFIG' | 'ANALYZING' | 'RESULT';

export function HomePage() {
  const [step, setStep] = useState<Step>('UPLOAD');
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [uploadError, setUploadError] = useState<string | null>(null);

  const [selectedMethod, setSelectedMethod] = useState<ModelMethod>(DEFAULT_MODEL_METHOD);
  const [useTta, setUseTta] = useState<boolean>(false);
  const [selectedResultIndex, setSelectedResultIndex] = useState(0);

  const { predictions, activeIds, error: pollError, startPolling, reset } = usePredictionSession();
  const sessionStep: Step | null = activeIds.length > 0
    ? (predictions.length > 0 ? 'RESULT' : 'ANALYZING')
    : null;
  const visibleStep = sessionStep ?? step;
  const visiblePreviewUrl = previewUrl ?? predictions[0]?.image_key ?? null;

  const handleFileSelected = useCallback((file: File) => {
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
    setSelectedFile(file);
    setStep('CONFIG');
    setUploadError(null);
  }, []);

  const handleClassify = useCallback(async () => {
    if (!selectedFile) return;

    setStep('ANALYZING');
    setUploadError(null);
    setSelectedResultIndex(0);

    try {
      if (selectedMethod === 'all') {
        const createdBatch = await predictionApi.createPredictionBatch(selectedFile, useTta);
        startPolling(createdBatch.map((p) => p.id));
      } else {
        const created = await predictionApi.createPrediction(selectedFile, selectedMethod, useTta);
        startPolling([created.id]);
      }
    } catch (err: unknown) {
      setUploadError(formatApiError(err, 'Tải ảnh lên thất bại. Vui lòng thử lại.'));
      setStep('CONFIG');
    }
  }, [selectedFile, selectedMethod, useTta, startPolling]);

  const handleReset = useCallback(() => {
    reset();
    setStep('UPLOAD');
    setSelectedFile(null);
    setUploadError(null);
    setSelectedResultIndex(0);
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
      setPreviewUrl(null);
    }
  }, [reset, previewUrl]);

  return (
    <div
      style={{
        minHeight: '100vh',
        display: 'flex',
        flexDirection: 'column',
        padding: '0 16px 60px',
        maxWidth: 1100,
        margin: '0 auto',
        width: '100%',
      }}
    >
      {/* Hero Header - Only show in UPLOAD step */}
      {visibleStep === 'UPLOAD' && (
        <header style={{ textAlign: 'center', padding: '52px 0 40px' }} className="animate-fade-in">
          <div
            style={{
              display: 'inline-flex',
              alignItems: 'center',
              gap: 8,
              padding: '6px 16px',
              background: 'rgba(139,92,246,0.1)',
              border: '1px solid rgba(139,92,246,0.25)',
              borderRadius: 999,
              fontSize: 13,
              color: 'var(--color-accent)',
              fontWeight: 600,
              marginBottom: 24,
            }}
          >
            <Sparkles size={14} />
            Nhận diện 37 giống chó &amp; mèo bằng AI
          </div>

          <h1
            style={{
              fontSize: 'clamp(32px, 5vw, 60px)',
              fontWeight: 900,
              letterSpacing: '-0.03em',
              lineHeight: 1.05,
              marginBottom: 20,
            }}
          >
            <span className="gradient-text">Discover</span>{' '}
            <span style={{ color: 'var(--color-text)' }}>Your Pet&apos;s Breed</span>
          </h1>

          <p
            style={{
              color: 'var(--color-text-muted)',
              fontSize: 17,
              maxWidth: 580,
              margin: '0 auto',
              lineHeight: 1.7,
            }}
          >
            Tải lên ảnh chó hoặc mèo để nhận diện nhanh và chính xác thuộc 37 giống phổ biến với 5 mô hình mạng nơ-ron tự huấn luyện.
          </p>

          <div
            style={{
              display: 'flex',
              gap: 28,
              justifyContent: 'center',
              marginTop: 32,
              flexWrap: 'wrap',
            }}
          >
            {[
              { label: '5 Mô hình CNN', sub: 'So sánh trực tiếp', icon: <Zap size={16} /> },
              { label: 'Dual-Stream', sub: 'Thân + Mặt (86.5% TTA)', icon: <Sparkles size={16} /> },
              { label: '37 Giống', sub: '12 Mèo & 25 Chó', icon: <Cat size={16} /> },
            ].map((stat) => (
              <div
                key={stat.label}
                style={{
                  display: 'flex',
                  flexDirection: 'column',
                  alignItems: 'center',
                  gap: 4,
                }}
              >
                <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                  <span style={{ color: 'var(--color-accent)' }}>{stat.icon}</span>
                  <span style={{ fontSize: 20, fontWeight: 800, color: 'var(--color-text)' }}>
                    {stat.label}
                  </span>
                </div>
                <span style={{ fontSize: 12, color: 'var(--color-text-faint)' }}>{stat.sub}</span>
              </div>
            ))}
          </div>
        </header>
      )}

      {/* Main content */}
      <div style={{ marginTop: visibleStep === 'UPLOAD' ? 0 : 32 }} className="animate-fade-in">
        {visibleStep === 'UPLOAD' && (
          <div style={{ maxWidth: 680, margin: '0 auto', width: '100%' }}>
            <ImageUploader onFileSelected={handleFileSelected} />
          </div>
        )}

        {visibleStep !== 'UPLOAD' && (
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fit, minmax(min(100%, 340px), 1fr))',
              gap: 28,
              alignItems: 'start',
              maxWidth: 1040,
              margin: '0 auto',
              width: '100%',
            }}
          >
            {/* Left Column: Image Preview */}
            <div style={{ minWidth: 0 }}>
              <h3
                style={{
                  fontSize: 13,
                  fontWeight: 600,
                  color: 'var(--color-text-muted)',
                  textTransform: 'uppercase',
                  letterSpacing: '0.08em',
                  marginBottom: 14,
                }}
              >
                Ảnh đã tải lên
              </h3>

              <div
                style={{
                  borderRadius: 24,
                  overflow: 'hidden',
                  border: '1px solid var(--color-border)',
                  background: 'var(--color-surface)',
                  boxShadow: '0 10px 40px -10px rgba(0,0,0,0.05)',
                  aspectRatio: '1 / 1',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  position: 'relative',
                }}
              >
                {visiblePreviewUrl && (
                  <img
                    src={visiblePreviewUrl}
                    alt="Uploaded Pet"
                    style={{
                      width: '100%',
                      height: '100%',
                      objectFit: 'cover',
                      display: 'block',
                      opacity: visibleStep === 'ANALYZING' ? 0.7 : 1,
                      transition: 'opacity 0.3s',
                    }}
                  />
                )}

                {visibleStep === 'ANALYZING' && (
                  <>
                    <style>
                      {`
                        @keyframes scan-line {
                          0% { transform: translateY(-100%); opacity: 0; }
                          10% { opacity: 1; }
                          90% { opacity: 1; }
                          100% { transform: translateY(600%); opacity: 0; }
                        }
                      `}
                    </style>
                    <div
                      style={{
                        position: 'absolute',
                        top: 0,
                        left: 0,
                        right: 0,
                        height: '20%',
                        background:
                          'linear-gradient(to bottom, transparent, rgba(99,102,241,0.2), rgba(99,102,241,0.6))',
                        borderBottom: '3px solid #6366f1',
                        animation: 'scan-line 2s linear infinite',
                        zIndex: 20,
                        pointerEvents: 'none',
                        boxShadow: '0 4px 20px rgba(99,102,241,0.5)',
                      }}
                    />
                  </>
                )}
              </div>

              <button
                type="button"
                className="btn btn-ghost"
                onClick={handleReset}
                style={{ marginTop: 14, width: '100%' }}
              >
                Tải ảnh khác
              </button>
            </div>

            {/* Right Column: Dynamic Panel */}
            <div style={{ minWidth: 0 }}>
              <h3
                style={{
                  fontSize: 13,
                  fontWeight: 600,
                  color: 'var(--color-text-muted)',
                  textTransform: 'uppercase',
                  letterSpacing: '0.08em',
                  marginBottom: 14,
                  visibility: visibleStep === 'RESULT' ? 'visible' : 'hidden',
                }}
              >
                {visibleStep === 'RESULT' ? 'Kết quả phân loại' : 'Cấu hình'}
              </h3>

              {visibleStep === 'CONFIG' && (
                <ConfigPanel
                  selectedMethod={selectedMethod}
                  onSelectMethod={setSelectedMethod}
                  useTta={useTta}
                  onToggleTta={setUseTta}
                  onClassify={handleClassify}
                />
              )}

              {visibleStep === 'ANALYZING' && <AnalyzingSpinner />}

              {visibleStep === 'RESULT' && predictions.length > 0 && (
                <div style={{ display: 'flex', flexDirection: 'column', gap: 16, minWidth: 0 }}>
                  {predictions.length > 1 && (
                    <>
                      <BatchComparisonTable
                        predictions={predictions}
                        selectedIndex={selectedResultIndex}
                        onSelectIndex={setSelectedResultIndex}
                      />

                      <div
                        style={{
                          display: 'flex',
                          gap: 8,
                          overflowX: 'auto',
                          paddingBottom: 6,
                          scrollbarWidth: 'thin',
                        }}
                      >
                        {predictions.map((p, idx) => (
                          <button
                            key={p.id}
                            type="button"
                            onClick={() => setSelectedResultIndex(idx)}
                            style={{
                              padding: '6px 12px',
                              borderRadius: 20,
                              border: '1px solid var(--color-border)',
                              background:
                                idx === selectedResultIndex
                                  ? 'var(--color-accent)'
                                  : 'var(--color-surface)',
                              color: idx === selectedResultIndex ? 'white' : 'var(--color-text)',
                              fontSize: 12,
                              fontWeight: 600,
                              cursor: 'pointer',
                              whiteSpace: 'nowrap',
                              display: 'flex',
                              alignItems: 'center',
                              gap: 6,
                              transition: 'all 0.2s ease',
                              flexShrink: 0,
                            }}
                          >
                            {getModelDisplayName(p.method)}
                            {p.status === 'COMPLETED' ? ' ✓' : p.status === 'FAILED' ? ' ✗' : ' ...'}
                          </button>
                        ))}
                      </div>
                    </>
                  )}

                  {predictions[selectedResultIndex] && (
                    <PredictionResult prediction={predictions[selectedResultIndex]} />
                  )}
                </div>
              )}

              {/* Errors */}
              {(uploadError || pollError) && (
                <div
                  role="alert"
                  style={{
                    marginTop: 16,
                    padding: '14px 16px',
                    background: 'rgba(239,68,68,0.1)',
                    border: '1px solid rgba(239,68,68,0.35)',
                    borderRadius: 12,
                    color: 'var(--color-error)',
                    fontSize: 14,
                    fontWeight: 500,
                    overflowWrap: 'anywhere',
                  }}
                >
                  ⚠ {uploadError || pollError}
                </div>
              )}
            </div>
          </div>
        )}
      </div>

      {/* Footer - Only show on Home */}
      {visibleStep === 'UPLOAD' && (
        <footer
          style={{
            marginTop: '72px',
            padding: '40px 0',
            borderTop: '1px solid var(--color-border)',
            textAlign: 'center',
          }}
        >
          <h2 style={{ fontSize: 24, fontWeight: 700, color: 'var(--color-text)', marginBottom: 16 }}>
            Cách hoạt động
          </h2>
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fit, minmax(min(100%, 240px), 1fr))',
              gap: 24,
              maxWidth: 900,
              margin: '0 auto',
              textAlign: 'left',
            }}
          >
            <div
              style={{
                background: 'var(--color-surface)',
                padding: 24,
                borderRadius: 16,
                border: '1px solid var(--color-border)',
                boxShadow: '0 4px 20px rgba(0,0,0,0.02)',
              }}
            >
              <h3
                style={{
                  fontSize: 16,
                  fontWeight: 700,
                  color: 'var(--color-text)',
                  marginBottom: 8,
                  display: 'flex',
                  alignItems: 'center',
                  gap: 8,
                }}
              >
                <span
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    width: 24,
                    height: 24,
                    background: 'var(--color-accent-glow)',
                    color: 'var(--color-accent)',
                    borderRadius: '50%',
                    fontSize: 12,
                  }}
                >
                  1
                </span>
                Tải ảnh lên
              </h3>
              <p style={{ fontSize: 14, color: 'var(--color-text-muted)', lineHeight: 1.6 }}>
                Chọn hoặc chụp ảnh chó/mèo rõ nét. Hệ thống tự động tách nền thông minh trước khi đưa vào mạng phân loại.
              </p>
            </div>
            <div
              style={{
                background: 'var(--color-surface)',
                padding: 24,
                borderRadius: 16,
                border: '1px solid var(--color-border)',
                boxShadow: '0 4px 20px rgba(0,0,0,0.02)',
              }}
            >
              <h3
                style={{
                  fontSize: 16,
                  fontWeight: 700,
                  color: 'var(--color-text)',
                  marginBottom: 8,
                  display: 'flex',
                  alignItems: 'center',
                  gap: 8,
                }}
              >
                <span
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    width: 24,
                    height: 24,
                    background: 'var(--color-accent-glow)',
                    color: 'var(--color-accent)',
                    borderRadius: '50%',
                    fontSize: 12,
                  }}
                >
                  2
                </span>
                Chọn mô hình AI
              </h3>
              <p style={{ fontSize: 14, color: 'var(--color-text-muted)', lineHeight: 1.6 }}>
                Chọn 1 trong 5 mô hình CNN (mặc định Dual-Stream EfficientNet-B0 cho độ chính xác cao nhất) hoặc chạy All Models để so sánh.
              </p>
            </div>
            <div
              style={{
                background: 'var(--color-surface)',
                padding: 24,
                borderRadius: 16,
                border: '1px solid var(--color-border)',
                boxShadow: '0 4px 20px rgba(0,0,0,0.02)',
              }}
            >
              <h3
                style={{
                  fontSize: 16,
                  fontWeight: 700,
                  color: 'var(--color-text)',
                  marginBottom: 8,
                  display: 'flex',
                  alignItems: 'center',
                  gap: 8,
                }}
              >
                <span
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    width: 24,
                    height: 24,
                    background: 'var(--color-accent-glow)',
                    color: 'var(--color-accent)',
                    borderRadius: '50%',
                    fontSize: 12,
                  }}
                >
                  3
                </span>
                Xem kết quả Top-5
              </h3>
              <p style={{ fontSize: 14, color: 'var(--color-text-muted)', lineHeight: 1.6 }}>
                Nhận kết quả giống loài Top-1, độ tin cậy, biểu đồ xác suất Top-5 và đối chiếu đồng thuận giữa các mô hình.
              </p>
            </div>
          </div>
        </footer>
      )}
    </div>
  );
}
