import React from 'react';

export const AnalyzingSpinner: React.FC = () => {
  return (
    <div style={{
      background: 'var(--color-surface)',
      borderRadius: 24,
      border: '1px solid var(--color-border)',
      boxShadow: '0 10px 40px -10px rgba(0,0,0,0.05)',
      height: '100%',
      minHeight: 400,
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      justifyContent: 'center',
      padding: 40,
      textAlign: 'center',
    }}>
      <style>
        {`
          @keyframes spin-slow {
            from { transform: rotate(0deg); }
            to { transform: rotate(360deg); }
          }
          .custom-spinner {
            width: 80px;
            height: 80px;
            border-radius: 50%;
            border: 4px solid rgba(139,92,246,0.1);
            border-top-color: #8b5cf6;
            animation: spin-slow 1s linear infinite;
            margin-bottom: 24px;
          }
        `}
      </style>
      <div className="custom-spinner" />
      <h2 style={{
        fontSize: 22,
        fontWeight: 700,
        color: 'var(--color-text)',
        marginBottom: 12,
        letterSpacing: '-0.02em',
      }}>
        Analyzing image...
      </h2>
      <p style={{
        fontSize: 15,
        color: 'var(--color-text-muted)',
      }}>
        AI is identifying your pet's breed
      </p>
    </div>
  );
};
