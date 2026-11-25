import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import type { TopKResult } from '../../types/prediction';

interface PredictionChartProps {
  topKResults: TopKResult[];
}

const COLORS = [
  '#8b5cf6',
  '#6366f1',
  '#3b82f6',
  '#06b6d4',
  '#10b981',
];

interface TooltipPayloadEntry {
  value: number;
  payload: TopKResult;
}

interface CustomTooltipProps {
  active?: boolean;
  payload?: TooltipPayloadEntry[];
}

const CustomTooltip = ({ active, payload }: CustomTooltipProps) => {
  if (active && payload && payload.length) {
    const data = payload[0];
    const breedDisplayName = (data.payload.breed_name || '')
      .replace(/_/g, ' ')
      .replace(/\b\w/g, (c) => c.toUpperCase());

    return (
      <div
        style={{
          background: '#ffffff',
          color: '#0f172a',
          border: '1px solid rgba(139,92,246,0.3)',
          borderRadius: 10,
          padding: '10px 14px',
          boxShadow: '0 8px 24px rgba(0, 0, 0, 0.16)',
        }}
      >
        <p style={{ color: '#0f172a', fontSize: 14, fontWeight: 700, marginBottom: 4 }}>
          {breedDisplayName}
        </p>
        <p style={{ color: '#64748b', fontSize: 12, marginBottom: 2 }}>
          Species: <span style={{ color: '#4f46e5', fontWeight: 600 }}>{data.payload.species}</span>
        </p>
        <p style={{ color: '#7c3aed', fontSize: 14, fontWeight: 800, marginTop: 4 }}>
          Score: {(data.value * 100).toFixed(1)}%
        </p>
      </div>
    );
  }
  return null;
};

export function PredictionChart({ topKResults }: PredictionChartProps) {
  const validResults = (topKResults || []).filter(
    (r): r is TopKResult => Boolean(r && typeof r.breed_name === 'string' && typeof r.score === 'number')
  );

  const chartData = validResults.map((r) => ({
    ...r,
    displayName: r.breed_name.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase()),
    percentage: r.score,
  }));

  return (
    <div style={{ width: '100%', height: 240, outline: 'none' }} tabIndex={-1}>
      <ResponsiveContainer width="100%" height="100%" style={{ outline: 'none' }}>
        <BarChart
          data={chartData}
          layout="vertical"
          margin={{ top: 0, right: 20, bottom: 0, left: 0 }}
          style={{ outline: 'none' }}
        >
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(139,92,246,0.1)" horizontal={false} />
          <XAxis
            type="number"
            domain={[0, 1]}
            tickFormatter={(v) => `${(v * 100).toFixed(0)}%`}
            tick={{ fill: 'var(--color-text-muted)', fontSize: 11 }}
            axisLine={{ stroke: 'rgba(139,92,246,0.2)' }}
            tickLine={false}
          />
          <YAxis
            type="category"
            dataKey="displayName"
            width={120}
            tick={{ fill: 'var(--color-text)', fontSize: 12 }}
            axisLine={false}
            tickLine={false}
          />
          <Tooltip content={<CustomTooltip />} cursor={{ fill: 'rgba(139,92,246,0.05)' }} />
          <Bar dataKey="percentage" radius={[0, 6, 6, 0]} maxBarSize={28}>
            {chartData.map((_, index) => (
              <Cell
                key={`cell-${index}`}
                fill={COLORS[index % COLORS.length]}
                fillOpacity={index === 0 ? 1 : 0.6}
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
