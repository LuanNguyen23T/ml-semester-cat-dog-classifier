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
    return (
      <div style={{
        background: 'rgba(17,18,30,0.95)',
        border: '1px solid rgba(139,92,246,0.3)',
        borderRadius: 10,
        padding: '10px 14px',
        backdropFilter: 'blur(10px)',
      }}>
        <p style={{ color: 'var(--color-text)', fontSize: 13, fontWeight: 600, marginBottom: 4 }}>
          {data.payload.breed_name}
        </p>
        <p style={{ color: 'var(--color-text-muted)', fontSize: 12 }}>
          Species: <span style={{ color: '#818cf8' }}>{data.payload.species}</span>
        </p>
        <p style={{ color: 'var(--color-accent)', fontSize: 14, fontWeight: 700, marginTop: 4 }}>
          Score: {(data.value * 100).toFixed(1)}%
        </p>
      </div>
    );
  }
  return null;
};

export function PredictionChart({ topKResults }: PredictionChartProps) {
  const chartData = topKResults.map((r) => ({
    ...r,
    displayName: r.breed_name.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase()),
    percentage: r.score,
  }));

  return (
    <div style={{ width: '100%', height: 240 }}>
      <ResponsiveContainer width="100%" height="100%">
        <BarChart
          data={chartData}
          layout="vertical"
          margin={{ top: 0, right: 20, bottom: 0, left: 0 }}
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
