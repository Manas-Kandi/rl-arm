import {
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
  CartesianGrid,
  Legend
} from "recharts";
import { MetricRecord } from "../lib/api";

interface MetricsChartProps {
  records: MetricRecord[];
  metricKeys: string[];
  title: string;
  yLabel?: string;
}

const COLORS = ["#38bdf8", "#c084fc", "#fb7185", "#fbbf24", "#34d399"];

export const MetricsChart = ({
  records,
  metricKeys,
  title,
  yLabel
}: MetricsChartProps) => {
  const data = records.map((record) => ({
    step: record.step,
    ...metricKeys.reduce((acc, key) => {
      acc[key] = record.metrics[key];
      return acc;
    }, {} as Record<string, number>)
  }));

  return (
    <section className="rounded-xl border border-surface-700 bg-surface-900/70 p-5 shadow-lg shadow-surface-900/40">
      <h3 className="text-lg font-semibold text-slate-100">{title}</h3>
      <div className="mt-4 h-64">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data}>
            <CartesianGrid stroke="#1f2937" strokeDasharray="3 3" />
            <XAxis dataKey="step" stroke="#94a3b8" />
            <YAxis stroke="#94a3b8" label={{ value: yLabel, angle: -90, position: "insideLeft", fill: "#94a3b8" }} />
            <Tooltip
              contentStyle={{
                backgroundColor: "#0f172a",
                borderColor: "#1f2937",
                color: "#e2e8f0"
              }}
            />
            <Legend />
            {metricKeys.map((key, index) => (
              <Line
                key={key}
                type="monotone"
                dataKey={key}
                stroke={COLORS[index % COLORS.length]}
                dot={false}
                strokeWidth={2}
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>
    </section>
  );
};
