import { MetricRecord } from "../lib/api";

interface MetricsTableProps {
  records: MetricRecord[];
  limit?: number;
}

const IMPORTANT_KEYS = [
  "episode/return",
  "episode/success",
  "env/door_angle",
  "loss/actor_loss",
  "loss/critic_loss"
];

export const MetricsTable = ({ records, limit = 10 }: MetricsTableProps) => {
  const latest = records.slice(-limit).reverse();
  return (
    <section className="rounded-xl border border-surface-700 bg-surface-900/70 p-5 shadow-lg shadow-surface-900/40">
      <h3 className="text-lg font-semibold text-slate-100">
        Latest Metrics
      </h3>
      <div className="mt-4 overflow-x-auto">
        <table className="min-w-full divide-y divide-slate-800 text-sm">
          <thead className="bg-surface-800/60 text-slate-300">
            <tr>
              <th className="px-3 py-2 text-left">Step</th>
              <th className="px-3 py-2 text-left">Timestamp</th>
              {IMPORTANT_KEYS.map((key) => (
                <th key={key} className="px-3 py-2 text-left">
                  {key}
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-800 text-slate-200">
            {latest.map((record) => (
              <tr key={record.step}>
                <td className="px-3 py-2 font-mono">{record.step}</td>
                <td className="px-3 py-2">
                  {new Date(record.timestamp * 1000).toLocaleTimeString()}
                </td>
                {IMPORTANT_KEYS.map((key) => (
                  <td key={key} className="px-3 py-2 font-mono">
                    {record.metrics[key]?.toFixed(4) ?? "—"}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
};
