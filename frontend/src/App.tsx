import { useMemo } from "react";
import { useTrainingStatus } from "./hooks/useTrainingStatus";
import { useMetrics } from "./hooks/useMetrics";
import { StatusCard } from "./components/StatusCard";
import { MetricsChart } from "./components/MetricsChart";
import { MetricsTable } from "./components/MetricsTable";

const SUCCESS_METRIC = "episode/success";
const RETURN_METRIC = "episode/return";

const App = () => {
  const { data: status } = useTrainingStatus();
  const metrics = useMetrics();

  const successRate = useMemo(() => {
    if (metrics.records.length === 0) return null;
    const recent = metrics.records.slice(-50);
    const successes = recent.filter(
      (record) => (record.metrics[SUCCESS_METRIC] ?? 0) > 0.5
    );
    return successes.length / recent.length;
  }, [metrics.records]);

  const avgReturn = useMemo(() => {
    if (metrics.records.length === 0) return null;
    const recent = metrics.records.slice(-50);
    const sum = recent.reduce(
      (acc, record) => acc + (record.metrics[RETURN_METRIC] ?? 0),
      0
    );
    return sum / recent.length;
  }, [metrics.records]);

  return (
    <div className="min-h-screen bg-surface-900/80 pb-16">
      <header className="border-b border-surface-700 bg-surface-900/90 px-8 py-6 backdrop-blur">
        <h1 className="text-3xl font-semibold text-slate-50">
          Panda Door RL Dashboard
        </h1>
        <p className="mt-1 text-sm text-slate-400">
          Monitor training metrics and system status in real time.
        </p>
      </header>
      <main className="mx-auto max-w-6xl space-y-8 px-6 py-8">
        <div className="grid gap-6 md:grid-cols-3">
          <StatusCard
            status={status}
            lastUpdated={
              metrics.lastUpdate
                ? new Date(
                    metrics.lastUpdate.timestamp * 1000
                  ).toLocaleTimeString()
                : undefined
            }
          />
          <div className="rounded-xl border border-surface-700 bg-surface-900/70 p-5 shadow-lg shadow-surface-900/40">
            <h2 className="text-sm uppercase tracking-wide text-slate-400">
              Success Rate (last 50)
            </h2>
            <p className="mt-2 text-3xl font-semibold text-slate-100">
              {successRate != null ? `${(successRate * 100).toFixed(1)}%` : "—"}
            </p>
          </div>
          <div className="rounded-xl border border-surface-700 bg-surface-900/70 p-5 shadow-lg shadow-surface-900/40">
            <h2 className="text-sm uppercase tracking-wide text-slate-400">
              Avg Return (last 50)
            </h2>
            <p className="mt-2 text-3xl font-semibold text-slate-100">
              {avgReturn != null ? avgReturn.toFixed(2) : "—"}
            </p>
          </div>
        </div>

        <MetricsChart
          records={metrics.records}
          metricKeys={[RETURN_METRIC, SUCCESS_METRIC]}
          title="Episode Performance"
          yLabel="Score"
        />
        <MetricsChart
          records={metrics.records}
          metricKeys={["loss/actor_loss", "loss/critic_loss"]}
          title="Loss Curves"
          yLabel="Loss"
        />
        <MetricsTable records={metrics.records} />
      </main>
    </div>
  );
};

export default App;
