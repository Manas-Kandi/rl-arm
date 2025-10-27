import { TrainingStatus } from "../lib/api";

interface StatusCardProps {
  status?: TrainingStatus;
  lastUpdated?: string;
}

export const StatusCard = ({ status, lastUpdated }: StatusCardProps) => {
  const running = status?.running ?? false;
  const startedAt =
    status?.start_time != null
      ? new Date(status.start_time * 1000).toLocaleString()
      : "—";

  return (
    <section className="rounded-xl border border-surface-700 bg-surface-900/70 p-5 shadow-lg shadow-surface-900/40">
      <div className="flex items-center justify-between">
        <div>
          <p className="text-sm uppercase tracking-wide text-slate-400">
            Training Status
          </p>
          <h2 className="mt-1 text-2xl font-semibold text-slate-50">
            {running ? "Running" : "Stopped"}
          </h2>
        </div>
        <span
          className={`inline-flex items-center rounded-full px-3 py-1 text-sm font-medium ${
            running
              ? "bg-emerald-500/10 text-emerald-300 border border-emerald-500/40"
              : "bg-rose-500/10 text-rose-300 border border-rose-500/40"
          }`}
        >
          {running ? "Active" : "Idle"}
        </span>
      </div>
      <dl className="mt-4 grid grid-cols-2 gap-4 text-sm text-slate-300">
        <div>
          <dt className="text-slate-400">PID</dt>
          <dd className="font-mono text-slate-200">
            {status?.pid ?? "—"}
          </dd>
        </div>
        <div>
          <dt className="text-slate-400">Started</dt>
          <dd className="text-slate-200">{startedAt}</dd>
        </div>
        <div className="col-span-2">
          <dt className="text-slate-400">Command</dt>
          <dd className="font-mono text-xs text-slate-300">
            {status?.command?.join(" ") ?? "—"}
          </dd>
        </div>
      </dl>
      <p className="mt-4 text-xs text-slate-500">
        Last updated: {lastUpdated ?? "—"}
      </p>
    </section>
  );
};
