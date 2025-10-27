import { useEffect, useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  MetricRecord,
  fetchMetricHistory,
  fetchLatestMetrics
} from "../lib/api";

interface MetricsState {
  records: MetricRecord[];
  lastUpdate?: MetricRecord;
}

const mergeRecords = (
  existing: MetricRecord[],
  incoming: MetricRecord[]
): MetricRecord[] => {
  const map = new Map<number, MetricRecord>();
  for (const rec of existing) {
    map.set(rec.step, rec);
  }
  for (const rec of incoming) {
    map.set(rec.step, rec);
  }
  return Array.from(map.values()).sort((a, b) => a.step - b.step);
};

export const useMetrics = () => {
  const [streamed, setStreamed] = useState<MetricRecord[]>([]);
  const historyQuery = useQuery({
    queryKey: ["metrics-history"],
    queryFn: () => fetchMetricHistory(200),
    refetchOnWindowFocus: false
  });

  const latestQuery = useQuery({
    queryKey: ["metrics-latest"],
    queryFn: fetchLatestMetrics,
    refetchInterval: 5000
  });

  useEffect(() => {
    const baseURL =
      import.meta.env.VITE_API_URL ?? "http://localhost:8000";
    const wsUrl = baseURL.replace(/^http/, "ws") + "/ws/metrics";
    let ws: WebSocket | null = null;
    try {
      ws = new WebSocket(wsUrl);
      ws.onmessage = (event) => {
        try {
          const payload = JSON.parse(event.data) as MetricRecord;
          setStreamed((prev) => mergeRecords(prev, [payload]));
        } catch (_err) {
          // Ignore malformed messages
        }
      };
    } catch (_err) {
      // WebSocket connection failed; rely on polling
    }
    return () => {
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.close();
      }
    };
  }, []);

  const records = useMemo(() => {
    const historyItems = historyQuery.data?.items ?? [];
    return mergeRecords(historyItems, streamed);
  }, [historyQuery.data, streamed]);

  const lastUpdate = useMemo(() => {
    if (records.length === 0) {
      return latestQuery.data ?? undefined;
    }
    return records[records.length - 1];
  }, [records, latestQuery.data]);

  return {
    loading: historyQuery.isLoading,
    error: historyQuery.error ?? latestQuery.error,
    records,
    lastUpdate
  } as MetricsState & {
    loading: boolean;
    error: unknown;
  };
};
