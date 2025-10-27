import axios from "axios";

const baseURL = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

export const api = axios.create({
  baseURL,
  timeout: 8000
});

export interface TrainingStatus {
  running: boolean;
  pid?: number | null;
  command?: string[];
  start_time?: number | null;
  return_code?: number | null;
}

export interface MetricRecord {
  timestamp: number;
  run_id?: string | null;
  step: number;
  metrics: Record<string, number>;
}

export const fetchStatus = async (): Promise<TrainingStatus> => {
  const { data } = await api.get<TrainingStatus>("/api/status");
  return data;
};

export const fetchLatestMetrics = async (): Promise<MetricRecord | null> => {
  const { data } = await api.get<MetricRecord | null>("/api/metrics/latest");
  return data;
};

export interface MetricHistoryResponse {
  items: MetricRecord[];
  total: number;
}

export const fetchMetricHistory = async (
  limit = 200
): Promise<MetricHistoryResponse> => {
  const { data } = await api.get<MetricHistoryResponse>(
    "/api/metrics/history",
    { params: { limit } }
  );
  return data;
};
