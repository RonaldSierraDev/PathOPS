const API_URL: string = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

export const LABELS = ["no_tumor", "tumor"] as const;
export type Label = (typeof LABELS)[number];

export interface HealthResponse {
  status: string;
}

export interface PredictResponse {
  label: Label;
  confidence: number;
  prediction_id?: number;
}

export interface RecentPrediction {
  input_hash: string;
  predicted_label: Label;
  confidence: number;
  latency_ms: number | null;
  created_at: string;
}

export class ApiError extends Error {
  status: number;
  constructor(status: number, detail: string) {
    super(detail);
    this.status = status;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_URL}${path}`, init);
  if (!res.ok) {
    let detail = res.statusText;
    try {
      detail = (await res.json()).detail ?? detail;
    } catch {
      /* non-JSON error body */
    }
    throw new ApiError(res.status, detail);
  }
  return res.json();
}

export const fetchHealth = () => request<HealthResponse>("/health");

export const fetchRecentPredictions = (limit = 200) =>
  request<RecentPrediction[]>(`/predictions/recent?limit=${limit}`);

export interface FeedbackRow {
  input_hash: string;
  corrected_label: Label;
}

export const fetchFeedbackExport = () => request<FeedbackRow[]>("/feedback/export");

export interface ModelVersion {
  model_name: string;
  version: number;
  alias: string | null;
  mlflow_run_id: string | null;
  created_at: string;
  prediction_count: number;
  serving: boolean;
}

export const fetchModels = () => request<ModelVersion[]>("/models");

export interface DriftPoint {
  timestamp: string;
  share: number;
}

export interface DriftReport {
  key: string;
  created_at: string;
  /** Presigned S3 URL; expires an hour after the response was generated. */
  url: string;
}

export interface DriftStatus {
  threshold: number;
  current_share: number | null;
  checked_at: string | null;
  history: DriftPoint[];
  reports: DriftReport[];
}

export const fetchDriftStatus = () => request<DriftStatus>("/monitoring/drift");

export function predict(file: File): Promise<PredictResponse> {
  const body = new FormData();
  body.append("file", file);
  return request<PredictResponse>("/predict", { method: "POST", body });
}

export function submitFeedback(predictionId: number, correctedLabel: Label, note?: string) {
  return request<{ feedback_id: number }>("/feedback", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      prediction_id: predictionId,
      corrected_label: correctedLabel,
      note: note || null,
    }),
  });
}
