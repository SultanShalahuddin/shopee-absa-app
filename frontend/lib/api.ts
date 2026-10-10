export type Sentiment = "tidak_disebut" | "negatif" | "netral" | "positif";

export interface AspectPrediction {
  aspect: string;
  aspect_label: string;
  sentiment: Sentiment;
  confidence: number;
}

export interface PredictionResponse {
  id: string | null;
  review_text: string;
  predictions: AspectPrediction[];
  model_version: string;
  latency_ms: number;
  stored: boolean;
  demo_mode: boolean;
  created_at: string;
}

export interface HistoryItem {
  id: string;
  review_text: string;
  model_version: string;
  latency_ms: number;
  created_at: string;
  predictions: AspectPrediction[];
}

const API_URL = (process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000").replace(/\/$/, "");

export async function predictReview(reviewText: string): Promise<PredictionResponse> {
  const response = await fetch(`${API_URL}/api/v1/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ review_text: reviewText }),
  });
  if (!response.ok) {
    const payload = await response.json().catch(() => ({}));
    throw new Error(payload.detail || "Model gagal menganalisis review.");
  }
  return response.json();
}

export async function fetchHistory(limit = 8): Promise<HistoryItem[]> {
  const response = await fetch(`${API_URL}/api/v1/history?limit=${limit}`, { cache: "no-store" });
  if (!response.ok) return [];
  const payload = await response.json();
  return payload.items || [];
}
