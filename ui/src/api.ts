import type {
  HealthResponse,
  StatsResponse,
  SearchResponse,
  AskResponse,
  IngestResponse,
} from "./types";

const BASE = "http://localhost:8000";

async function request<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, init);
  if (!res.ok) {
    const body = await res.json().catch(() => null);
    throw new Error(body?.detail || `Request failed: ${res.status}`);
  }
  return res.json();
}

export function fetchHealth(): Promise<HealthResponse> {
  return request(`${BASE}/api/health`);
}

export function fetchStats(): Promise<StatsResponse> {
  return request(`${BASE}/api/stats`);
}

export function searchEvents(q: string, topK = 5): Promise<SearchResponse> {
  return request(`${BASE}/api/search?q=${encodeURIComponent(q)}&top_k=${topK}`);
}

export function askQuestion(
  question: string,
  topK = 5,
  model?: string
): Promise<AskResponse> {
  return request(`${BASE}/api/ask`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, top_k: topK, model: model ?? null }),
  });
}

export function triggerIngest(
  sampleSize = 100,
  maxFiles = 1
): Promise<IngestResponse> {
  return request(`${BASE}/api/ingest`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ sample_size: sampleSize, max_files: maxFiles }),
  });
}
