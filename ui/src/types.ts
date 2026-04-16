export interface HealthResponse {
  status: "ok" | "degraded" | "error";
  pinecone_connected: boolean;
  hf_available: boolean;
  hf_model: string;
}

export interface StatsResponse {
  vector_count: number;
  namespace_count: number;
  processed_files: number;
  index_fullness: number;
}

export interface EventMetadata {
  event_date: string;
  country_code: string;
  event_code: string;
  event_base_code: string;
  actor1_name: string;
  actor2_name: string;
  avg_tone: number;
  num_mentions: number;
  source_url: string;
}

export interface SearchResult {
  vector_id: string;
  similarity_score: number;
  metadata: EventMetadata;
}

export interface SearchResponse {
  query: string;
  top_k: number;
  results: SearchResult[];
}

export interface AskResponse {
  answer: string | null;
  events: SearchResult[];
  model: string;
  error?: string;
  message?: string;
}

export interface IngestResponse {
  summaries: unknown[];
  files_processed: number;
}
