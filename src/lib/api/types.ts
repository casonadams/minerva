// API response types matching Minerva server

export interface ChatMessage {
  role: 'user' | 'assistant' | 'system';
  content: string;
}

export interface ChatCompletionRequest {
  model: string;
  messages: ChatMessage[];
  temperature?: number;
  top_p?: number;
  max_tokens?: number;
  stream?: boolean;
}

export interface ChatCompletionResponse {
  id: string;
  object: string;
  created: number;
  model: string;
  choices: Choice[];
  usage: Usage;
}

export interface Choice {
  index: number;
  message: ChatMessage;
  finish_reason: string;
}

export interface Usage {
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
}

export interface ModelInfo {
  id: string;
  object: string;
  created: number;
  owned_by: string;
  context_window?: number;
  max_output_tokens?: number;
}

export interface ModelsListResponse {
  object: string;
  data: ModelInfo[];
}

export interface HealthResponse {
  status: string;
  timestamp: string;
}

export interface ReadinessResponse {
  ready: boolean;
  timestamp: string;
}

export interface MetricsResponse {
  timestamp: string;
  uptime_seconds: number;
  requests: RequestMetrics;
  response_times: ResponseTimeMetrics;
  errors: ErrorMetrics;
  cache: CacheMetrics;
}

export interface RequestMetrics {
  total: number;
  successful: number;
  failed: number;
  rps: number;
}

export interface ResponseTimeMetrics {
  avg_ms: number;
  min_ms: number;
  max_ms: number;
  p50_ms: number;
  p95_ms: number;
  p99_ms: number;
}

export interface ErrorMetrics {
  total: number;
  rate_percent: number;
  top_error: string | null;
}

export interface CacheMetrics {
  hits: number;
  misses: number;
  hit_rate_percent: number;
}
