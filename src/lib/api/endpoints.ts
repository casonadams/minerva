// API endpoint methods for Minerva server
// Groups related operations by resource

import { apiClient } from './client';
import type {
  ChatCompletionRequest,
  ChatCompletionResponse,
  ModelsListResponse,
  ModelInfo,
  HealthResponse,
  ReadinessResponse,
  MetricsResponse,
} from './types';

/**
 * Chat completion endpoints
 */
export const chat = {
  /**
   * Create a chat completion
   */
  completions: async (req: ChatCompletionRequest): Promise<ChatCompletionResponse> =>
    apiClient.post('/v1/chat/completions', req),

  /**
   * Create a streaming chat completion (returns Response for streaming)
   */
  completionsStream: async (req: ChatCompletionRequest): Promise<Response> => {
    const isDev = typeof window !== 'undefined' && 
                  (window.location.hostname === 'localhost' || 
                   window.location.hostname === '127.0.0.1');
    const baseUrl = isDev 
      ? `http://localhost:${import.meta.env.VITE_API_PORT || '3000'}`
      : '/api';
    
    return fetch(`${baseUrl}/v1/chat/completions`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'x-client-id': localStorage.getItem('minerva_client_id') || '',
      },
      body: JSON.stringify(req),
    });
  },
};

/**
 * Model management endpoints
 */
export const models = {
  /**
   * List available models
   */
  list: async (): Promise<ModelsListResponse> => apiClient.get('/v1/models'),

  /**
   * Get specific model info
   */
  get: async (modelId: string): Promise<ModelInfo> =>
    apiClient.get(`/v1/models/${modelId}`),

  /**
   * Download a model
   */
  download: async (modelId: string): Promise<{ status: string; message: string }> =>
    apiClient.post(`/v1/models/${modelId}/download`, {}),

  /**
   * Get download progress
   */
  downloadProgress: async (modelId: string): Promise<{ progress: number; status: string }> =>
    apiClient.get(`/v1/models/${modelId}/download/progress`),

  /**
   * Delete a model
   */
  delete: async (modelId: string): Promise<{ status: string; message: string }> =>
    apiClient.delete(`/v1/models/${modelId}`),
};

/**
 * Server health and status endpoints
 */
export const server = {
  /**
   * Check server health
   */
  health: async (): Promise<HealthResponse> => apiClient.health(),

  /**
   * Check server readiness
   */
  ready: async (): Promise<ReadinessResponse> => apiClient.ready(),

  /**
   * Get server metrics
   */
  metrics: async (): Promise<MetricsResponse> => apiClient.metrics<MetricsResponse>(),
};

/**
 * Configuration endpoints
 */
export const config = {
  /**
   * Get current configuration
   */
  get: async (): Promise<Record<string, unknown>> => apiClient.get('/config'),

  /**
   * Update configuration
   */
  update: async (cfg: Record<string, unknown>): Promise<{ status: string }> =>
    apiClient.put('/config', cfg),

  /**
   * Get configuration schema
   */
  schema: async (): Promise<Record<string, unknown>> => apiClient.get('/config/schema'),
};

/**
 * Inference endpoints
 */
export const inference = {
  /**
   * Get inference capabilities
   */
  capabilities: async (): Promise<Record<string, unknown>> =>
    apiClient.get('/v1/inference/capabilities'),

  /**
   * Get inference limits
   */
  limits: async (): Promise<Record<string, unknown>> =>
    apiClient.get('/v1/inference/limits'),
};
