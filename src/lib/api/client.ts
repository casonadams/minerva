// HTTP API client for Minerva server
// Handles retries, timeouts, and error handling

export interface ApiError {
  message: string;
  code: string;
  status: number;
}

export interface ApiClientOptions {
  baseUrl?: string;
  timeout?: number;
  maxRetries?: number;
  retryDelay?: number;
}

/**
 * HTTP API client for communicating with Minerva server
 * Supports automatic retries with exponential backoff
 */
export class ApiClient {
  private baseUrl: string;
  private timeout: number;
  private maxRetries: number;
  private retryDelay: number;

  constructor(options: ApiClientOptions = {}) {
    this.baseUrl = options.baseUrl || this.detectBaseUrl();
    this.timeout = options.timeout || 30000;
    this.maxRetries = options.maxRetries || 3;
    this.retryDelay = options.retryDelay || 1000;
  }

  /**
   * Detect API base URL (dev vs production)
   */
  private detectBaseUrl(): string {
    if (typeof window === 'undefined') {
      return 'http://localhost:3000';
    }

    const isDev = window.location.hostname === 'localhost' || 
                  window.location.hostname === '127.0.0.1';
    
    if (isDev) {
      const port = import.meta.env.VITE_API_PORT || '3000';
      return `http://localhost:${port}`;
    }

    return '/api';
  }

  /**
   * Make GET request
   */
  async get<T>(path: string, headers?: Record<string, string>): Promise<T> {
    return this.request<T>('GET', path, undefined, headers);
  }

  /**
   * Make POST request
   */
  async post<T>(
    path: string,
    data?: unknown,
    headers?: Record<string, string>,
  ): Promise<T> {
    return this.request<T>('POST', path, data, headers);
  }

  /**
   * Make PUT request
   */
  async put<T>(
    path: string,
    data?: unknown,
    headers?: Record<string, string>,
  ): Promise<T> {
    return this.request<T>('PUT', path, data, headers);
  }

  /**
   * Make DELETE request
   */
  async delete<T>(path: string, headers?: Record<string, string>): Promise<T> {
    return this.request<T>('DELETE', path, undefined, headers);
  }

  /**
   * Core request method with retry logic
   */
  private async request<T>(
    method: string,
    path: string,
    data?: unknown,
    headers?: Record<string, string>,
  ): Promise<T> {
    const url = `${this.baseUrl}${path}`;
    
    let lastError: Error | undefined;
    
    for (let attempt = 0; attempt <= this.maxRetries; attempt++) {
      try {
        const response = await this.fetchWithTimeout(
          url,
          {
            method,
            headers: {
              'Content-Type': 'application/json',
              'x-client-id': this.getClientId(),
              ...headers,
            },
            body: data ? JSON.stringify(data) : undefined,
          },
          this.timeout,
        );

        if (!response.ok) {
          const errorData = await response.json().catch(() => ({}));
          const error: ApiError = {
            message: errorData.error?.message || response.statusText,
            code: errorData.error?.code || 'UNKNOWN',
            status: response.status,
          };
          throw new Error(JSON.stringify(error));
        }

        return await response.json() as T;
      } catch (error) {
        lastError = error instanceof Error ? error : new Error(String(error));
        
        // Don't retry on client errors (4xx)
        if (error instanceof Error && error.message.includes('status')) {
          throw error;
        }
        
        // Exponential backoff for retries
        if (attempt < this.maxRetries) {
          const delay = this.retryDelay * Math.pow(2, attempt);
          await new Promise(resolve => setTimeout(resolve, delay));
        }
      }
    }
    
    throw lastError || new Error('Request failed');
  }

  /**
   * Fetch with timeout support
   */
  private fetchWithTimeout(
    url: string,
    options: RequestInit,
    timeout: number,
  ): Promise<Response> {
    return Promise.race([
      fetch(url, options),
      new Promise<Response>((_, reject) =>
        setTimeout(() => reject(new Error('Request timeout')), timeout),
      ),
    ]);
  }

  /**
   * Get unique client ID for rate limiting tracking
   */
  private getClientId(): string {
    let clientId = localStorage.getItem('minerva_client_id');
    
    if (!clientId) {
      clientId = `client_${Math.random().toString(36).substr(2, 9)}`;
      localStorage.setItem('minerva_client_id', clientId);
    }
    
    return clientId;
  }

  /**
   * Check server health
   */
  async health(): Promise<{ status: string; timestamp: string }> {
    return this.get('/health');
  }

  /**
   * Check server readiness
   */
  async ready(): Promise<{ ready: boolean; timestamp: string }> {
    return this.get('/ready');
  }

  /**
   * Get server metrics
   */
  async metrics<T = unknown>(): Promise<T> {
    return this.get<T>('/metrics');
  }
}

export const apiClient = new ApiClient();
