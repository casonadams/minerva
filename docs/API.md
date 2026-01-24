# Minerva API Documentation

## Overview

Minerva provides a REST API compatible with OpenAI's Chat Completion interface. All endpoints require proper input validation and respect rate limiting rules.

## Base URL

```
http://localhost:3000/v1
```

## Authentication

Currently, Minerva uses a client ID header for rate limiting identification:

```
x-client-id: your-client-id
```

## Endpoints

### 1. Chat Completions

Create a text completion for the provided prompt and messages.

**Endpoint:** `POST /v1/chat/completions`

**Request Body:**

```json
{
  "model": "llama-2-7b",
  "messages": [
    {
      "role": "user",
      "content": "Hello, how are you?"
    }
  ],
  "temperature": 0.7,
  "top_p": 0.9,
  "max_tokens": 100,
  "stream": false
}
```

**Parameters:**

| Parameter | Type | Required | Description | Constraints |
|-----------|------|----------|-------------|-------------|
| model | string | Yes | Model identifier | Max 255 chars, alphanumeric + `-_/.` |
| messages | array | Yes | Array of message objects | At least 1 message |
| temperature | float | No | Sampling temperature | Range: [0, 2] |
| top_p | float | No | Nucleus sampling parameter | Range: (0, 1] |
| max_tokens | integer | No | Max completion tokens | Positive integer |
| stream | boolean | No | Stream response | Default: false |

**Message Object:**

```json
{
  "role": "user|assistant|system",
  "content": "Message text"
}
```

**Response (Non-Streaming):**

```json
{
  "id": "chatcmpl-8a3b9c2e",
  "object": "chat.completion",
  "created": 1704067200,
  "model": "llama-2-7b",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Response text here"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 25,
    "total_tokens": 35
  }
}
```

**Response (Streaming):**

Server-Sent Events format with chunked responses:

```
data: {"id":"chatcmpl-...","object":"chat.completion.chunk","created":1704067200,"model":"llama-2-7b","choices":[{"index":0,"delta":{"role":"assistant","content":" Hello"},"finish_reason":null}]}

data: {"id":"chatcmpl-...","object":"chat.completion.chunk","created":1704067200,"model":"llama-2-7b","choices":[{"index":0,"delta":{"content":" there"},"finish_reason":null}]}

data: [DONE]
```

**Status Codes:**

| Code | Description |
|------|-------------|
| 200 | Successful completion |
| 206 | Partial content (streaming) |
| 400 | Bad request (validation error, rate limit) |
| 404 | Model not found |
| 500 | Server error |

### 2. List Models

Get list of available models.

**Endpoint:** `GET /v1/models`

**Response:**

```json
{
  "object": "list",
  "data": [
    {
      "id": "llama-2-7b",
      "object": "model",
      "created": 1704067200,
      "owned_by": "local",
      "context_window": 4096,
      "max_output_tokens": 2048
    }
  ]
}
```

### 3. Health Check

Check server health status.

**Endpoint:** `GET /health`

**Response:**

```json
{
  "status": "healthy",
  "timestamp": "2024-01-03T12:00:00Z"
}
```

### 4. Readiness Check

Check if server is ready to accept requests.

**Endpoint:** `GET /ready`

**Response:**

```json
{
  "ready": true,
  "timestamp": "2024-01-03T12:00:00Z"
}
```

### 5. Metrics

Get server metrics and performance data.

**Endpoint:** `GET /metrics`

**Response:**

```json
{
  "timestamp": "2024-01-03T12:00:00Z",
  "uptime_seconds": 3600,
  "requests": {
    "total": 1000,
    "successful": 980,
    "failed": 20,
    "rps": 0.28
  },
  "response_times": {
    "avg_ms": 150,
    "min_ms": 10,
    "max_ms": 500,
    "p50_ms": 100,
    "p95_ms": 400,
    "p99_ms": 480
  },
  "errors": {
    "total": 20,
    "rate_percent": 2.0,
    "top_error": "rate_limit_exceeded"
  },
  "cache": {
    "hits": 500,
    "misses": 500,
    "hit_rate_percent": 50.0
  }
}
```

## Rate Limiting

Minerva implements token bucket rate limiting per client.

**Default Limits:**

- **Burst Capacity:** 100 tokens
- **Refill Rate:** 10 tokens/second
- **Per Client:** Rate limits are tracked per `x-client-id` header

**Rate Limit Headers:**

```
x-ratelimit-limit: 100
x-ratelimit-remaining: 45
x-ratelimit-reset: 1704067260
```

**Rate Limit Exceeded Response (HTTP 429):**

```json
{
  "error": {
    "message": "Rate limit exceeded. Retry after 5 seconds",
    "type": "rate_limit_exceeded",
    "code": "rate_limit_exceeded"
  }
}
```

**Retry-After Header:**

```
retry-after: 5
```

## Input Validation

All requests are validated according to these rules:

### Prompt Validation
- **Requirement:** Non-empty
- **Max Length:** 2000 characters
- **Error:** 400 Bad Request

### Model ID Validation
- **Requirement:** Non-empty, valid format
- **Allowed Characters:** alphanumeric, `-`, `_`, `/`, `.`
- **Max Length:** 255 characters
- **Error:** 400 Bad Request

### Temperature Validation
- **Range:** [0, 2]
- **Type:** Float
- **Default:** 0.7
- **Error:** 400 Bad Request

### Top-P Validation
- **Range:** (0, 1]
- **Type:** Float
- **Default:** 1.0
- **Error:** 400 Bad Request

### Token Count Validation
- **Range:** [1, context_window)
- **Type:** Integer
- **Error:** 400 Bad Request

### Message Role Validation
- **Allowed Values:** `user`, `assistant`, `system`
- **Error:** 400 Bad Request

## Error Responses

All errors follow this format:

```json
{
  "error": {
    "message": "Human-readable error message",
    "type": "error_code",
    "code": "error_code",
    "param": null
  }
}
```

**Common Error Codes:**

| Code | HTTP Status | Description |
|------|-------------|-------------|
| invalid_request_error | 400 | Invalid request parameters |
| model_not_found | 404 | Requested model not found |
| validation_error | 400 | Input validation failed |
| rate_limit_exceeded | 429 | Rate limit exceeded |
| server_error | 500 | Internal server error |

## Examples

### Basic Chat Completion

```bash
curl -X POST http://localhost:3000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "x-client-id: my-app" \
  -d '{
    "model": "llama-2-7b",
    "messages": [
      {
        "role": "user",
        "content": "What is 2+2?"
      }
    ]
  }'
```

### Streaming Response

```bash
curl -X POST http://localhost:3000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "x-client-id: my-app" \
  -d '{
    "model": "llama-2-7b",
    "messages": [
      {
        "role": "user",
        "content": "Tell me a story"
      }
    ],
    "stream": true
  }'
```

### With Parameters

```bash
curl -X POST http://localhost:3000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "x-client-id: my-app" \
  -d '{
    "model": "llama-2-7b",
    "messages": [
      {
        "role": "system",
        "content": "You are a helpful assistant"
      },
      {
        "role": "user",
        "content": "Explain quantum computing"
      }
    ],
    "temperature": 0.8,
    "top_p": 0.95,
    "max_tokens": 500
  }'
```

## Best Practices

1. **Use Unique Client IDs:** Different applications/services should use distinct `x-client-id` values for proper rate limiting
2. **Handle Rate Limiting:** Implement exponential backoff when receiving 429 responses
3. **Streaming for Long Responses:** Use streaming for responses that may take time
4. **Validate Inputs Client-Side:** Pre-validate before sending to reduce server load
5. **Monitor Metrics:** Regularly check `/metrics` endpoint for performance insights
6. **Health Checks:** Use `/health` and `/ready` endpoints for deployment health checks

## Changelog

### v1.0.0 (Phase 10)
- Initial REST API release
- Rate limiting with token bucket algorithm
- Input validation for all parameters
- Server health and metrics endpoints
- Streaming support (SSE)
- Mock inference engine
