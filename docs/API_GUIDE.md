# Minerva API Guide

Complete reference for Minerva's OpenAI-compatible REST API (v0.2.0+)

## Base URL

```
http://localhost:3000
```

## Authentication

Currently, Minerva does not require authentication. For production deployments, use a reverse proxy (nginx, Envoy) to add authentication.

## Rate Limiting

Default: 1000 requests per minute (configurable)

Rate limit headers:
- `X-RateLimit-Limit`: Maximum requests per minute
- `X-RateLimit-Remaining`: Requests remaining
- `X-RateLimit-Reset`: Unix timestamp when limit resets

## Response Format

All responses include metadata:

```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1705980000,
  "model": "gpt-4",
  "choices": [...],
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 20,
    "total_tokens": 30
  },
  "request_id": "req-abc123",
  "timestamp": 1705980000
}
```

## Endpoints

### Models

#### List Models
```http
GET /v1/models
```

**Response:**
```json
{
  "object": "list",
  "data": [
    {
      "id": "gpt-4",
      "object": "model",
      "created": 1705980000,
      "owned_by": "minerva"
    },
    {
      "id": "mistral-7b",
      "object": "model",
      "created": 1705980000,
      "owned_by": "minerva"
    }
  ]
}
```

#### Get Model Details
```http
GET /v1/models/{model_id}
```

**Parameters:**
- `model_id` (path): Model identifier (1-256 chars)

**Response:**
```json
{
  "id": "gpt-4",
  "object": "model",
  "created": 1705980000,
  "owned_by": "minerva"
}
```

**Errors:**
- `404`: Model not found

### Chat Completions

#### Create Chat Completion
```http
POST /v1/chat/completions
Content-Type: application/json
```

**Request:**
```json
{
  "model": "gpt-4",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello, how are you?"}
  ],
  "temperature": 0.7,
  "max_tokens": 100,
  "top_p": 1.0,
  "stream": false
}
```

**Parameters:**
- `model` (required): Model identifier
- `messages` (required): Array of message objects
  - `role` (required): "system", "user", or "assistant"
  - `content` (required): Message text
- `temperature` (optional): 0-2, default 0.7
- `max_tokens` (optional): 1-4096, default 2048
- `top_p` (optional): 0-1, default 1.0
- `stream` (optional): Enable streaming, default false

**Response:**
```json
{
  "id": "chatcmpl-123",
  "object": "chat.completion",
  "created": 1705980000,
  "model": "gpt-4",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Hello! I'm doing well, thank you for asking."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 25,
    "completion_tokens": 12,
    "total_tokens": 37
  }
}
```

**Errors:**
- `400`: Invalid request (bad parameters, missing fields)
- `404`: Model not found
- `429`: Rate limit exceeded
- `500`: Server error

#### Streaming

Enable streaming with `"stream": true`:

```bash
curl -X POST http://localhost:3000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4",
    "messages": [{"role": "user", "content": "Hello"}],
    "stream": true
  }'
```

**Response (Server-Sent Events):**
```
data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1705980000,"model":"gpt-4","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}

data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1705980000,"model":"gpt-4","choices":[{"index":0,"delta":{"content":" there"},"finish_reason":null}]}

data: [DONE]
```

### Text Completions

#### Create Completion
```http
POST /v1/completions
Content-Type: application/json
```

**Request:**
```json
{
  "model": "gpt-4",
  "prompt": "Write a poem about",
  "temperature": 0.7,
  "max_tokens": 100,
  "top_p": 1.0,
  "stream": false
}
```

**Parameters:**
- `model` (required): Model identifier
- `prompt` (required): Input text to complete
- `temperature` (optional): 0-2, default 0.7
- `max_tokens` (optional): 1-4096, default 2048
- `top_p` (optional): 0-1, default 1.0
- `stream` (optional): Enable streaming, default false

**Response:**
```json
{
  "id": "cmpl-123",
  "object": "text_completion",
  "created": 1705980000,
  "model": "gpt-4",
  "choices": [
    {
      "index": 0,
      "text": " nature\n\nNature's beauty knows no bound,\nWith colors vibrant all around.",
      "finish_reason": "length"
    }
  ],
  "usage": {
    "prompt_tokens": 4,
    "completion_tokens": 20,
    "total_tokens": 24
  }
}
```

### Embeddings

#### Create Embeddings
```http
POST /v1/embeddings
Content-Type: application/json
```

**Request (single string):**
```json
{
  "model": "gpt-4",
  "input": "Hello, how are you?"
}
```

**Request (multiple strings):**
```json
{
  "model": "gpt-4",
  "input": [
    "Hello, how are you?",
    "I'm doing well, thank you!"
  ]
}
```

**Parameters:**
- `model` (required): Model identifier
- `input` (required): String or array of strings to embed

**Response:**
```json
{
  "object": "list",
  "model": "gpt-4",
  "data": [
    {
      "object": "embedding",
      "embedding": [0.1, 0.2, 0.3, ...],
      "index": 0
    }
  ],
  "usage": {
    "prompt_tokens": 9,
    "total_tokens": 9
  }
}
```

### System

#### Health Check
```http
GET /v1/health
```

**Response:**
```json
{
  "status": "healthy",
  "uptime_seconds": 3600,
  "components": {
    "inference": "healthy",
    "api": "healthy"
  }
}
```

**Status codes:**
- `200`: Healthy
- `503`: Unhealthy (degraded or down)

#### Get Configuration
```http
GET /v1/config
```

**Response:**
```json
{
  "server": {
    "host": "0.0.0.0",
    "port": 3000,
    "workers": 4
  },
  "api": {
    "version": "v1",
    "rate_limit": {
      "requests_per_minute": 1000
    }
  },
  "inference": {
    "timeout_seconds": 300,
    "max_batch_size": 32
  },
  "streaming": {
    "enabled": true,
    "chunk_size": 1024
  }
}
```

## Error Responses

All errors follow OpenAI format:

```json
{
  "error": {
    "message": "Model 'unknown-model' not found",
    "type": "model_not_found",
    "code": 404,
    "param": "model"
  }
}
```

### Error Types

| Type | HTTP Status | Description |
|------|-------------|-------------|
| `invalid_request` | 400 | Invalid parameters or missing required fields |
| `model_not_found` | 404 | Specified model does not exist |
| `rate_limit_exceeded` | 429 | Too many requests |
| `server_error` | 500 | Internal server error |

## Examples

### Using cURL

```bash
# List models
curl http://localhost:3000/v1/models

# Chat completion
curl -X POST http://localhost:3000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4",
    "messages": [{"role": "user", "content": "Hello"}],
    "temperature": 0.7,
    "max_tokens": 50
  }'

# Streaming
curl -N -X POST http://localhost:3000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4",
    "messages": [{"role": "user", "content": "Hello"}],
    "stream": true
  }'
```

### Using Python

```python
import requests
import json

BASE_URL = "http://localhost:3000"

# Chat completion
response = requests.post(
    f"{BASE_URL}/v1/chat/completions",
    json={
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "Hello"}],
        "temperature": 0.7,
        "max_tokens": 100
    }
)

result = response.json()
print(result["choices"][0]["message"]["content"])

# Streaming
response = requests.post(
    f"{BASE_URL}/v1/chat/completions",
    json={
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "Hello"}],
        "stream": True
    },
    stream=True
)

for line in response.iter_lines():
    if line.startswith(b"data: "):
        data = json.loads(line[6:])
        if data.get("choices"):
            delta = data["choices"][0].get("delta", {})
            content = delta.get("content", "")
            print(content, end="", flush=True)
```

### Using TypeScript/JavaScript

```typescript
import Minerva from './lib/api/client';

const client = new Minerva({
  baseURL: 'http://localhost:3000',
  retryConfig: {
    maxRetries: 3,
    backoffMs: 100
  }
});

// Chat completion
const response = await client.chat.create({
  model: 'gpt-4',
  messages: [{ role: 'user', content: 'Hello' }],
  temperature: 0.7,
  max_tokens: 100
});

console.log(response.choices[0].message.content);

// Streaming
const stream = await client.chat.stream({
  model: 'gpt-4',
  messages: [{ role: 'user', content: 'Hello' }]
});

for await (const chunk of stream) {
  const content = chunk.choices[0]?.delta?.content;
  if (content) {
    process.stdout.write(content);
  }
}
```

## Best Practices

### 1. Connection Pooling
Reuse connections when making multiple requests:

```python
import requests

session = requests.Session()
for prompt in prompts:
    response = session.post(
        "http://localhost:3000/v1/chat/completions",
        json={...}
    )
```

### 2. Timeout Configuration
Always set timeouts:

```bash
curl --max-time 30 \
  -X POST http://localhost:3000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{...}'
```

### 3. Streaming for Long Responses
Use streaming to get results faster for long completions:

```json
{
  "model": "gpt-4",
  "messages": [...],
  "stream": true
}
```

### 4. Batch Requests
For multiple requests, submit them sequentially with proper error handling:

```python
responses = []
for prompt in prompts:
    try:
        response = requests.post(
            "http://localhost:3000/v1/chat/completions",
            json={...},
            timeout=30
        )
        responses.append(response.json())
    except requests.RequestException as e:
        print(f"Error: {e}")
```

### 5. Monitor Rate Limits
Check rate limit headers in responses:

```python
response = requests.post(...)
remaining = response.headers.get('X-RateLimit-Remaining')
if int(remaining) < 10:
    print("Approaching rate limit!")
```

## Compatibility

Minerva is compatible with OpenAI SDKs:

```bash
# Python
pip install openai

# Then configure endpoint
export OPENAI_API_BASE=http://localhost:3000/v1
export OPENAI_API_KEY=dummy
```

## Support

- **Issues**: GitHub Issues
- **Documentation**: See `/docs` directory
- **OpenAPI**: Available at `/docs/openapi.yaml`
