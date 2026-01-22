# Tauri LLM Server - OpenAI-Compatible Desktop App

## Project Overview

Build a Tauri desktop application that serves local LLMs via OpenAI-compatible APIs. This allows tools like OpenCode, LangChain, and other LLM clients to connect to locally-running models without modification.

### Key Goals
- Serve GGUF-format LLM models locally
- Expose OpenAI-compatible REST API endpoints
- Support `/v1/chat/completions` (primary endpoint)
- Support `/v1/models` for model discovery
- Enable Mac Metal GPU acceleration
- Run on `localhost:11434` (or configurable port)

---

## Architecture Overview

```
┌─────────────────────────────────────────────┐
│         Tauri Desktop Application           │
├─────────────────────────────────────────────┤
│  Frontend (Svelte + TypeScript)             │
│  - Model manager UI                         │
│  - Server status/logs                       │
│  - Settings panel                           │
├─────────────────────────────────────────────┤
│  Tauri Bridge (IPC)                         │
├─────────────────────────────────────────────┤
│  Backend (Rust)                             │
│  ┌───────────────────────────────────────┐  │
│  │ HTTP Server (Axum)                    │  │
│  │ - GET  /v1/models                     │  │
│  │ - POST /v1/chat/completions           │  │
│  │ - Health checks                       │  │
│  └───────────────────────────────────────┘  │
│  ┌───────────────────────────────────────┐  │
│  │ LLM Inference Engine                  │  │
│  │ - GGUF Model Loader                   │  │
│  │ - llama.cpp Integration               │  │
│  │ - Metal GPU Acceleration              │  │
│  │ - Context Management                  │  │
│  └───────────────────────────────────────┘  │
│  ┌───────────────────────────────────────┐  │
│  │ Model Management                      │  │
│  │ - Local model registry                │  │
│  │ - Model loading/unloading             │  │
│  │ - Cache management                    │  │
│  └───────────────────────────────────────┘  │
└─────────────────────────────────────────────┘
```

---

## Core Components

### 1. HTTP Server (Axum)
**File:** `src-tauri/src/server.rs`

- Runs on background thread
- Handles REST API requests
- Returns JSON responses matching OpenAI format
- Implements CORS for external tool access

**Key Endpoints:**
```
GET /v1/models
POST /v1/chat/completions
GET /health
```

### 2. Model Manager
**File:** `src-tauri/src/models.rs`

- Tracks available GGUF models
- Loads/unloads models from disk
- Maintains model metadata
- Manages model cache

### 3. Inference Engine
**File:** `src-tauri/src/inference.rs`

- Integrates llama.cpp via bindings
- Manages LLM context
- Handles token generation
- Supports streaming responses
- Metal GPU acceleration on macOS

### 4. Tauri Commands
**File:** `src-tauri/src/commands.rs`

- Bridge between frontend and backend
- Model discovery commands
- Server control (start/stop)
- Settings management

### 5. Frontend UI (Svelte)
**File:** `src/routes/+page.svelte`

- Model management interface
- Server status display
- Settings configuration
- Request logging

---

## Technology Stack

### Backend (Rust)
- **tauri** (v2) - Desktop framework
- **axum** - HTTP server framework
- **tokio** - Async runtime
- **serde** / **serde_json** - JSON serialization
- **llama-cpp-sys-3** or **rs-llama-cpp** - llama.cpp bindings
- **gguf-rs-lib** - GGUF file parsing
- **metal** - Mac GPU acceleration
- **uuid** - For request IDs

### Frontend (Svelte/TypeScript)
- **Svelte 5** - UI framework
- **SvelteKit** - Application framework
- **Tailwind CSS** - Styling
- **TypeScript** - Type safety

---

## API Response Format

### GET /v1/models
```json
{
  "object": "list",
  "data": [
    {
      "id": "model-name",
      "object": "model",
      "created": 1704067200,
      "owned_by": "local",
      "context_window": 4096,
      "max_output_tokens": 2048
    }
  ]
}
```

### POST /v1/chat/completions
**Request:**
```json
{
  "model": "model-name",
  "messages": [
    {"role": "user", "content": "Hello"}
  ],
  "temperature": 0.7,
  "max_tokens": 1024,
  "stream": false
}
```

**Response (non-streaming):**
```json
{
  "id": "chatcmpl-123abc",
  "object": "chat.completion",
  "created": 1704067200,
  "model": "model-name",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Hello! How can I help?"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 10,
    "total_tokens": 20
  }
}
```

**Streaming Response:**
Server-Sent Events (SSE) format with delta updates
```
data: {"choices":[{"delta":{"content":"Hello"}}]}
...
data: [DONE]
```

---

## Implementation Phases

### Phase 1: Foundation (Week 1)
- [ ] Set up Tauri backend structure
- [ ] Create HTTP server with Axum
- [ ] Implement `/v1/models` endpoint (mock data)
- [ ] Implement `/v1/chat/completions` endpoint (mock responses)
- [ ] Create basic Tauri commands

### Phase 2: Model Loading (Week 2)
- [ ] Add GGUF file parsing
- [ ] Implement model discovery from filesystem
- [ ] Create model registry
- [ ] Add model metadata extraction
- [ ] Implement model loading/unloading

### Phase 3: LLM Inference (Week 3)
- [ ] Integrate llama.cpp bindings
- [ ] Implement token generation
- [ ] Add context management
- [ ] Implement streaming responses
- [ ] Add error handling

### Phase 4: GPU Acceleration (Week 4)
- [ ] Add Metal framework support
- [ ] Implement device detection
- [ ] Configure GPU-accelerated inference
- [ ] Optimize memory management
- [ ] Performance benchmarking

### Phase 5: UI & Polish (Week 5)
- [ ] Create model management UI
- [ ] Add server status dashboard
- [ ] Implement settings panel
- [ ] Add request logging
- [ ] Create documentation

### Phase 6: Testing (Week 6)
- [ ] Test with OpenCode
- [ ] Test with curl commands
- [ ] Performance testing
- [ ] Edge case handling
- [ ] Bug fixes and optimization

---

## Key Implementation Details

### Streaming Response Implementation
Use Axum's `BodyStream` or SSE helper to stream tokens as they're generated from llama.cpp

### Error Handling
Match OpenAI error format:
```json
{
  "error": {
    "message": "error description",
    "type": "error_type",
    "param": null,
    "code": "error_code"
  }
}
```

### Model Configuration
Store model config in JSON:
```json
{
  "models": [
    {
      "id": "model-name",
      "path": "/path/to/model.gguf",
      "context_window": 4096,
      "max_output_tokens": 2048
    }
  ]
}
```

### CORS Configuration
Allow external tools to access the API:
```rust
CorsLayer::permissive()
```

---

## Configuration

### Environment Variables
- `LLM_SERVER_PORT` - Server port (default: 11434)
- `LLM_MODELS_DIR` - Models directory path
- `LLM_USE_GPU` - Enable GPU acceleration (default: true on macOS)

### Config File
`~/.lmstudio/config.json` or platform-specific config dir

---

## Testing Strategy

### Unit Tests
- GGUF parsing
- Model registry
- Request/response formatting

### Integration Tests
- Full request/response cycle
- Model loading
- Inference execution

### External Testing
```bash
# List models
curl http://localhost:11434/v1/models

# Chat completion
curl http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "my-model",
    "messages": [{"role": "user", "content": "Hi"}]
  }'
```

---

## Performance Considerations

1. **Lazy Loading**: Models loaded on-demand, unloaded when unused
2. **GPU Memory**: Metal acceleration for faster inference on Apple Silicon
3. **Streaming**: Stream tokens as generated to reduce latency perception
4. **Caching**: Cache model metadata and loaded models in memory
5. **Threading**: Use tokio for async request handling

---

## Security Considerations

1. **localhost-only**: Server runs only on localhost (not exposed to network by default)
2. **No authentication**: Optional API key for external access
3. **Input validation**: Validate all incoming requests
4. **Resource limits**: Maximum tokens, context size limits
5. **Model validation**: Verify GGUF files before loading

---

## Future Enhancements

- [ ] Support for multiple concurrent models
- [ ] Model quantization options
- [ ] Custom system prompts
- [ ] Function calling support
- [ ] Embeddings endpoint
- [ ] Vision capabilities
- [ ] Multi-turn conversation history
- [ ] Rate limiting
- [ ] Metrics/monitoring
