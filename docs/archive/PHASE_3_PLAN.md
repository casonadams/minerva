# Phase 3: LLM Inference Engine

## Overview

Phase 3 focuses on integrating the LLM inference engine, enabling the application to process natural language requests and generate responses using the local GGUF models discovered in Phase 2.

**Duration:** 2-3 weeks  
**Priority:** CRITICAL  
**Depends On:** Phase 2 ✅ (Complete)  
**Blocks:** Phase 4 (Streaming & Advanced Features)

## Goals

1. Integrate llama.cpp for GGUF model inference
2. Replace mock responses with real LLM completions
3. Implement token generation & response streaming
4. Handle context window management
5. Implement proper error handling for inference
6. Add inference performance monitoring
7. Support configurable generation parameters
8. Enable concurrent model loading

## Architecture

```
┌─────────────────────────────────────┐
│  Frontend (Svelte)                  │
│  Chat Interface                     │
└──────────────┬──────────────────────┘
               │ HTTP POST /v1/chat/completions
               ▼
┌─────────────────────────────────────┐
│  HTTP Server (Axum)                 │
│  Request parsing & validation       │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  LLM Inference Engine               │
│  - Model loading                    │
│  - Token generation                 │
│  - Context management               │
│  - Parameter handling               │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  llama.cpp Bindings                 │
│  - Model inference                  │
│  - Memory management                │
│  - GPU acceleration                 │
└─────────────────────────────────────┘
```

## Implementation Plan

### Step 1: Add Inference Dependencies (1 hour)

**File:** `src-tauri/Cargo.toml`

Add dependencies for LLM inference:
```toml
llama-cpp-rs = { version = "0.1", features = ["metal"] }  # llama.cpp bindings
tokio = { version = "1", features = ["full"] }             # async runtime
serde = { version = "1", features = ["derive"] }           # serialization
```

**Tasks:**
- [ ] Add llama.cpp Rust bindings
- [ ] Verify Metal GPU support enabled
- [ ] Run `cargo check` to verify compatibility

### Step 2: Create Inference Module (2 hours)

**New File:** `src-tauri/src/inference/mod.rs`

```rust
pub struct InferenceEngine {
    model_path: PathBuf,
    context: LlamaContext,
    config: GenerationConfig,
}

pub struct GenerationConfig {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: u32,
    pub repeat_penalty: f32,
    pub max_tokens: usize,
}

impl InferenceEngine {
    pub fn new(model_path: PathBuf) -> MinervaResult<Self> { ... }
    pub async fn generate(&mut self, prompt: &str) -> MinervaResult<String> { ... }
    pub async fn generate_stream(&mut self, prompt: &str) 
        -> MinervaResult<impl Stream<Item = String>> { ... }
}
```

**Tasks:**
- [ ] Initialize llama.cpp context
- [ ] Implement token generation
- [ ] Handle context window limits
- [ ] Add error handling for inference failures

### Step 3: Model Context Manager (1.5 hours)

**New File:** `src-tauri/src/inference/context_manager.rs`

```rust
pub struct ContextManager {
    models: HashMap<String, ModelContext>,
}

pub struct ModelContext {
    engine: InferenceEngine,
    last_used: Instant,
    usage_stats: UsageStats,
}

impl ContextManager {
    pub fn load_model(&mut self, id: &str, path: PathBuf) 
        -> MinervaResult<()> { ... }
    pub fn unload_model(&mut self, id: &str) { ... }
    pub fn get_model(&mut self, id: &str) 
        -> MinervaResult<&mut InferenceEngine> { ... }
}
```

**Tasks:**
- [ ] Track loaded model contexts
- [ ] Implement lazy loading
- [ ] Add model unloading with cleanup
- [ ] Track memory usage per model

### Step 4: Update Chat Completions Endpoint (1.5 hours)

**File:** `src-tauri/src/server.rs`

Replace mock implementation with real inference:

```rust
async fn chat_completions(
    State(state): State<ServerState>,
    Json(req): Json<ChatCompletionRequest>,
) -> MinervaResult<Response> {
    // Load model if needed
    let model = state.inference_engine.get_model(&req.model)?;
    
    // Build prompt from messages
    let prompt = build_prompt(&req.messages);
    
    // Generate response
    if req.stream.unwrap_or(false) {
        Ok(create_streaming_response(model, prompt).into_response())
    } else {
        let response = model.generate(&prompt).await?;
        Ok(create_completion_response(req, response).into_response())
    }
}
```

**Tasks:**
- [ ] Remove mock response
- [ ] Integrate with InferenceEngine
- [ ] Handle streaming responses
- [ ] Update error messages with real errors

### Step 5: Implement Streaming Response (2 hours)

**File:** `src-tauri/src/server.rs`

```rust
async fn create_streaming_response(
    engine: &mut InferenceEngine,
    prompt: String,
) -> impl IntoResponse {
    let stream = engine.generate_stream(&prompt).await.unwrap();
    
    Sse::new(stream.map(|token| {
        let data = json!({
            "id": format!("chatcmpl-{}", Uuid::new_v4()),
            "object": "chat.completion.chunk",
            "created": chrono::Utc::now().timestamp(),
            "model": "current_model",
            "choices": [{
                "index": 0,
                "delta": { "content": token },
                "finish_reason": null
            }]
        });
        Event::default().data(data.to_string())
    }))
}
```

**Tasks:**
- [ ] Implement server-sent events (SSE)
- [ ] Stream tokens as they're generated
- [ ] Add proper chunking format
- [ ] Handle stream errors gracefully

### Step 6: Generation Parameter Handling (1.5 hours)

**File:** `src-tauri/src/inference/mod.rs`

```rust
impl InferenceEngine {
    pub fn set_temperature(&mut self, temp: f32) { ... }
    pub fn set_top_p(&mut self, top_p: f32) { ... }
    pub fn set_top_k(&mut self, top_k: u32) { ... }
    pub fn set_repeat_penalty(&mut self, penalty: f32) { ... }
    pub fn set_max_tokens(&mut self, tokens: usize) { ... }
}
```

**Tasks:**
- [ ] Map request parameters to llama.cpp settings
- [ ] Validate parameter ranges
- [ ] Apply defaults for missing parameters
- [ ] Document parameter meanings

### Step 7: Error Handling & Logging (1.5 hours)

**Files:** `src-tauri/src/error.rs`, `src-tauri/src/inference/mod.rs`

```rust
pub enum MinervaError {
    // ... existing errors ...
    InferenceError(String),
    ModelNotLoaded(String),
    ContextLimitExceeded { max: usize, required: usize },
    GenerationTimeout,
    OutOfMemory,
}

impl From<LlamaCppError> for MinervaError {
    fn from(err: LlamaCppError) -> Self {
        MinervaError::InferenceError(err.to_string())
    }
}
```

**Tasks:**
- [ ] Add inference-specific error types
- [ ] Implement proper error conversion
- [ ] Add structured logging with tracing
- [ ] Log inference metrics

### Step 8: Performance Monitoring (1 hour)

**New File:** `src-tauri/src/inference/metrics.rs`

```rust
pub struct InferenceMetrics {
    pub tokens_generated: u64,
    pub total_time_ms: u128,
    pub tokens_per_second: f64,
    pub model_load_time_ms: u128,
}

impl InferenceMetrics {
    pub fn calculate_tps(&self) -> f64 {
        self.tokens_generated as f64 / (self.total_time_ms as f64 / 1000.0)
    }
}
```

**Tasks:**
- [ ] Track token generation speed
- [ ] Measure model loading time
- [ ] Monitor memory usage
- [ ] Log performance metrics

### Step 9: Add Inference Tests (2 hours)

**New File:** `src-tauri/src/inference_tests.rs`

```rust
#[tokio::test]
async fn test_generate_simple_response() { ... }

#[tokio::test]
async fn test_generate_with_temperature() { ... }

#[tokio::test]
async fn test_streaming_response() { ... }

#[tokio::test]
async fn test_context_window_limit() { ... }

#[tokio::test]
async fn test_concurrent_generations() { ... }
```

**Tasks:**
- [ ] Test text generation
- [ ] Test parameter handling
- [ ] Test streaming responses
- [ ] Test error cases
- [ ] Test concurrent requests

### Step 10: Integration Testing (1.5 hours)

**Update:** `src-tauri/src/integration_tests.rs`

**New Tests:**
- [ ] Full chat completion workflow
- [ ] Multiple model loading
- [ ] Streaming response integrity
- [ ] Error handling in production

### Step 11: Documentation (1.5 hours)

**New File:** `INFERENCE.md`

Document:
- [ ] LLM inference architecture
- [ ] Supported generation parameters
- [ ] Performance tuning guide
- [ ] Troubleshooting common issues
- [ ] API streaming format

## Deliverables

### Code Changes

**New Files:**
- `src-tauri/src/inference/mod.rs` - Inference engine
- `src-tauri/src/inference/context_manager.rs` - Context management
- `src-tauri/src/inference/metrics.rs` - Performance monitoring
- `INFERENCE.md` - Inference documentation

**Modified Files:**
- `src-tauri/Cargo.toml` - Add llama.cpp dependency
- `src-tauri/src/lib.rs` - Add inference module
- `src-tauri/src/server.rs` - Real completions & streaming
- `src-tauri/src/error.rs` - Inference error types
- `src-tauri/src/integration_tests.rs` - Inference tests

### Features Enabled

- ✅ Real LLM inference using local models
- ✅ Response streaming support
- ✅ Temperature & generation parameters
- ✅ Context window management
- ✅ Model context caching
- ✅ Performance monitoring

## Testing Strategy

### Unit Tests
- Token generation with various parameters
- Context window boundary conditions
- Error handling for edge cases
- Metric calculations

### Integration Tests
- Full chat completion workflow
- Multiple sequential requests
- Model switching
- Streaming response format
- Concurrent requests

### Performance Tests
- Measure tokens/second
- Track memory usage
- Monitor API response times
- Benchmark different models

### Manual Testing
```bash
# 1. Start the app
pnpm tauri dev

# 2. Test via curl
curl http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mistral-7b",
    "messages": [{"role": "user", "content": "Hello"}],
    "temperature": 0.7,
    "max_tokens": 100
  }'

# 3. Test streaming
curl http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mistral-7b",
    "messages": [{"role": "user", "content": "Hello"}],
    "stream": true
  }'
```

## Success Criteria

- ✅ Real LLM responses (not mock)
- ✅ Streaming responses working
- ✅ All generation parameters supported
- ✅ Error handling robust
- ✅ Performance acceptable (>1 token/second)
- ✅ All tests passing
- ✅ Zero linting warnings
- ✅ Code properly formatted
- ✅ Documentation comprehensive

## Estimated Time

| Task | Hours |
|------|-------|
| Dependencies | 1 |
| Inference Module | 2 |
| Context Manager | 1.5 |
| Chat Endpoint | 1.5 |
| Streaming | 2 |
| Parameter Handling | 1.5 |
| Error Handling | 1.5 |
| Metrics | 1 |
| Unit Tests | 2 |
| Integration Tests | 1.5 |
| Documentation | 1.5 |
| **Total** | **~17.5 hours** |

## Risks & Mitigation

**Risk:** llama.cpp integration complexity
- **Mitigation:** Start simple, expand gradually; use stable Rust bindings

**Risk:** VRAM exhaustion with large models
- **Mitigation:** Implement model unloading, monitor memory usage

**Risk:** Slow inference affecting UX
- **Mitigation:** Use background tasks, implement streaming from start

**Risk:** GPU compatibility issues
- **Mitigation:** Fallback to CPU, test on multiple devices

**Risk:** Token encoding mismatches
- **Mitigation:** Use llama.cpp's tokenizer directly

## Next Phase Preview (Phase 4)

After Phase 3, we'll implement:
- Multi-model conversation history
- Advanced streaming features
- Chat session management
- Model switching mid-conversation
- Prompt templates & presets

## Rollback Plan

If Phase 3 needs rollback:
1. Revert inference module additions
2. Restore mock responses in server.rs
3. Remove llama.cpp dependency
4. Keep HTTP API structure (no breaking changes)

---

**Phase Status:** Ready to Begin  
**Estimated Start:** After Phase 2 ✅  
**Blocks:** Phase 4 (Advanced Features)  
**Unblocks:** Real LLM inference capability
