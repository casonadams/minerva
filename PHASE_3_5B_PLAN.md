# Phase 3.5b: GPU Acceleration & Real llama.cpp Integration

**Status:** Planning & Research  
**Objective:** Integrate real LLM inference with GPU acceleration  
**Timeline:** 6-8 hours estimated  
**Complexity:** High - requires proper error handling and performance tuning  

---

## Overview

Phase 3.5b transforms the mock inference engine into a production-ready LLM server with:
1. Real llama.cpp-based inference
2. GPU acceleration (Metal on macOS, CUDA on Linux/Windows)
3. Token streaming with callbacks
4. Performance optimization
5. Comprehensive error handling

---

## Architecture Overview

```
HTTP Request (POST /v1/chat/completions)
    ↓
ParameterParser (validate + parse)
    ↓
GpuContext (detect GPU, allocate memory)
    ↓
LlamaCppBackend::load_model()
    ├─ Parse GGUF file path
    ├─ Create LlamaParams
    ├─ Load: LlamaModel::load_from_file()
    └─ Create: model.create_context()
    ↓
TokenStream (prepare streaming)
    ↓
LlamaCppBackend::generate()
    ├─ Tokenize prompt
    ├─ Evaluate tokens
    ├─ Sample loop (with temperature/top_p)
    ├─ Callback: stream.push_token(token)
    └─ Repeat until EOS
    ↓
StreamingResponse (SSE format)
    ↓
HTTP Response (chunked streaming)
```

---

## Implementation Steps

### Step 1: Research & Setup ✅
**Objectives:**
- Understand llama_cpp crate API
- Review GPU acceleration options
- Plan error handling

**Key Findings:**
```
llama_cpp crate (v0.3.2):
├── LlamaModel::load_from_file(path, params)
├── LlamaContext (context management)
├── Sampling & token generation
└── GPU support (Metal, CUDA configurable)
```

### Step 2: Implement LlamaCppBackend (2 hours)

**File:** `src/inference/llama_adapter.rs`

**LlamaCppBackend Structure:**
```rust
pub struct LlamaCppBackend {
    model: Option<LlamaModel>,
    context: Option<LlamaContext>,
    n_ctx: usize,
    n_threads: usize,
}

impl InferenceBackend for LlamaCppBackend {
    fn load_model(&mut self, path: &Path, n_ctx: usize) -> MinervaResult<()> {
        // 1. Create LlamaParams with n_ctx
        let params = LlamaParams::default()
            .with_context_size(n_ctx as u32)
            .with_n_gpu_layers(40);  // GPU layers
        
        // 2. Load model
        self.model = LlamaModel::load_from_file(path, params)?;
        
        // 3. Create context
        self.context = self.model.create_context()?;
        
        // 4. Store metadata
        self.n_ctx = n_ctx;
        self.n_threads = num_cpus::get();
        
        Ok(())
    }
    
    fn generate(&self, prompt: &str, max_tokens: usize, 
                temperature: f32, top_p: f32) -> MinervaResult<String> {
        // 1. Tokenize
        let tokens = self.model.tokenize(prompt)?;
        
        // 2. Validate context fit
        if tokens.len() + max_tokens > self.n_ctx {
            return Err(MinervaError::ContextLimitExceeded {
                max: self.n_ctx,
                required: tokens.len() + max_tokens,
            });
        }
        
        // 3. Evaluate
        self.context.eval(&tokens, self.n_threads)?;
        
        // 4. Sample tokens
        let mut generated = Vec::new();
        for _ in 0..max_tokens {
            let token = self.context.sample(
                temperature,
                top_p,
                40,        // top_k
                1.1,       // repeat_penalty
            );
            
            if token < 0 {  // EOS
                break;
            }
            generated.push(token);
        }
        
        // 5. Detokenize
        let text = self.model.detokenize(&generated)?;
        Ok(text)
    }
    
    fn tokenize(&self, text: &str) -> MinervaResult<Vec<i32>> {
        self.model.tokenize(text)
    }
    
    fn detokenize(&self, tokens: &[i32]) -> MinervaResult<String> {
        self.model.detokenize(tokens)
    }
}
```

**Deliverables:**
- [ ] Full LlamaCppBackend implementation
- [ ] Error handling for all operations
- [ ] Token sampling with parameters
- [ ] 8 unit tests

### Step 3: GPU Acceleration (1.5 hours)

**File:** `src/inference/gpu_context.rs`

**GPU Initialization:**
```rust
impl GpuContext {
    pub fn initialize(&mut self) -> MinervaResult<()> {
        match self.device {
            GpuDevice::Metal => {
                // Metal context already initialized on macOS
                tracing::info!("Metal GPU context ready");
            }
            GpuDevice::Cuda => {
                // CUDA initialization
                // Load CUDA libraries
                // Set device properties
                tracing::info!("CUDA GPU context ready");
            }
            GpuDevice::Cpu => {
                tracing::info!("CPU-only inference");
            }
        }
        Ok(())
    }
}
```

**LlamaParams GPU Configuration:**
```rust
// Metal (Apple Silicon) - unified memory
let params = LlamaParams::default()
    .with_gpu_device(0)           // Metal device
    .with_n_gpu_layers(40);       // Offload layers to GPU

// CUDA - dedicated VRAM
let params = LlamaParams::default()
    .with_gpu_device(0)           // GPU device ID
    .with_n_gpu_layers(40)        // Offload to VRAM
    .with_main_gpu(0);            // Primary GPU

// CPU - no GPU
let params = LlamaParams::default()
    .with_n_gpu_layers(0);        // All CPU
```

**Deliverables:**
- [ ] Metal GPU initialization
- [ ] CUDA GPU initialization
- [ ] CPU fallback
- [ ] 6 GPU tests

### Step 4: Token Streaming with Callbacks (1.5 hours)

**File:** `src/inference/token_stream.rs` (enhance)

**Streaming Pattern:**
```rust
pub async fn stream_generation(
    backend: &LlamaCppBackend,
    prompt: &str,
    stream: TokenStream,
) -> MinervaResult<()> {
    // Spawn inference in background
    tokio::spawn(async move {
        let tokens = backend.tokenize(prompt).unwrap();
        backend.context.eval(&tokens, backend.n_threads).unwrap();
        
        // Token generation loop
        for _ in 0..max_tokens {
            let token = backend.context.sample(temperature, top_p, top_k, repeat_penalty);
            if token < 0 { break; }  // EOS
            
            let text = backend.model.detokenize(&[token]).unwrap();
            stream.push_token(text);  // Thread-safe push
        }
    });
    
    Ok(())
}
```

**HTTP Streaming Endpoint:**
```rust
// Stream the tokens to client
axum::response::sse::Sse::new(
    stream_tokens(backend, prompt, params)
        .map(|token| {
            let chunk = ChatCompletionChunk { /* ... */ };
            Ok(Event::default().data(serde_json::to_string(&chunk)?))
        })
        .throttle(Duration::from_millis(10))  // Rate limit
)
```

**Deliverables:**
- [ ] Streaming token collection
- [ ] HTTP SSE endpoint
- [ ] Async/tokio integration
- [ ] 5 streaming tests

### Step 5: Error Handling & Recovery (1 hour)

**Error Categories:**

```rust
// Model Loading Errors
├── FileNotFound
├── InvalidFormat
├── MemoryAllocated
├── GPUNotAvailable
└── ContextTooSmall

// Inference Errors
├── ContextOverflow
├── TokenizationFailed
├── SamplingError
├── TimeoutExceeded
└── OutOfMemory

// Recovery Strategies
├── Fallback to CPU
├── Retry with smaller context
├── Clear cache and retry
├── Graceful degradation
└── User-friendly errors
```

**Implementation:**
```rust
pub fn generate_with_fallback(
    backend: &mut LlamaCppBackend,
    prompt: &str,
    config: &GenerationConfig,
) -> MinervaResult<String> {
    // Try with GPU first
    match backend.generate(prompt, config.max_tokens, config.temperature, config.top_p) {
        Ok(text) => return Ok(text),
        Err(MinervaError::OutOfMemory(_)) => {
            tracing::warn!("GPU OOM, falling back to CPU");
            // Reload with n_gpu_layers=0
            backend.reload_cpu_only()?;
            return backend.generate(prompt, config.max_tokens, config.temperature, config.top_p);
        }
        Err(e) => return Err(e),
    }
}
```

**Deliverables:**
- [ ] Comprehensive error handling
- [ ] Fallback mechanisms
- [ ] 6 error handling tests
- [ ] Clear error messages

### Step 6: Performance & Benchmarking (1 hour)

**Metrics to Track:**
```rust
pub struct InferenceMetrics {
    model_load_time_ms: u128,
    prompt_tokens: usize,
    generated_tokens: usize,
    inference_time_ms: u128,
    tokens_per_second: f32,
    memory_used_mb: u64,
    gpu_utilized: bool,
}
```

**Benchmark Tests:**
```rust
#[test]
fn bench_7b_model_inference() {
    // Load 7B model
    // Generate 256 tokens
    // Measure: latency, throughput, memory
    // Assert: meets performance requirements
}

#[test]
fn bench_gpu_vs_cpu() {
    // Compare Metal vs CPU
    // Compare CUDA vs CPU
    // Verify GPU provides speedup
}
```

**Expected Performance:**
| Operation | GPU (Metal) | GPU (CUDA) | CPU |
|-----------|-----------|-----------|-----|
| 7B Load | 2-5s | 2-5s | 1-3s |
| 256 Tokens | 100-300ms | 80-200ms | 500-2000ms |
| Tokens/sec | 50-100 | 100-200 | 10-50 |

**Deliverables:**
- [ ] Performance benchmarks
- [ ] Metrics collection
- [ ] 4 benchmark tests
- [ ] Performance documentation

### Step 7: Integration Tests (1 hour)

**End-to-End Tests:**
```rust
#[test]
fn test_real_inference_7b_model() {
    // Load real 7B model
    // Generate response
    // Verify quality and performance
}

#[test]
fn test_gpu_acceleration() {
    // Load model on GPU
    // Generate tokens
    // Verify GPU is utilized
}

#[test]
fn test_streaming_large_response() {
    // Stream 1000+ tokens
    // Verify all tokens received
    // Check performance
}

#[test]
fn test_concurrent_requests() {
    // 5 concurrent chat requests
    // Verify all complete correctly
    // Check performance degradation
}
```

**Deliverables:**
- [ ] 8 integration tests
- [ ] Performance verification
- [ ] GPU verification
- [ ] Concurrent request tests

### Step 8: Documentation (30 mins)

**Files to Update:**
- [ ] `PHASE_3_5B_COMPLETION.md` - Implementation details
- [ ] `README.md` - GPU support info
- [ ] `docs/GPU_ACCELERATION.md` - Technical details
- [ ] Inline code comments

---

## Technical Challenges & Solutions

### Challenge 1: Model Loading with GPU
**Problem:** Different GPU APIs require different initialization
**Solution:** Abstract via `GpuContext::detect_device()` and conditional compilation

### Challenge 2: Token Streaming Performance
**Problem:** Callback overhead with each token
**Solution:** Batch tokens (5-10) before sending to reduce overhead

### Challenge 3: Memory Management
**Problem:** GPU memory fills quickly
**Solution:** Monitor via `GpuContext`, fallback to CPU if needed

### Challenge 4: Concurrent Requests
**Problem:** Multiple inference requests compete for GPU
**Solution:** Request queue with priority, fallback to CPU

### Challenge 5: Error Recovery
**Problem:** GPU failures need graceful handling
**Solution:** Try GPU → fallback to CPU → return error

---

## Testing Strategy

### Unit Tests (per module)
- llama_adapter: 12 tests (load, generate, tokenize, etc.)
- gpu_context: 8 tests (detection, allocation, etc.)
- token_stream: 6 tests (streaming, callbacks, etc.)

### Integration Tests (end-to-end)
- Real model inference (4 tests)
- GPU acceleration (3 tests)
- Streaming (2 tests)
- Concurrent requests (2 tests)

### Performance Benchmarks
- Model loading time
- Generation throughput
- GPU utilization
- Memory usage
- Concurrent request impact

**Total New Tests:** 37 (bringing total to 138)

---

## Quality Checklist

Before completion, verify:

- [ ] **Compilation**
  - [ ] Zero clippy warnings
  - [ ] All features compile
  - [ ] Optional GPU features work

- [ ] **Tests**
  - [ ] 138 total tests passing
  - [ ] All unit tests pass
  - [ ] All integration tests pass
  - [ ] Benchmarks meet targets

- [ ] **Formatting**
  - [ ] 100% code format compliance
  - [ ] Zero warnings
  - [ ] Clear comments

- [ ] **Performance**
  - [ ] GPU provides speedup
  - [ ] Memory usage acceptable
  - [ ] Concurrent requests work

- [ ] **Documentation**
  - [ ] API documented
  - [ ] Error codes explained
  - [ ] GPU setup instructions
  - [ ] Troubleshooting guide

---

## Risks & Mitigation

| Risk | Impact | Mitigation |
|------|--------|-----------|
| llama.cpp API incompatibility | High | Test with current version, pin dependency |
| GPU memory exhaustion | High | Monitor memory, fallback to CPU |
| CUDA not available on test system | Medium | Skip CUDA tests if unavailable |
| Performance regression | Medium | Benchmark before/after |
| Breaking changes to API | Medium | Version compatibility testing |

---

## Success Criteria

✅ **Functional:**
- Real LLM inference works end-to-end
- GPU acceleration active and verified
- Streaming produces proper SSE format
- Error handling prevents crashes

✅ **Performance:**
- GPU provides 5-10x speedup
- Tokens stream smoothly (no buffering)
- Concurrent requests handled
- Memory stays under limits

✅ **Quality:**
- 138 tests passing
- 0 warnings
- 100% format compliance
- Comprehensive documentation

✅ **Robustness:**
- Graceful GPU fallback
- Proper error messages
- Recovery from failures
- Production-ready code

---

## Post-Phase Work

### Phase 3.5c: Optimization (Future)
- Request batching
- KV cache optimization
- Prompt caching
- Multi-model serving

### Phase 4: Advanced Features (Future)
- Chat session management
- Conversation history
- Model switching
- Prompt templates

---

## Resources

### llama.cpp Crate
- API: https://docs.rs/llama_cpp/0.3/
- Examples: https://github.com/setzer22/llama-rs

### GPU Acceleration
- Metal: https://developer.apple.com/metal/
- CUDA: https://docs.nvidia.com/cuda/

### Performance Tuning
- Rust Performance: https://nnethercote.github.io/perf-book/
- LLM Optimization: https://github.com/ggerganov/llama.cpp/wiki

---

## Summary

Phase 3.5b transforms the Minerva inference engine from mock to production-ready with real LLM inference and GPU acceleration. The modular architecture (via `InferenceBackend` trait) allows seamless integration without breaking existing code.

**Estimated Completion:** 6-8 hours  
**Tests Delivered:** +37 (total 138)  
**Documentation:** Complete  
**Production Readiness:** Full  

Starting with Step 1 research. Let's build real inference! 🚀
