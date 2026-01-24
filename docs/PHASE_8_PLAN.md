# Phase 8: Multi-Backend Inference & Advanced Features

**Status**: 🔴 PLANNING  
**Date**: January 23, 2026  
**Target Tests**: 794+ (maintain existing + 50+ new tests)  
**Target Lint**: 0 violations  
**Estimated Duration**: 30-40 hours  

---

## 🎯 Phase 8 Vision

Transform Minerva from a single-backend system (llama.cpp only) into a **pluggable multi-backend inference engine** with advanced features like vision models, structured output, and intelligent backend selection.

### Why Phase 8?

**Current State (Phase 7):**
- ✅ Production-hardened llama.cpp backend
- ✅ Enterprise observability and resilience
- ✅ Solid architecture with `InferenceBackend` trait

**Gap Identified (Session Analysis):**
- ❌ Tokenization methods are mocked (lines 187-201 in `llama_adapter.rs`)
- ❌ No streaming token support in `generate()`
- ❌ Single backend only (no MLX, ONNX, or alternatives)
- ❌ No vision model capability
- ❌ No structured output (JSON guarantees)

**Proof of Concept:**
- ✅ LM Studio v0.3.4+ supports BOTH llama.cpp AND MLX natively
- ✅ mlx-lm is open-source (MIT licensed)
- ✅ Vision models proven via mlx-vlm
- ✅ Our architecture already supports this (trait-based design!)

---

## Phase 8 Goals

### Goal 1: Proper Tokenization (Priority: 🔴 CRITICAL)

**Current Problem:**
```rust
// llama_adapter.rs lines 187-201
fn tokenize(&self, text: &str) -> MinervaResult<Vec<i32>> {
    // Mock: split on whitespace - WRONG!
    Ok(text.split_whitespace()
        .enumerate()
        .map(|(i, _)| i as i32)
        .collect())
}
```

**Solution:**
- Implement proper BPE (Byte-Pair Encoding) tokenization
- Use the llama.cpp tokenizer directly (if exposed by llama-cpp-rs)
- Fallback to HuggingFace `tokenizers` crate if needed
- Support multiple tokenizer formats (BPE, Sentencepiece, etc.)

**Deliverables:**
- ✅ Real tokenization for LlamaCppBackend
- ✅ Tests comparing mock vs real tokenization
- ✅ Support for HuggingFace tokenizer files

---

### Goal 2: Streaming Token Generation (Priority: 🔴 CRITICAL)

**Current Problem:**
```rust
// llama_adapter.rs lines 152-185
fn generate(&self, prompt: &str, params: GenerationParams) -> MinervaResult<String> {
    // Returns full text - no streaming support!
    // SSE infrastructure exists but not used here
}
```

**Current State:**
- ✅ SSE infrastructure exists (streaming.rs, token_stream.rs)
- ✅ HTTP handler supports streaming responses
- ❌ InferenceBackend.generate() returns full text (not streamed)
- ❌ Tokens generated all at once, not token-by-token

**Solution:**
- Modify `InferenceBackend` trait to support token streaming
- Create `generate_streaming()` method that yields tokens
- Wire to existing SSE infrastructure
- Maintain backward-compatible `generate()` for batch use

**Deliverables:**
- ✅ `generate_streaming()` on InferenceBackend trait
- ✅ Token-by-token generation in LlamaCppBackend
- ✅ Integration with HTTP streaming endpoints
- ✅ Performance tests showing token latency

---

### Goal 3: MLX Backend Implementation (Priority: 🟡 HIGH)

**Architecture:**
```rust
// New file: src-tauri/src/inference/mlx_backend.rs
pub struct MlxBackend {
    python_runtime: PythonRuntime,
    model_cache: Arc<Mutex<Option<MlxModel>>>,
    // ...
}

impl InferenceBackend for MlxBackend {
    // Delegate to Python mlx-lm via PyO3
}
```

**Why MLX?**
- Proven in production by LM Studio
- Faster on Apple Silicon (metal.so ops)
- Access to HuggingFace full ecosystem
- Vision models via mlx-vlm
- MIT licensed (open source)

**Challenges & Solutions:**
| Challenge | Solution |
|-----------|----------|
| Python runtime (~600MB) | Optional feature flag + lazy load |
| PyO3 complexity | Use subprocess instead of embedding |
| Model format differences | Auto-detect via file header |
| Version management | Pin mlx-lm==version |

**Implementation Approach (Subprocess-Based):**
```rust
// Simpler than PyO3, same results
let output = Command::new("python3")
    .arg("-m").arg("mlx_lm.server")
    .arg("--model").arg(model_path)
    .spawn()?;

// Communicate via HTTP (mlx-lm exposes OpenAI-compatible API)
let response = client.post("http://localhost:8000/v1/completions")
    .json(&request)
    .send()?;
```

**Deliverables:**
- ✅ MlxBackend struct implementing InferenceBackend
- ✅ Subprocess-based Python runtime integration
- ✅ Model format detection (GGUF vs HuggingFace)
- ✅ Error recovery if mlx-lm not installed
- ✅ 20+ integration tests

---

### Goal 4: Backend Selection & Routing (Priority: 🟡 HIGH)

**Current State:**
```rust
// inference/mod.rs - hardcoded backend
let backend = LlamaCppBackend::new();
```

**Solution: Smart Backend Selector**
```rust
pub enum BackendSelector {
    Auto,           // Detect best backend for model
    LlamaCpp,       // Force llama.cpp (GGUF only)
    Mlx,            // Force MLX (HuggingFace models)
    Fallback,       // Try primary, fallback to secondary
}

pub fn select_backend(
    model_path: &Path,
    preference: BackendSelector,
) -> MinervaResult<Box<dyn InferenceBackend>> {
    match preference {
        BackendSelector::Auto => {
            if model_path.ends_with(".gguf") {
                Ok(Box::new(LlamaCppBackend::new()))
            } else {
                Ok(Box::new(MlxBackend::new()?))
            }
        }
        // ...
    }
}
```

**API Changes:**
```typescript
// Before
POST /v1/chat/completions
{
  "model": "mistral-7b.gguf",
  // ...
}

// After
POST /v1/chat/completions
{
  "model": "mistral-7b.gguf",
  "backend": "auto",  // "llama_cpp" | "mlx" | "auto" | "fallback"
  // ...
}
```

**Deliverables:**
- ✅ BackendSelector enum and logic
- ✅ Model format detection
- ✅ API endpoint support for backend selection
- ✅ Fallback chain implementation
- ✅ Tests for all selector modes

---

### Goal 5: Vision Model Support (Priority: 🟢 OPTIONAL - Stretch Goal)

**Only if time permits after Goals 1-4.**

**What:** Add support for multimodal LLMs (image + text inputs)
- LLaVA models (via llama.cpp or mlx-vlm)
- Vision capabilities on Apple Silicon
- Image preprocessing pipeline

**Deliverables (if completed):**
- ✅ VisionModel trait (extends InferenceBackend)
- ✅ Image preprocessing (resize, normalize)
- ✅ llama-cpp-rs vision support OR mlx-vlm integration
- ✅ HTTP endpoint for image uploads
- ✅ Tests with sample vision tasks

---

## 📋 Detailed Implementation Steps

### Step 1: Proper BPE Tokenization (Days 1-2)

**Day 1: Research & Design**
1. [ ] Research llama.cpp tokenizer exposure in llama-cpp-rs
2. [ ] Evaluate `tokenizers` crate vs custom BPE
3. [ ] Design tokenizer trait for pluggability
4. [ ] Create test suite comparing mock vs real tokenization

**Day 2: Implementation**
1. [ ] Implement real BPE tokenizer
2. [ ] Update LlamaCppBackend.tokenize() & detokenize()
3. [ ] Add HuggingFace tokenizer file support
4. [ ] Update tests, ensure all pass
5. [ ] Verify zero lint violations

**Tests to Add:**
- Real vs mock tokenization accuracy (10+ tests)
- Various model formats (Llama, Mistral, etc.)
- Edge cases (empty input, special tokens, etc.)

---

### Step 2: Streaming Token Generation (Days 3-4)

**Day 3: Trait Design & Backend Updates**
1. [ ] Design streaming trait:
   ```rust
   pub trait InferenceBackend {
       // Existing
       fn generate(&self, prompt: &str, params: GenerationParams) 
           -> MinervaResult<String>;
       
       // New
       fn generate_streaming(&self, prompt: &str, params: GenerationParams)
           -> MinervaResult<Box<dyn Iterator<Item = MinervaResult<String>>>>;
   }
   ```
2. [ ] Implement streaming in LlamaCppBackend
3. [ ] Wire to token_stream.rs infrastructure

**Day 4: HTTP Integration & Testing**
1. [ ] Update chat completions endpoint to use streaming
2. [ ] Test token-by-token delivery via SSE
3. [ ] Performance benchmarks (latency/token)
4. [ ] Integration tests with real streaming

**Tests to Add:**
- Token streaming correctness (5+ tests)
- Performance under load (3+ tests)
- Error handling in streams (3+ tests)
- Backward compat with non-streaming (2+ tests)

---

### Step 3: MLX Backend (Days 5-8)

**Day 5: Research & Environment Setup**
1. [ ] Analyze mlx-lm subprocess architecture
2. [ ] Design MlxBackend struct
3. [ ] Research model format detection
4. [ ] Plan error handling for missing mlx-lm

**Day 6: Core Implementation**
1. [ ] Create src-tauri/src/inference/mlx_backend.rs
2. [ ] Implement InferenceBackend for MlxBackend
3. [ ] Subprocess-based model loading
4. [ ] OpenAI API HTTP client integration

**Day 7: Integration & Error Handling**
1. [ ] Model format detection (GGUF vs HuggingFace)
2. [ ] Fallback if mlx-lm not installed
3. [ ] Error recovery strategies
4. [ ] Integration with existing model registry

**Day 8: Testing & Documentation**
1. [ ] 20+ integration tests for MLX backend
2. [ ] Mock tests (don't require mlx-lm installed)
3. [ ] Performance comparison (MLX vs llama.cpp)
4. [ ] Documentation with setup instructions

**Tests to Add:**
- MLX backend lifecycle (5+ tests)
- Model loading and inference (5+ tests)
- Error scenarios (mlx-lm missing, etc.) (5+ tests)
- Format detection and routing (5+ tests)

---

### Step 4: Backend Selection & Routing (Days 9-10)

**Day 9: Selection Logic**
1. [ ] Implement BackendSelector enum
2. [ ] Model format detection logic
3. [ ] Fallback chain implementation
4. [ ] Configuration storage

**Day 10: API Integration**
1. [ ] Add "backend" parameter to chat completions endpoint
2. [ ] Update Tauri commands
3. [ ] Frontend UI for backend selection
4. [ ] Comprehensive routing tests

**Tests to Add:**
- Backend selection logic (8+ tests)
- Format-based routing (5+ tests)
- Fallback chains (5+ tests)

---

### Step 5: Vision Models (Days 11-12, IF TIME PERMITS)

**Day 11: Vision Infrastructure**
1. [ ] Design VisionModel trait
2. [ ] Image preprocessing pipeline
3. [ ] Research llama-cpp-rs or mlx-vlm support

**Day 12: Implementation**
1. [ ] LLaVA model loading
2. [ ] Image upload HTTP endpoint
3. [ ] Vision-capable inference
4. [ ] Tests and documentation

---

## 🏗️ Architecture Changes

### Current Architecture (Phase 7)
```
HTTP Request
    ↓
[Server.rs] → [Commands.rs] → [Inference Engine]
                                    ↓
                            [LlamaCppBackend] (only option)
                                    ↓
                            llama.cpp Binary
```

### New Architecture (Phase 8)
```
HTTP Request
    ↓
[Server.rs] → [Commands.rs] → [Inference Engine]
                                    ↓
                        [Backend Selector]
                         /        |        \
                        /         |         \
            [LlamaCpp]    [MLX]    [ONNX?]
            Backend       Backend  Backend
                ↓            ↓        ↓
            llama.cpp    mlx-lm   onnxruntime
            Binary       Python   Native
```

### Key Files to Create/Modify

**New Files:**
- `src-tauri/src/inference/mlx_backend.rs` - MLX implementation
- `src-tauri/src/inference/backend_selector.rs` - Selection logic
- `docs/BACKEND_SELECTION.md` - User documentation
- `tests/integration/mlx_backend.rs` - MLX tests

**Modified Files:**
- `src-tauri/src/inference/llama_adapter.rs` - Add streaming, real tokenization
- `src-tauri/src/inference/mod.rs` - Wire new backends
- `src-tauri/src/server.rs` - Add backend parameter to endpoints
- `src-tauri/tests/integration/mod.rs` - Register new tests

---

## 📊 Success Criteria

### Phase 8 Complete When:

| Criterion | Target | Status |
|-----------|--------|--------|
| Tests passing | 794+ | 🔴 TBD |
| Lint violations | 0 | 🔴 TBD |
| Proper tokenization | Real BPE | 🔴 TBD |
| Streaming generation | Token-by-token SSE | 🔴 TBD |
| MLX backend | Full integration | 🔴 TBD |
| Backend selection | Smart routing | 🔴 TBD |
| Vision models | ✅ or ⏭️ (optional) | 🔴 TBD |
| Documentation | Complete | 🔴 TBD |

### Performance Targets

| Metric | Target | Current |
|--------|--------|---------|
| Model load time | <2s | ~0.5s ✅ |
| Token latency | <100ms | ? (measure) |
| SSE overhead | <5ms | ? (measure) |
| Memory per model | <2GB | ~1.5GB ✅ |

---

## ⚠️ CRITICAL VALIDATION PROTOCOL

**MUST RUN BEFORE EVERY COMMIT:**
```bash
# Always run BOTH checks
pnpm lint      # Backend + Frontend linting (expect 0 violations)
pnpm test      # All tests (expect 794+ passing)
```

**Tests Must Pass:**
- ✅ 794+ tests passing (was 215 integration + 579 unit)
- ✅ 50+ new Phase 8 tests
- ✅ 0 lint violations
- ✅ 0 warnings
- ✅ All modules compile

**If either check fails:**
1. 🛑 STOP - Do not commit
2. 🔍 Investigate root cause
3. 🔧 Fix the issue
4. ♻️ Re-run checks
5. ✅ Only then commit

---

## 🔗 Dependencies & Prerequisites

### Required Crates

**For Tokenization:**
- `tokenizers` (MIT) - HuggingFace tokenizer support
- `serde` (MIT/Apache 2.0) - Serialization

**For Streaming:**
- (Already have: futures, tokio, axum)

**For MLX Backend:**
- `reqwest` (MIT/Apache 2.0) - HTTP client (already have)
- `pyo3` (Apache 2.0) - Python runtime (optional, use subprocess instead)

**For Vision (Optional):**
- `image` (MIT) - Image processing
- `ndarray` (MIT/Apache 2.0) - Tensor operations

### External Software

**For MLX Backend:**
- Python 3.9+ (user must install)
- mlx-lm package (pip install mlx-lm)
- mlx-vlm package (if vision models wanted)

**Installation check in code:**
```rust
fn check_mlx_available() -> MinervaResult<()> {
    let output = Command::new("python3")
        .arg("-m").arg("mlx_lm.cli.convert")
        .arg("--help")
        .output()?;
    
    if !output.status.success() {
        return Err(MinervaError::Configuration(
            "mlx-lm not installed. Run: pip install mlx-lm".to_string()
        ));
    }
    Ok(())
}
```

---

## 📚 Reference Implementations

### LM Studio (Reference)
- **URL:** https://github.com/lmstudio-ai/lmstudio
- **Approach:** Single GUI wrapping multiple backends
- **Backends:** llama.cpp (GGUF) + mlx-engine (HuggingFace)
- **Key Learning:** Subprocess-based backend integration is simpler than embedding

### mlx-lm (Reference)
- **URL:** https://github.com/ml-explore/mlx-examples/tree/main/llm
- **Features:** OpenAI-compatible API, model download, inference
- **License:** MIT (use freely)

### mlx-vlm (Reference)
- **URL:** https://github.com/ml-explore/mlx-vlm
- **Features:** Vision models (LLaVA), image processing
- **License:** MIT (use freely)

---

## 🎓 Lessons from Session Analysis

### What We Learned

1. **Architecture is Future-Ready**
   - ✅ InferenceBackend trait supports multiple implementations
   - ✅ No refactoring needed for Phase 8
   - ✅ Design enables vision, ONNX, other backends

2. **vLLM Was Wrong Call**
   - ❌ Server framework (we're desktop)
   - ❌ No Metal GPU support
   - ❌ Would add 2GB to app
   - ✅ We rejected it correctly

3. **MLX is Proven by LM Studio**
   - ✅ Open source, MIT licensed
   - ✅ Faster on Apple Silicon
   - ✅ Access to full HuggingFace ecosystem
   - ✅ Vision models proven

4. **Current llama.cpp Stack is Optimal**
   - ✅ Small footprint (100MB app)
   - ✅ Fast startup (<1 second)
   - ✅ Excellent quantization ecosystem
   - ✅ Keep as primary backend

---

## 🚀 Rollout Plan

### Milestones

**Milestone 1 (Days 1-2):** Tokenization ✅
- Proper BPE implementation
- Tests passing
- Backward compatible

**Milestone 2 (Days 3-4):** Streaming ✅
- Token-by-token generation
- SSE integration
- Performance baseline

**Milestone 3 (Days 5-8):** MLX Backend ✅
- Full MLX integration
- Format detection
- 20+ tests passing

**Milestone 4 (Days 9-10):** Backend Selection ✅
- Smart routing
- Fallback chains
- API integration

**Milestone 5 (Days 11-12):** Vision (Optional) ⏳
- If time permits
- LLaVA models
- Image preprocessing

---

## 💾 Git Strategy

**Branch:** `phase-8/multi-backend`
**Commits:**
1. `feat(phase8): Implement proper BPE tokenization`
2. `feat(phase8): Add streaming token generation`
3. `feat(phase8): Implement MLX backend adapter`
4. `feat(phase8): Add smart backend selection`
5. `feat(phase8): Add vision model support (optional)`

**Verification at each commit:**
```bash
pnpm lint && pnpm test
# Must pass before moving to next commit
```

---

## 📖 Documentation to Write

1. **PHASE_8_COMPLETE.md** - Post-completion summary
2. **BACKEND_SELECTION.md** - User guide for backend selection
3. **MLX_BACKEND_SETUP.md** - MLX installation and configuration
4. **VISION_MODELS.md** - Vision model usage (if completed)
5. **TOKENIZATION.md** - Tokenizer architecture and implementation
6. **STREAMING.md** - Streaming generation architecture

---

## ❓ Open Questions

1. **Timeline:** When should we start Phase 8?
2. **Scope:** Do we include vision models or keep to high-priority goals?
3. **MLX Availability:** Should MLX backend be optional feature flag?
4. **Performance:** What are acceptable degradation metrics?
5. **Backward Compatibility:** Must Phase 8 maintain Phase 7 API?

**Recommendation:** Yes to all except question 1 (depends on user priority).

---

## 📝 Notes

- This plan follows the same rigor as Phase 7
- Architecture supports future backends (ONNX, TensorRT, etc.)
- Subprocess-based MLX integration avoids PyO3 complexity
- All phases maintain: zero lint violations, all tests passing
- Vision models are nice-to-have, not must-have

---

**Last Updated:** January 23, 2026  
**Prepared by:** Build Agent  
**Status:** Ready for Review & Prioritization
