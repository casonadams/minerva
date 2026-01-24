# Phase 8-Step 3b: MLX Subprocess Integration Roadmap

**Status:** Ready to Start  
**Estimated Duration:** 2-3 days  
**Prerequisites:** MLX Backend Foundation (✅ Complete)  
**Target Tests:** 806 → 830+ (add 24+ integration tests)  
**Target Lint:** 0 violations (maintained)  

---

## Overview

**Goal:** Connect the MlxBackend foundation to actual mlx-lm via subprocess, enabling real HuggingFace model inference.

**What Exists:**
- ✅ MlxBackend struct fully defined
- ✅ InferenceBackend trait implemented
- ✅ Model format detection working
- ✅ 8 unit tests proving foundation

**What We'll Add:**
- 🔨 Real subprocess integration with mlx-lm
- 🔨 HTTP client for mlx-lm server API
- 🔨 Model loading via subprocess
- 🔨 Caching for loaded models
- 🔨 Error recovery and fallbacks
- 🔨  24+ integration tests

---

## Implementation Roadmap - REVISED: Pure Rust Approach

**⚠️ CRITICAL DECISION:** No Python subprocess! All inference in pure Rust.

### Architecture: Why Pure Rust?

**❌ Python Subprocess Problems:**
- Process spawn overhead on every request
- Complex process management (start/stop/restart)
- Python runtime dependency (+600MB)
- Network overhead (subprocess → HTTP → subprocess)
- Hard to debug and maintain
- Performance hit from serialization/deserialization

**✅ Pure Rust Solution:**
- No external dependencies (use what we have)
- Minimal performance overhead
- Direct inference without serialization
- Cleaner architecture
- Easier testing and debugging
- Self-contained, portable

### Design: Hybrid Backend Strategy

Instead of MlxBackend calling Python, implement **intelligent model format detection**:

```rust
// Strategy: Use what we have + what works
match model_format {
    Format::GGUF => {
        // Use LlamaCppBackend (existing, proven)
        // Fast, efficient, no dependencies
        use_llama_cpp_backend(model)
    }
    Format::HuggingFaceFormat => {
        // Option 1: Support via llama.cpp (works with many HF models)
        // Option 2: Implement light-weight pure Rust inference
        // Don't spawn Python subprocess!
    }
}
```

### Day 1: Research & Architecture

#### Morning: Understand Model Formats (1 hour)

**Question:** What formats do users actually need?

1. **GGUF** (90% of cases)
   - ✅ Already supported via llama.cpp
   - Already working
   - Don't change this!

2. **HuggingFace Format** (safetensors, .bin, etc.)
   - Many models available
   - But can they be converted to GGUF?
   - Research: llama.cpp can load some HF models directly!

3. **Other formats?**
   - ONNX, TensorFlow, etc.
   - Are these needed for Phase 8?
   - Probably not MVP-critical

**Key Insight:** Maybe MlxBackend isn't needed yet. Improve LlamaCppBackend to handle more formats instead!

#### Afternoon: Design Pure Rust Inference (2 hours)

**Option 1: Extend LlamaCppBackend** (Simplest)
```rust
// LlamaCppBackend can already load GGUF
// Can it load HuggingFace safetensors?
// Research what llama.cpp supports natively

// If yes → Just improve llama_adapter.rs, don't create MlxBackend!
// If no → Create lightweight HF loader
```

**Option 2: Lightweight Rust HF Loader** (If needed)
```rust
// Create RustInferenceBackend that:
// 1. Loads safetensors files (pure Rust, no Python)
// 2. Implements basic transformer inference
// 3. Doesn't try to be feature-complete
// 4. Focuses on common models (Llama, Mistral, Phi, etc.)

// Crates available:
// - `safetensors` - Load HF model weights (MIT licensed!)
// - `ndarray` - Tensor operations (already used)
// - `nalgebra` - Linear algebra
// - `burn` - Rust DL framework (but might be overkill)
```

**Recommendation:** Start with Option 1 (Extend llama.cpp support)
- Less code to write
- Proven, tested technology
- Better performance
- No new dependencies

#### Evening: Plan Testing Strategy (1 hour)

**Test Pure Rust Instead of Python:**
```rust
// No more:
// - Installing Python
// - Running subprocess commands
// - Dealing with Python paths

// Instead:
// - Load model files directly
// - Test with real model weights
// - Benchmark vs llama.cpp
// - Profile memory usage
```

---

### Day 2: Implementation (Pure Rust Only)

**Strategy: Improve LlamaCppBackend Instead**

Rather than creating new Python-calling infrastructure, enhance what we have:

```rust
// APPROACH: Use llama.cpp for everything it can handle
// - GGUF format: 100% supported
// - HuggingFace safetensors: Research if llama.cpp can load
// - Other formats: Convert to GGUF (user responsibility or auto-convert)

// If llama.cpp doesn't support format X:
// - Don't spawn Python subprocess!
// - Instead: Create light-weight pure Rust loader

// Key: All code stays in Rust. No external process calls.
```

#### Morning Option A: Research llama.cpp Capabilities (2 hours)

Check what llama.cpp/llama-cpp-rs already supports:
```bash
# Questions to research:
1. Can llama.cpp load safetensors directly?
2. Can llama.cpp load HuggingFace .bin format?
3. What's the complete list of supported formats?
4. Are there Rust crates that already solve this?
```

**Likely findings:**
- llama.cpp primarily supports GGUF
- For other formats, users convert to GGUF first
- This is actually the right approach (quantization matters!)

**Implication:** Maybe MlxBackend can just wrap llama.cpp with format detection + auto-conversion guide?

#### Morning Option B: Add Pure Rust Inference Layer (If Needed)

If llama.cpp doesn't support needed formats, create lightweight Rust layer:

```rust
// New file: src-tauri/src/inference/pure_rust_inference.rs

// Use these crates (already in dependencies or add sparingly):
use ndarray::Array2;  // Tensors (already in use)
use serde_json;       // Config parsing

// Minimal implementation:
// 1. Load model weights from safetensors (pure Rust crate!)
// 2. Implement forward pass for common architectures
// 3. Focus on Llama, Mistral, Phi (90% of use cases)

// Performance: Won't match llama.cpp, but:
// - No external dependencies
// - Pure Rust, fully controlled
// - Good enough for basic inference
// - Can be optimized later

pub struct PureRustBackend {
    weights: HashMap<String, ndarray::Array2<f32>>,
    config: ModelConfig,
    n_ctx: usize,
}

impl InferenceBackend for PureRustBackend {
    fn generate(&self, prompt: &str, params: GenerationParams) -> MinervaResult<String> {
        // Tokenize input
        let tokens = self.tokenize(prompt)?;
        
        // Run transformer forward pass
        let mut output_tokens = Vec::new();
        for _ in 0..params.max_tokens {
            let logits = self.forward_pass(&tokens)?;
            let next_token = self.sample_token(&logits, params.temperature)?;
            output_tokens.push(next_token);
            tokens.push(next_token);
        }
        
        // Detokenize
        self.detokenize(&output_tokens)
    }
    
    fn forward_pass(&self, tokens: &[i32]) -> MinervaResult<Array2<f32>> {
        // Simplified transformer implementation
        // This is the hard part - but doable!
    }
}
```

**Available Rust Crates for This:**
- `safetensors` (0.3) - Load HF model files (MIT license!)
- `ndarray` - Tensor operations
- `burn` - Full ML framework (if we want it)
- `tch-rs` - PyTorch bindings to libtorch (not Python!)

#### Afternoon: Decision & Plan (1 hour)

**CHOSEN: Path 3 (Hybrid) - For Fastest Results**

**Why Path 3?**
- ✅ Immediate value (use existing llama.cpp)
- ✅ Future flexibility (add pure Rust later)
- ✅ No external dependencies (pure Rust)
- ✅ Fast to implement (phased approach)
- ✅ Supports all model formats
- ✅ Users get best of both

**Timeline:**
- Days 1-2: Path 1 foundation (immediate working solution)
- Days 3-5: Path 2 enhancement (pure Rust inference)
- Days 6-7: Integration (smart backend selection)

**Result:** Complete, flexible, fast system in one week

---

### Day 2 (Continued): Quick Proof of Concept

If going with Path 1 (llama.cpp only):
- Time: 1-2 hours maximum
- Work: Update documentation to guide users
- Result: Clear path for model format support

If going with Path 3 (Hybrid):
- Day 2: Implement llama.cpp format detection
- Days 3-4: Add basic pure Rust inference scaffold
- Goal: Prove both paths work together

---

### Day 3: Testing & Documentation

#### Morning: Integration Tests (For your chosen path)

**If Path 1 (llama.cpp only) - File:** `src-tauri/tests/integration/format_detection_tests.rs` (new file, ~100 lines)

```rust
// NO PYTHON SUBPROCESS TESTS!
// Only test what Rust can do directly

use minerva_lib::inference::llama_adapter::{InferenceBackend, GenerationParams};
use std::path::Path;

#[test]
fn test_format_detection_gguf() {
    // Test: GGUF files go to llama.cpp backend
    let path = Path::new("model.gguf");
    let format = detect_format(path);
    assert_eq!(format, Format::GGUF);
}

#[test]
fn test_format_detection_safetensors() {
    // Test: Safetensors go to pure Rust backend (if available)
    let path = Path::new("model.safetensors");
    let format = detect_format(path);
    assert_eq!(format, Format::HuggingFaceSafetensors);
}

#[test]
fn test_backend_selection_based_on_format() {
    // Test: Correct backend chosen based on format
    let gguf_path = Path::new("model.gguf");
    let backend = select_backend(gguf_path).unwrap();
    // Should be LlamaCppBackend
    
    let hf_path = Path::new("model.safetensors");
    let backend = select_backend(hf_path).unwrap();
    // Should be PureRustBackend (if available)
    // or error with helpful message
}
```

**For Path 3 (Hybrid) - Additional File:** `src-tauri/src/inference/pure_rust_backend.rs` (new file, ~500-700 lines)

This is the pure Rust transformer inference implementation:

```rust
// Pure Rust transformer inference - no external dependencies!
use crate::error::{MinervaError, MinervaResult};
use crate::inference::llama_adapter::InferenceBackend;
use crate::inference::llama_tokenizer::LLaMATokenizer;
use std::collections::HashMap;
use std::path::Path;
use ndarray::Array2;

/// Pure Rust transformer-based inference
/// Supports HuggingFace safetensors format natively
pub struct PureRustBackend {
    weights: Arc<Mutex<Option<HashMap<String, Array2<f32>>>>>,
    config: Arc<Mutex<Option<ModelConfig>>>,
    tokenizer: Arc<Mutex<Option<LLaMATokenizer>>>,
    n_ctx: usize,
    n_threads: usize,
}

#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub model_type: String,  // "llama", "mistral", "phi", etc.
}

impl PureRustBackend {
    pub fn new() -> Self {
        Self {
            weights: Arc::new(Mutex::new(None)),
            config: Arc::new(Mutex::new(None)),
            tokenizer: Arc::new(Mutex::new(None)),
            n_ctx: 0,
            n_threads: num_cpus::get(),
        }
    }

    /// Load safetensors model file
    fn load_safetensors(path: &Path) -> MinervaResult<HashMap<String, Array2<f32>>> {
        // Phase 9: Use safetensors crate to load weights
        // For now: scaffold the structure
        Ok(HashMap::new())
    }

    /// Load model config from JSON
    fn load_config(path: &Path) -> MinervaResult<ModelConfig> {
        // Phase 9: Parse config.json from model directory
        // For now: scaffold
        Ok(ModelConfig {
            vocab_size: 32000,
            hidden_size: 4096,
            num_layers: 32,
            num_heads: 32,
            model_type: "llama".to_string(),
        })
    }

    /// Forward pass through transformer
    fn forward_pass(&self, tokens: &[i32]) -> MinervaResult<Vec<f32>> {
        // Phase 9: Implement actual transformer forward pass
        // For now: mock output matching vocab size
        let config = self.config.lock().unwrap();
        let cfg = config.as_ref().ok_or_else(|| 
            MinervaError::InferenceError("Model not loaded".to_string())
        )?;
        
        // Return logits over vocab
        Ok(vec![0.1; cfg.vocab_size])
    }

    /// Sample next token from logits
    fn sample_token(&self, logits: &[f32], temperature: f32) -> MinervaResult<i32> {
        // Apply temperature
        let scaled = logits.iter().map(|&x| x / temperature).collect::<Vec<_>>();
        
        // Sample argmax for now (Phase 9: use proper sampling)
        let max_idx = scaled.iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(idx, _)| idx as i32)
            .unwrap_or(0);
        
        Ok(max_idx)
    }
}

impl Default for PureRustBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl InferenceBackend for PureRustBackend {
    fn load_model(&mut self, path: &Path, n_ctx: usize) -> MinervaResult<()> {
        // Load model weights
        let weights = Self::load_safetensors(path)?;
        *self.weights.lock().unwrap() = Some(weights);
        
        // Load model config
        let config = Self::load_config(path)?;
        *self.config.lock().unwrap() = Some(config);
        
        // Create/load tokenizer
        let tokenizer = LLaMATokenizer::new(
            (0..32000).map(|i| format!("token_{}", i)).collect()
        )?;
        *self.tokenizer.lock().unwrap() = Some(tokenizer);
        
        self.n_ctx = n_ctx;
        
        tracing::info!("Pure Rust backend loaded model: {}", path.display());
        Ok(())
    }

    fn unload_model(&mut self) {
        *self.weights.lock().unwrap() = None;
        *self.config.lock().unwrap() = None;
        *self.tokenizer.lock().unwrap() = None;
        self.n_ctx = 0;
    }

    fn generate(&self, prompt: &str, params: crate::inference::llama_adapter::GenerationParams) -> MinervaResult<String> {
        let tokenizer = self.tokenizer.lock().unwrap();
        let tok = tokenizer.as_ref()
            .ok_or_else(|| MinervaError::InferenceError("No tokenizer".to_string()))?;
        
        // Tokenize input
        let tokens = tok.encode(prompt)?;
        let mut tokens: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
        
        // Generate tokens
        for _ in 0..params.max_tokens {
            let logits = self.forward_pass(&tokens)?;
            let next_token = self.sample_token(&logits, params.temperature)?;
            tokens.push(next_token);
        }
        
        // Detokenize
        let u32_tokens: Vec<u32> = tokens.iter().map(|&t| t as u32).collect();
        tok.decode(&u32_tokens)
    }

    fn tokenize(&self, text: &str) -> MinervaResult<Vec<i32>> {
        let tokenizer = self.tokenizer.lock().unwrap();
        let tok = tokenizer.as_ref()
            .ok_or_else(|| MinervaError::InferenceError("No tokenizer".to_string()))?;
        
        let tokens = tok.encode(text)?;
        Ok(tokens.iter().map(|&t| t as i32).collect())
    }

    fn detokenize(&self, tokens: &[i32]) -> MinervaResult<String> {
        let tokenizer = self.tokenizer.lock().unwrap();
        let tok = tokenizer.as_ref()
            .ok_or_else(|| MinervaError::InferenceError("No tokenizer".to_string()))?;
        
        let u32_tokens: Vec<u32> = tokens.iter().map(|&t| t as u32).collect();
        tok.decode(&u32_tokens)
    }

    fn is_loaded(&self) -> bool {
        self.weights.lock().unwrap().is_some()
    }

    fn context_size(&self) -> usize {
        self.n_ctx
    }

    fn thread_count(&self) -> usize {
        self.n_threads
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pure_rust_backend_creation() {
        let backend = PureRustBackend::new();
        assert!(!backend.is_loaded());
        assert_eq!(backend.context_size(), 0);
    }

    #[test]
    fn test_pure_rust_backend_config_loading() {
        let config = PureRustBackend::load_config(Path::new("dummy.json")).unwrap();
        assert_eq!(config.model_type, "llama");
    }

    #[test]
    fn test_pure_rust_token_sampling() {
        let backend = PureRustBackend::new();
        let logits = vec![0.1, 0.5, 0.3, 0.1];
        let token = backend.sample_token(&logits, 1.0).unwrap();
        assert!(token >= 0 && token < 4);
    }
}
```

**Integration Tests for Path 3:**

```rust
// tests/integration/format_detection_and_routing.rs

#[test]
fn test_gguf_uses_llama_cpp_backend() {
    let path = Path::new("model.gguf");
    let backend = select_backend(path).unwrap();
    // Should be LlamaCppBackend
}

#[test]
fn test_safetensors_uses_pure_rust_backend() {
    let path = Path::new("model.safetensors");
    let backend = select_backend(path).unwrap();
    // Should be PureRustBackend
}

#[test]
fn test_backend_fallback_chain() {
    let path = Path::new("model.safetensors");
    // Try native first, fallback to conversion if needed
    let backend = select_backend_with_fallback(path).unwrap();
    // Should work regardless
}
```

#### Afternoon: Update Documentation (1 hour)

**File:** `docs/PHASE_8_STEP_3b_ROADMAP.md`
- Mark "Day 1-3" sections as ✅ Complete
- Add "Day 4" section for Phase 3d planning
- Update timeline

**File:** `docs/PHASE_8_PLAN.md`
- Update progress section
- Mark Step 3b as complete
- Update timeline for remaining steps

#### Final: Run All Tests & Lint (30 minutes)

```bash
# Run all tests
pnpm test
# Expected: 830+ passing (806 existing + 24 new)

# Run lint
pnpm lint
# Expected: 0 violations

# Commit
git add -A
git commit -m "feat(phase8-step3b): Implement MLX subprocess integration

- Create MlxServerClient for HTTP communication with mlx-lm
- Implement server startup/shutdown in MlxBackend.load_model()
- Add real model inference via HTTP API
- Support process management and cleanup
- Add 24+ integration tests (with #[ignore] for optional running)
- All 830+ tests passing, 0 lint violations"
```

---

## Testing Strategy - Pure Rust Only

### Unit Tests (No external dependencies!)
- Format detection logic
- Backend selection
- Tokenization (basic)
- Error handling
- Model type identification

### All Tests Run in CI
No special setup needed:
```bash
# Just run tests normally
cargo test  # All tests run without external dependencies!

# No subprocess calls
# No Python installation needed
# No network dependencies
# Pure Rust only
```

### Performance Tests (Optional)
```rust
// Benchmark pure Rust inference vs llama.cpp
#[bench]
fn bench_pure_rust_inference(b: &mut Bencher) {
    let backend = PureRustBackend::new();
    // Load model...
    b.iter(|| backend.generate("test prompt"))
}

#[bench]
fn bench_llama_cpp_inference(b: &mut Bencher) {
    let backend = LlamaCppBackend::new();
    // Load model...
    b.iter(|| backend.generate("test prompt"))
}
```

---

## Error Cases to Handle

### 1. Unsupported Model Format
```
Error: "Model format .xyz not supported"
→ Return MinervaError::InvalidRequest
→ User sees: "Use GGUF format or convert with: [instructions]"
→ No subprocess call attempts
→ Clear guidance on what to do
```

### 2. Model File Corrupted
```
Error: "Failed to load model weights from file"
→ Return MinervaError::ModelLoadingError
→ Provide context: "File may be corrupted or incomplete"
→ Suggest re-download
```

### 3. Insufficient Memory for Model
```
Error: "Model requires 8GB but only 4GB available"
→ Return MinervaError::OutOfMemory
→ Suggest: quantize model, or use llama.cpp
→ No subprocess fallback - just fail gracefully
```

### 4. Invalid Tokenizer Configuration
```
Error: "Model config missing required fields"
→ Return MinervaError::ModelLoadingError
→ Provide helpful recovery: check model metadata
→ No subprocess workaround - stay in Rust
```

---

## Performance Targets

| Metric | Target | Notes |
|--------|--------|-------|
| Format detection | <1ms | Pure Rust, instant |
| Model load | <2s | llama.cpp or pure Rust |
| First inference | <1s | Cached model ready |
| Subsequent requests | <200ms | Direct inference, no subprocess |
| Memory overhead | 0 | No external process |
| Startup | Instant | No server to start |

---

## Checkpoints

### Day 1 Evening
- [ ] mlx-lm installed and tested
- [ ] Subprocess architecture designed
- [ ] HTTP client design finalized
- [ ] No code written yet, just planning

### Day 2 Evening
- [ ] MlxServerClient module complete (150 lines, ~8 tests)
- [ ] MlxBackend integration started (subprocess calls)
- [ ] All existing tests still passing
- [ ] Lint checks passing

### Day 3 Evening
- [ ] Full subprocess integration working
- [ ] 24+ integration tests written
- [ ] All 830+ tests passing
- [ ] 0 lint violations
- [ ] Ready for Phase 8-Step 3d

---

## Success Criteria

✅ Subprocess-based mlx-lm server integration works
✅ Real HuggingFace model inference possible
✅ 830+ tests passing (806 existing + 24 new)
✅ 0 lint violations
✅ Error handling robust (all edge cases covered)
✅ Performance meets targets
✅ Documentation complete
✅ Ready for next developer to continue Phase 8-Step 3d

---

## Next Phase (After This)

### Phase 8-Step 3d: Integration Tests & Refinement
- Comprehensive testing with real mlx-lm
- Performance benchmarking vs llama.cpp
- Stress testing (concurrent requests, large models)
- Documentation and examples

### Phase 8-Step 4: Backend Selection
- Auto-routing based on model format
- User-configurable backend preference
- Fallback chains (prefer MLX, fallback to llama.cpp)
- API parameter for backend selection

---

## References

### Pure Rust Crates for Model Loading
- `safetensors` - Load HuggingFace model weights (MIT licensed!)
- `ndarray` - Tensor operations
- `burn` - Full ML framework if needed
- `tch-rs` - LibTorch bindings (not Python!)

### llama.cpp Documentation
- GitHub: https://github.com/ggerganov/llama.cpp
- Rust bindings: `llama-cpp-rs` (already in use)
- Docs: Extensive format support documentation

### Model Format Conversion
- GGUF is the gold standard (quantized, optimized)
- Most models available in GGUF on HuggingFace
- `llama.cpp` provides conversion tools

---

## Quick Start Command

When ready to start Phase 8-Step 3b:

```bash
# Create new feature branch
git checkout -b phase-8/pure-rust-backends

# Step 1: Research llama.cpp capabilities
# - What formats does it support?
# - Can we improve llama_adapter.rs?

# Step 2: Decide Path (1, 2, or 3)
# - Path 1: Just use llama.cpp (quickest)
# - Path 2: Pure Rust inference (ambitious)
# - Path 3: Hybrid (recommended)

# Step 3: Implement chosen approach
# - NO subprocess calls
# - All code stays in Rust
# - Tests run without external dependencies

# Start by improving llama_adapter.rs...
```

---

**Status:** Ready to Start  
**Complexity:** Medium (process management + async HTTP)  
**Time Estimate:** 2-3 focused days  
**Risk Level:** Low (subprocess is battle-tested approach)  

All groundwork complete. Ready for next developer! ✅
