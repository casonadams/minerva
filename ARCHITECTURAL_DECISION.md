# Critical Architectural Decision: Pure Rust, No Python

**Date:** January 23, 2026 - End of Session  
**Status:** ✅ ADOPTED  
**Impact:** Simplifies Phase 8 significantly, improves long-term maintainability  

---

## The Decision

**All inference work MUST stay in Rust.**

We will NOT call Python subprocesses, spawn external servers, or depend on mlx-lm or any Python package.

---

## Why This Matters

### Problems with Python Subprocess Approach

❌ **Complexity**
```
User request 
  → Rust HTTP handler
  → Spawn Python subprocess
  → Python loads model
  → Python runs inference
  → Python returns JSON
  → Rust parses JSON
  → Rust returns response

Multiple serialization/deserialization steps!
```

❌ **Dependencies**
- Python installation required
- mlx-lm package required
- User must have `pip install mlx-lm` working
- Version management complexity

❌ **Performance**
- Process spawn overhead (~100-500ms)
- Network serialization overhead
- Memory: Python runtime + model = extra 600MB+

❌ **Testing Complexity**
- Tests require Python environment
- CI must have Python + mlx-lm
- More failure points
- Harder to debug

❌ **Maintainability**
- Two language codebases to understand
- Process management complexity
- Error handling across process boundary
- Hard to profile and optimize

### Benefits of Pure Rust Approach

✅ **Simplicity**
```
User request
  → Rust HTTP handler
  → Load model (pure Rust)
  → Run inference (pure Rust)
  → Return response

Direct execution, no serialization!
```

✅ **Zero External Dependencies**
- No Python needed
- No subprocess management
- No version conflicts
- Self-contained binary

✅ **Performance**
- No process spawn overhead
- Direct memory access
- Better optimization opportunities
- Smaller app size

✅ **Testing**
- Tests run instantly
- No environment setup needed
- Works in CI without special config
- All tests pass: `cargo test`

✅ **Maintainability**
- Single language codebase
- Direct control and debugging
- Easy to optimize hot paths
- Clear error handling

---

## The Three Paths

### Path 1: Use llama.cpp Only (RECOMMENDED MVP)

**Strategy:**
- llama.cpp already supports GGUF format
- GGUF is the gold standard (quantized)
- Most popular models available in GGUF
- No new code needed!

**Immediate Action:**
```rust
// In llama_adapter.rs - improve existing code
// Don't create MlxBackend yet
// Just make llama.cpp path even better

// Add better format detection
// Add clearer error messages
// Add model format conversion guidance
```

**Time:** 0-1 days (maybe just documentation)

**Result:** 
- Works for 90% of use cases
- Zero new code
- Zero new complexity

### Path 2: Pure Rust Transformer Inference (Ambitious)

**Strategy:**
- Use `safetensors` crate (MIT licensed!) to load HF model files
- Implement basic transformer forward pass in pure Rust
- Support common architectures: Llama, Mistral, Phi, etc.

**Code Structure:**
```rust
// New file: src-tauri/src/inference/pure_rust_backend.rs

pub struct PureRustBackend {
    weights: HashMap<String, ndarray::Array2<f32>>,
    config: ModelConfig,
    tokenizer: LLaMATokenizer,
}

impl InferenceBackend for PureRustBackend {
    fn load_model(&mut self, path: &Path, n_ctx: usize) -> MinervaResult<()> {
        // Load safetensors file
        // Parse model config
        // Validate architecture
    }
    
    fn generate(&self, prompt: &str, params: GenerationParams) -> MinervaResult<String> {
        // Tokenize input
        // Run transformer forward pass
        // Sample output tokens
        // Detokenize result
    }
}
```

**Challenges:**
- Transformer inference is non-trivial (~500-1000 lines)
- Need to handle multiple architectures
- Performance won't match llama.cpp
- Requires knowledge of transformer architecture

**Benefits:**
- Supports any GGUF or HuggingFace format
- 100% pure Rust
- Learn transformer internals
- Future-proof

**Time:** 3-5 days of focused work

### Path 3: Hybrid Approach (RECOMMENDED Long-term)

**Strategy:**
1. Start with Path 1 (use llama.cpp)
2. As time permits, add Path 2 (pure Rust backup)
3. Users can choose via configuration

**Code Structure:**
```rust
// In llama_adapter.rs - existing LlamaCppBackend stays
// Add: pure_rust_backend.rs - new pure Rust backend

// In server.rs or commands.rs - add backend selector
pub fn select_backend(model_path: &Path) -> Box<dyn InferenceBackend> {
    match model_format {
        Format::GGUF => Box::new(LlamaCppBackend::new()),
        Format::HuggingFaceSafetensors => {
            match config.backend_preference {
                Preference::Native => Box::new(PureRustBackend::new()),
                Preference::External => Box::new(LlamaCppBackend::new()), // convert first
            }
        }
    }
}
```

**Timeline:**
- Days 1-2: llama.cpp improvement (Path 1)
- Days 3-5: Pure Rust foundation (Path 2)
- Days 6+: Backend selection & fallback chains

**Result:**
- Immediate working solution (Path 1)
- Future flexibility (Path 2)
- No Python dependencies ever

---

## Recommendation: PATH 3 (CHOSEN) ✅

**Hybrid Approach - Fastest Results**

We want the fastest possible results, so we're doing **Path 3**:

**Immediate (Days 1-2): Path 1 Foundation**
1. Improve llama.cpp format support in existing `llama_adapter.rs`
2. Add format detection (GGUF vs HuggingFace)
3. Improve error messages
4. Add tests - all passing instantly
5. **Result:** Working solution for GGUF (90% of users)

**Then (Days 3-5): Path 2 Enhancement**
1. Add pure Rust inference via `safetensors` crate
2. Implement basic transformer forward pass
3. Support HuggingFace format models natively
4. No performance regression (Rust is fast)
5. **Result:** Support for all formats, zero Python

**Final (Days 6-7): Integration & Polish**
1. Backend selection logic (auto-choose best)
2. Fallback chains (try native first, fallback to llama.cpp)
3. Comprehensive testing
4. Performance benchmarking
5. **Result:** Fully flexible, future-proof system

**Timeline:** 1 week total for complete Path 3  
**Payoff:** Both simple (Path 1) AND flexible (Path 2) working together

---

## What We're NOT Doing

❌ Creating MlxBackend that calls Python subprocess  
❌ Running `python3 -m mlx_lm.server` from Rust  
❌ HTTP calls to external Python process  
❌ Depending on mlx-lm installation  
❌ Process management complexity  
❌ Python runtime in our binary  

These were good ideas in theory but **added unnecessary complexity**. The Rust community has excellent crates for everything we need.

---

## Crates Available for Pure Rust Path

If we choose Path 2 (Pure Rust Inference):

**Model Loading:**
- `safetensors` (0.3.1) - Load HuggingFace model files
- MIT licensed, pure Rust
- Fast and reliable

**Tensor Operations:**
- `ndarray` (already in use)
- Full tensor/matrix operations
- Supports multi-dimensional arrays

**Linear Algebra:**
- `nalgebra` - If we need advanced operations
- Or use `ndarray` which covers most cases

**Full ML Frameworks (if ambitious):**
- `burn` - Native Rust deep learning
- `tch-rs` - Rust bindings to LibTorch (not Python!)

**Current Dependencies:**
- We already have: `ndarray`, `serde`, `tokio`, etc.
- No need for Python!

---

## Testing Strategy

### All tests run without setup:
```bash
# No Python needed
# No environment variables
# No external services

cargo test

# Tests complete in seconds
# All tests pass
# No dependencies to install
```

### Unit Tests:
```rust
#[test]
fn test_format_detection() {
    let format = detect_format(Path::new("model.gguf"));
    assert_eq!(format, Format::GGUF);
}

#[test]
fn test_backend_selection() {
    let backend = select_backend(Path::new("model.gguf")).unwrap();
    // Backend created successfully
}
```

### Integration Tests (for real models):
```rust
#[test]
#[ignore] // Optional - run only if model file available
fn test_real_inference() {
    // Load real model file
    // Test actual inference
    // Benchmark performance
}
```

---

## Implementation Order

### Immediate (Today/Tomorrow) - Phase 8-Step 3b
1. [ ] Improve llama.cpp format support in `llama_adapter.rs`
2. [ ] Add format detection logic
3. [ ] Improve error messages
4. [ ] Add tests (all in Rust)
5. [ ] Document for users

**Time:** 1-2 days  
**Output:** Fully working, simple solution  
**Tests:** 806+ passing, 0 lint violations  

### Later (Next week) - Phase 8-Extended  
1. [ ] Research pure Rust transformer inference
2. [ ] Implement `PureRustBackend` (if needed)
3. [ ] Add backend selection
4. [ ] Benchmark both approaches
5. [ ] Document for users

**Time:** 3-5 days (if done)  
**Output:** Flexible, future-proof architecture  
**Tests:** 830+ passing  

### Never (Don't waste time)
- ❌ Python subprocess calls
- ❌ mlx-lm integration
- ❌ External process management
- ❌ HTTP calls between Rust and Python

---

## Architectural Principles

From this point forward:

1. **All inference code stays in Rust**
2. **No external process calls** (Python, C++, etc.)
3. **No network calls** to external services for inference
4. **All tests must run** `cargo test` with no setup
5. **Use pure Rust crates** for everything (safetensors, ndarray, etc.)

These principles ensure:
- Simple, maintainable code
- Fast testing
- Easy deployment
- Clear responsibility boundaries
- No surprise dependencies

---

## Questions This Resolves

**Q: Should we use mlx-lm via subprocess?**  
A: ❌ No. All work stays in Rust.

**Q: Should we spawn Python processes for inference?**  
A: ❌ No. Use pure Rust or proven Rust crates.

**Q: Is it OK to call Python from Rust for some parts?**  
A: ❌ No. Complete separation: either all Rust or no Python.

**Q: What about performance compared to Python?**  
A: Pure Rust will match or beat Python when using appropriate crates.

**Q: What about models that need Python?**  
A: Anything possible in Python is possible in Rust (via crates).

---

## Decision Record

**Proposed:** Python subprocess approach to call mlx-lm  
**Evaluated:** Added complexity, dependencies, testing difficulty  
**Revised to:** Pure Rust approach using existing Rust crates  
**Status:** ✅ Adopted  
**Rationale:** Simpler, faster, more maintainable, zero external dependencies  
**Documentation:** Updated Phase_8_STEP_3b_ROADMAP.md  

---

## Next Developer Notes

When starting Phase 8-Step 3b:

1. **Forget the subprocess idea** - It's not happening
2. **Focus on Rust crates** - They solve this better
3. **Start with Path 1** - Make llama.cpp path awesome
4. **Add Path 2 later** - Only if time permits
5. **All tests must pass** - `cargo test` or nothing

Key files:
- `src-tauri/src/inference/llama_adapter.rs` - Where work starts
- `src-tauri/src/inference/mod.rs` - Module structure
- `docs/PHASE_8_STEP_3b_ROADMAP.md` - Updated with this decision

---

**Decision Made:** January 23, 2026  
**Status:** ✅ FINAL  
**Next Update:** When Phase 8-Step 3b starts

This is a foundational decision that shapes all future Phase 8+ work.
