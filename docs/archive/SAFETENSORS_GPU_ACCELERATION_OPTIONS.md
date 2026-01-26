# SafeTensors GPU Acceleration Options for Minerva

**Date:** January 25, 2026  
**Status:** Technical Analysis & Recommendation  
**Impact:** Could improve SafeTensors performance from 18-20 t/s to 50-100 t/s

---

## The Insight

You're absolutely right! SafeTensors itself is just a **file format** - the actual performance depends on **how you load and execute the weights**, not the format.

Current Minerva setup:
```
SafeTensors file → pure_rust_backend (CPU-only) → 18-20 t/s
```

This is like downloading a recipe but cooking with a candle instead of an oven - the recipe format isn't the problem, the execution is!

---

## Three Execution Strategies for SafeTensors

### Option 1: Pure Rust CPU Backend (Current)

**Architecture:**
```
SafeTensors File
    ↓ (read with safetensors crate)
    ↓
Rust Matrix Operations
    ↓ (CPU linear algebra)
    ↓
Output
```

**Performance:**
- **Throughput:** 10-20 t/s
- **TTFT:** 80-100ms
- **TpT:** 40-50ms/token
- **Memory:** Efficient (only active tensors in RAM)

**Pros:**
- ✅ No external dependencies
- ✅ Pure Rust (memory safe)
- ✅ Direct control over execution
- ✅ Good for edge devices (embedded systems)

**Cons:**
- ❌ CPU-only (limited parallelism)
- ❌ SIMD limited to 8-16 lanes (vs GPU 1000+ lanes)
- ❌ Memory bandwidth: 100 GB/s (vs GPU 1+ TB/s)
- ❌ 3-5x slower than GPU solutions
- ❌ Current implementation uses **synthetic computation** (sin-based fake weights!)

**Implementation Cost:** 2-3 hours to fix synthetic computation + add SIMD

**When to Use:**
- Edge devices with no GPU
- CPU-only servers
- Embedded systems
- Privacy-critical applications (all compute local)

---

### Option 2: GPU Backend for SafeTensors (Recommended) ⭐

**Architecture:**
```
SafeTensors File
    ↓ (read with safetensors crate)
    ↓
GPU Framework (burn-rs or ONNX Runtime)
    ↓ (GPU compute via CUDA/Metal/DirectML)
    ↓
Output
```

**Sub-Option 2a: Using burn-rs**
- **Framework:** burn-rs (Rust GPU framework)
- **Supported:** CUDA (NVIDIA), Metal (Apple Silicon), Compute (CPU fallback)
- **Performance:** 50-100 t/s
- **Code Example:**
```rust
use burn::backend::Wgpu;
use burn::tensor::Tensor;

let backend = Wgpu::new();
let weights = load_safetensors(path)?;
let embeddings = Tensor::from_data(weights["embedding.weight"], backend);
let output = forward_pass(embeddings, backend)?;
```

**Sub-Option 2b: Using ONNX Runtime**
- **Framework:** onnx-runtime (widely used)
- **Supported:** CUDA, TensorRT, Metal, DirectML
- **Performance:** 50-100 t/s
- **Benefit:** ONNX is industry standard
```rust
use ort::{Session, Value};

let session = Session::builder()?
    .with_optimization_level(ort::GraphOptimizationLevel::All)?
    .commit_from_file("model.onnx")?;

let output = session.run(vec![Value::from_array_onnx(&input)?])?;
```

**Pros:**
- ✅ GPU acceleration (50-100 t/s)
- ✅ Supports multiple backends (CUDA, Metal, DirectML)
- ✅ Cross-platform (Windows, macOS, Linux)
- ✅ Same performance as GGUF but with full precision
- ✅ Can load SafeTensors directly (no conversion needed)
- ✅ Industry-standard approach

**Cons:**
- ❌ Additional dependency (burn-rs or ort)
- ❌ GPU required for good performance
- ❌ More complex code
- ❌ 2-3 days to implement

**Implementation Cost:** 2-3 days (learning curve + integration)

**When to Use:**
- Production servers with GPU
- High-throughput requirements
- Cloud deployment
- Competitive benchmarking

**Recommended Crates:**
```toml
# Option A: burn-rs (modern, Rust-native)
burn = { version = "0.13", features = ["backend-wgpu", "backend-cuda"] }

# Option B: ONNX Runtime (industry standard)
ort = "2.0"
```

---

### Option 3: Convert SafeTensors to GGUF at Download (Quickest)

**Architecture:**
```
SafeTensors File
    ↓ (convert with llama.cpp tools)
    ↓
GGUF File (quantized)
    ↓ (load with existing llama_cpp_backend)
    ↓
GPU Execution
    ↓
Output
```

**Process:**
1. Download SafeTensors from HuggingFace
2. Convert using: `python -m llama_cpp.converter --safetensors model.safetensors -o model.gguf`
3. Load with existing GGUF backend (GPU-optimized)

**Performance:**
- **Throughput:** 30-100 t/s (GPU accelerated)
- **TTFT:** 40-60ms
- **TpT:** 20-30ms/token
- **Quantization:** 4-bit (1/8th original size)

**Pros:**
- ✅ Leverages existing working GGUF pipeline
- ✅ Quick implementation (30 minutes)
- ✅ GPU acceleration immediately
- ✅ Smaller model size (4-bit quantization)

**Cons:**
- ❌ Requires Python environment (during download)
- ❌ Need to store/distribute both formats
- ❌ Some precision loss from quantization (usually acceptable)
- ❌ One-time conversion overhead

**Implementation Cost:** 30 minutes to add conversion to download script

**When to Use:**
- Quick wins while developing GPU backend
- When quantization is acceptable (usually is)
- Maximizing compatibility

---

## Comparative Analysis

### Performance Comparison

| Metric | Pure Rust | GPU (burn/ort) | GGUF Conv. | Mistral-7B Comparison |
|--------|-----------|----------------|------------|----------------------|
| **Throughput** | 15-25 t/s | 50-100 t/s | 30-100 t/s | GGUF fastest |
| **TTFT** | 80-100ms | 40-60ms | 40-60ms | GPU has edge |
| **TpT** | 40-50ms | 15-25ms | 20-30ms | Quantization saves |
| **Memory** | 2.0GB | 2.0GB | 638MB | 4-bit wins |
| **Latency** | High | Low | Medium | GGUF wins |

### Implementation Cost Comparison

| Option | Time | Complexity | Benefit |
|--------|------|-----------|---------|
| **Fix Pure Rust (A)** | 2-3 hrs | Easy | 15-25 t/s |
| **Add GPU Backend (B)** | 2-3 days | Hard | 50-100 t/s ⭐ |
| **Convert GGUF (C)** | 30 min | Easy | 30-100 t/s |

### Maintenance & Support

| Option | Maintenance | Community | Long-term |
|--------|-------------|-----------|-----------|
| Pure Rust | Custom code | Small | Carry forward |
| GPU (burn) | Library updates | Growing | Future-proof |
| GPU (ort) | Library updates | Large (Microsoft) | Industry std |
| GGUF Conv. | llama.cpp | Huge | Very stable |

---

## Recommended Strategy for Minerva

### Phase 4A (Immediate - 2-3 Days)

**Quick Win: Add GGUF Conversion**
```bash
# Update download-mistral.rs to convert SafeTensors → GGUF
# Cost: 30 minutes
# Benefit: 50-100 t/s for SafeTensors users

download SafeTensors
  → check if llama-cpp-converter available
  → convert to GGUF
  → load with existing GPU backend
```

### Phase 4B (Short Term - 2-3 Days)

**Add GPU Backend for SafeTensors (burn-rs)**
```rust
// New module: src/inference/gpu_safetensors_backend.rs

pub struct GPUSafeTensorsBackend {
    device: burn::backend::gpu::Device,
    model: TransformerModel,
}

impl InferenceBackend for GPUSafeTensorsBackend {
    fn generate(&self, prompt: &str) -> Result<String> {
        let input = self.tokenize(prompt)?;
        let output = self.forward(input)?;  // GPU-accelerated
        Ok(self.decode(output)?)
    }
}
```

**Benefits:**
- ✅ Direct GPU execution (no conversion)
- ✅ Full precision retained (no quantization)
- ✅ Supports CUDA, Metal, DirectML
- ✅ Native Rust (no Python dependency)

### Phase 4C (Later - Optional)

**Optimize Pure Rust Fallback**
- Fix synthetic computation
- Add SIMD vectorization
- Multi-threading support
- Keep as CPU-only fallback

---

## My Recommendation

### Short-term (This Week): Do Options C + A

**Step 1: Convert SafeTensors to GGUF at download (30 min)**
```bash
# Modify download-mistral.rs
# After downloading SafeTensors, check for converter
# If available: convert to GGUF
# Result: Users get 50-100 t/s automatically
```

**Step 2: Fix Pure Rust backend (2-3 hours)**
```rust
// Replace synthetic computation with real weights
// Add basic SIMD
// Result: 15-25 t/s for edge devices
```

**Time:** ~4 hours total  
**Benefit:** SafeTensors now runs at GPU speeds (via GGUF) + Pure Rust fixed

---

### Medium-term (Next Week): Add GPU Backend

**Add burn-rs GPU backend (2-3 days)**
```rust
// New: src/inference/gpu_safetensors_backend.rs
// Direct SafeTensors → GPU execution
// Result: 50-100 t/s without conversion overhead
```

**Benefits:**
- Full precision (no quantization)
- Native GPU execution
- Cross-platform (CUDA, Metal, DirectML)
- Future-proof

---

## Architecture After Recommendations

### Current (Broken)
```
SafeTensors
    ↓
pure_rust_backend (synthetic computation)
    ↓
18-20 t/s (fake)
```

### After Phase 4A (Quick Fix)
```
SafeTensors
    ↓
[Check for converter]
    ↓ Yes
GGUF (quantized)
    ↓
llama_cpp_backend (GPU)
    ↓
50-100 t/s (real, GPU-accelerated)
```

### After Phase 4B (Optimal)
```
GGUF
    ↓
llama_cpp_backend (GPU) ← 30-100 t/s, quantized
    ↓
unified_backend
    ↓
SafeTensors
    ↓
gpu_safetensors_backend (burn-rs) ← 50-100 t/s, full precision
    ↓
[Fallback if no GPU]
    ↓
pure_rust_backend (SIMD-optimized) ← 15-25 t/s, CPU-only
```

---

## Decision Matrix

**Choose Option C (GGUF Conversion) if:**
- ✅ Want fastest implementation (30 min)
- ✅ Quantization acceptable (usually is)
- ✅ Want to leverage existing working code
- ✅ Time-constrained this week

**Choose Option B (GPU Backend) if:**
- ✅ Want to support SafeTensors directly
- ✅ Want full precision (no quantization)
- ✅ Have time for proper implementation (2-3 days)
- ✅ Building for long-term maintenance

**Choose Option A (Pure Rust Optimization) if:**
- ✅ Need CPU-only fallback
- ✅ Edge device support required
- ✅ Privacy-critical (no GPU upload)
- ✅ Have 2-3 hours available

**My Vote:** Do **C first (quick win), then B (proper solution)** over 1-2 weeks

This gives you:
1. **Immediate:** 50-100 t/s SafeTensors performance via GGUF conversion
2. **Soon:** 50-100 t/s SafeTensors native GPU execution via burn-rs
3. **Fallback:** 15-25 t/s Pure Rust for edge devices

---

## Technical References

### burn-rs Example
```rust
use burn::backend::Cuda;
use burn::tensor::Tensor;

#[derive(Module)]
pub struct Transformer {
    embedding: Embedding<2, f32>,
    attention: MultiHeadAttention,
    mlp: FeedForward,
}

impl Transformer {
    pub fn forward(&self, x: Tensor<Cuda, 2>) -> Tensor<Cuda, 2> {
        let embed = self.embedding.forward(x);
        let attn = self.attention.forward(embed);
        self.mlp.forward(attn)
    }
}
```

### GGUF Conversion Command
```bash
# Standard llama.cpp conversion
python -m llama_cpp.converter \
    --safetensors model.safetensors \
    --outfile model.gguf \
    --outtype q4_k_m
```

---

## Next Steps

1. **Decide:** Which option for Phase 4?
2. **Implement:** Start with chosen approach
3. **Benchmark:** Compare with current (fake) results
4. **Document:** Update performance analysis

---

## Conclusion

Yes, you're absolutely right! **SafeTensors can absolutely leverage GPU** - the current pure Rust backend is just not doing it. 

The fix is straightforward:
1. **Quick (30 min):** Convert to GGUF and use existing GPU backend
2. **Proper (2-3 days):** Add burn-rs GPU backend for native SafeTensors execution
3. **Fallback (2-3 hrs):** Fix pure Rust with SIMD for edge devices

**Recommended:** Do conversion (30 min) this week, then add GPU backend (2-3 days) next week.
