# Multi-Format GPU Backend Benchmarking Plan

**Goal:** Benchmark GPT-OSS 20B across 3 weight formats and measure throughput differences

**Status:** Day 1 - Infrastructure Setup

---

## Models & Formats

### Target Model: GPT-OSS 20B
- Architecture: LLaMA-based (20B parameters)
- Location: Downloaded via llama-server to ~/Library/Caches/llama.cpp/
- Use case: Larger model to show scaling differences across formats

### Format Comparison

| Format | Location | Status | Size | Precision | Use Case |
|--------|----------|--------|------|-----------|----------|
| **GGUF** | ~/Library/Caches/llama.cpp/ | Downloading | ~13GB (Q4) | 4-bit quantized | Production, offline |
| **SafeTensors** | ~/.lmstudio/models/ | Ready | ~40GB | Mixed (MXFP4/Q8) | GPU native, batching |
| **MLX** | ~/.lmstudio/models/ | Ready | ~40GB (sharded) | Mixed (MXFP4/Q8) | Apple Silicon optimized |

---

## Implementation Plan

### Phase 1: Format Loaders (Today)
Create modular loaders for each format:

```
src/inference/gpu/
├── loaders/
│   ├── mod.rs                 # Format loader registry
│   ├── safetensors_loader.rs  # DONE - SafeTensors
│   ├── gguf_loader.rs         # TODO - GGUF support
│   └── mlx_loader.rs          # TODO - MLX support
├── models/
│   ├── tinyllama.rs           # TinyLlama-1.1B metadata
│   ├── gpt_oss_20b.rs         # GPT-OSS 20B metadata
│   └── model_registry.rs      # Model configuration registry
└── bench/
    ├── multi_format.rs        # Unified benchmark runner
    └── formatter.rs           # Result formatting
```

### Phase 2: GGUF Loader (Next)
Implement GGUF format support:
- Use `ggml-rs` or `llama-cpp-rs` crate
- Load quantized weights
- Handle 4-bit/8-bit quantization
- Memory-efficient loading (streaming)

### Phase 3: MLX Loader (Next)
Implement MLX format support:
- Parse MLX safetensors with metadata
- Handle Apple Silicon optimizations
- Load sharded models (MLX uses model-00001-of-00003.safetensors)
- Metal GPU acceleration hints

### Phase 4: Unified Benchmark (Next)
Create benchmark runner:
```rust
pub trait FormatLoader {
    fn load_model(&self, path: &Path) -> Result<ModelWeights>;
    fn get_format_name(&self) -> &str;
    fn supports_streaming(&self) -> bool;
}

pub fn benchmark_format(
    format: &dyn FormatLoader,
    model_size: usize,
    num_iterations: usize,
) -> BenchmarkResult {
    // Load time
    // Forward pass time
    // Memory usage
    // Throughput (tokens/sec)
}
```

---

## Expected Results

### Benchmark Matrix: GPT-OSS 20B

```
Format          | Load Time | TTFT  | TpT    | Throughput | Memory
----------------|-----------|-------|--------|------------|--------
GGUF (Q4)       | 30-60s    | 100ms | 15ms   | 65 t/s     | 12GB
SafeTensors     | 60-90s    | 80ms  | 10ms   | 100 t/s    | 40GB
MLX (optimized) | 45-75s    | 70ms  | 8ms    | 125 t/s    | 35GB
```

**Key Insights:**
- GGUF: Fastest load, lowest memory, slowest inference (quantization overhead)
- SafeTensors: Standard format, GPU-friendly, moderate performance
- MLX: Best throughput on Apple Silicon (Metal), optimized kernels

---

## Files to Create

### 1. GGUF Loader Module
**File:** `src/inference/gpu/loaders/gguf_loader.rs`
- Implement GGUF deserialization
- Handle quantization (Q4_K_M, Q8, etc.)
- Create dequantization kernels
- Match SafeTensors interface for consistency

### 2. MLX Loader Module
**File:** `src/inference/gpu/loaders/mlx_loader.rs`
- Parse MLX-specific metadata
- Handle sharded models
- Merge shards during loading
- Optimize for Metal GPU

### 3. Model Registry
**File:** `src/inference/gpu/models/gpt_oss_20b.rs`
```rust
pub struct GPTOss20BConfig {
    hidden_size: usize,
    num_attention_heads: usize,
    num_kv_heads: usize,
    num_layers: usize,
    vocab_size: usize,
    // ... other config
}
```

### 4. Unified Benchmark
**File:** `src/bin/multi-format-benchmark.rs`
```
$ cargo run --release --bin multi-format-benchmark -- --model gpt-oss-20b --formats gguf,safetensors,mlx
```

---

## Progress Tracking

- [x] Day 1a: SafeTensors loader (TinyLlama working)
- [ ] Day 1b: GGUF loader (when file ready)
- [ ] Day 1c: MLX loader (MLX models ready)
- [ ] Day 1d: Unified benchmark (compare all 3)
- [ ] Day 2: GQA attention + forward pass
- [ ] Day 3-7: Optimization (Flash Attention, INT8, batching, etc.)

---

## Critical Path

1. **GPT-OSS 20B GGUF download** → Ready when llama-server finishes
2. **GGUF loader implementation** → 2-3 hours (blocking for comparison)
3. **MLX loader implementation** → 2-3 hours (MLX models already ready)
4. **Unified benchmark** → 1 hour (compare all 3)
5. **Results analysis** → 1-2 hours (understand trade-offs)

**Estimated total:** 6-9 hours for full multi-format comparison

---

## Next Actions

1. **User:** Download GPT-OSS 20B GGUF via llama-server
   - Destination: `~/Library/Caches/llama.cpp/gpt-oss-20b.gguf`
   - Size: ~13GB (Q4_K_M quantization expected)
   
2. **Me:** Create GGUF loader when file is ready
   - Parse GGUF format
   - Handle quantization
   - Match SafeTensors interface

3. **Me:** Create MLX loader
   - Handle sharded format
   - Metal optimization hints

4. **Me:** Create unified benchmark
   - Load all 3 formats
   - Run identical forward passes
   - Compare results

5. **Us:** Analyze results
   - Identify best format for production
   - Choose optimization strategy
   - Plan Days 2-7 accordingly

---

## Technical Notes

### GGUF Format (llama.cpp)
- Binary format optimized for inference
- Quantization-aware (Q4, Q5, Q8)
- Memory-mapped loading possible
- No gradient computation needed

### SafeTensors Format
- Simple binary format with metadata header
- Language-agnostic (Python, Rust, JS, etc.)
- Supports all precision levels
- Perfect for distributed/sharded models

### MLX Format
- Apple-specific variant of SafeTensors
- Optimized for Metal GPU
- Sharded into multiple files
- Contains Metal kernel hints

---

## Dependencies

Add to `Cargo.toml`:
```toml
[dependencies]
ggml = "0.1"           # For GGUF support
safetensors = "0.4"    # Already have
mlx-rs = "0.1"         # For MLX metadata (if available)
```

---

## Success Criteria

- [x] Load TinyLlama-1.1B SafeTensors (81.9s)
- [ ] Load GPT-OSS 20B GGUF (< 120s)
- [ ] Load GPT-OSS 20B SafeTensors (< 120s)
- [ ] Load GPT-OSS 20B MLX (< 120s)
- [ ] Unified benchmark showing all 3 side-by-side
- [ ] Analysis report on format trade-offs
- [ ] Decision on preferred format for optimization

---

**Created:** Jan 25, 2026  
**Updated:** When user provides GGUF file path
