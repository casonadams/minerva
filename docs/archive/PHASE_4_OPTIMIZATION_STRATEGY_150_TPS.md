# Phase 4 Optimization Strategy: Achieving 150+ t/s Performance

**Date:** January 25, 2026  
**Goal:** Implement GPU backend AND achieve 150+ tokens/sec throughput  
**Timeline:** 3-4 days (extended from 2-3 days for optimization)  
**Strategy:** Aggressive optimization across all layers

---

## The Path to 150+ t/s

### Current State
```
TinyLlama-1.1B (1.1B parameters):
  GGUF (baseline):         30.4 t/s
  SafeTensors (CPU):       18.8 t/s (fake)
  SafeTensors (GPU, naive): 50-100 t/s

Mistral-7B (7B parameters):
  GGUF (projected):        21-30 t/s
  SafeTensors GPU (naive):  40-80 t/s
```

### Target State
```
TinyLlama-1.1B:
  GGUF (optimized):        80-120 t/s  (+2.6-4x)
  SafeTensors GPU (opt):   100-150 t/s (+2-3x naive)
  
Mistral-7B:
  GGUF (optimized):        50-80 t/s   (+2-3x naive)
  SafeTensors GPU (opt):   100-150 t/s (same throughput despite 6x model size!)
```

### The Insight
**Throughput ≠ single-token latency**. You can achieve 150+ t/s even on large models through:
1. Batch processing (process multiple requests together)
2. KV cache (avoid recomputing past tokens)
3. Operator fusion (combine operations)
4. Quantization (reduce memory bandwidth bottleneck)
5. Flash Attention (reduce compute)

---

## Three Optimization Layers

### Layer 1: GPU Backend Implementation (Phase 4, Days 1-2)
**Goal:** Get 50-100 t/s working correctly  
**Tools:** burn-rs with CUDA/Metal  
**What:** Proper GPU acceleration for SafeTensors

### Layer 2: Deep Optimization (Phase 4, Days 2-3)
**Goal:** Increase to 100-150 t/s  
**Techniques:** KV cache, Flash Attention, operator fusion  
**Focus:** Peak performance on single request

### Layer 3: Batch Processing (Phase 4, Day 3-4)
**Goal:** Sustain 150+ t/s under realistic load  
**Techniques:** Request batching, pipelining, async processing  
**Focus:** Throughput under multiple concurrent requests

---

## Layer 1: GPU Backend (Days 1-2)

### 1.1 Base Implementation (Use Phase 4 Plan)
```rust
pub struct GPUSafeTensorsBackend {
    device: B::Device,
    model: TransformerModel,
    weights: TransformerWeights<B>,
}

impl InferenceBackend for GPUSafeTensorsBackend {
    fn generate(&self, prompt: &str, params: GenerationParams) -> MinervaResult<String> {
        // Standard forward pass
    }
}
```

**Timeline:** 6-8 hours (Day 1)  
**Expected Performance:** 50-100 t/s

### 1.2 Testing & Verification
```bash
# Compile
cargo build --release --bin gpu-safetensors

# Test with TinyLlama
cargo test --release gpu_safetensors

# Benchmark
./real-benchmark --format safetensors-gpu --runs 5
```

**Timeline:** 2 hours (Day 1-2)  
**Expected Performance:** Confirmed 50-100 t/s

---

## Layer 2: Deep Optimization (Days 2-3)

### 2.1 KV Cache Implementation

**What:** Cache key-value pairs from previous forward passes

**Impact:** Reduces computation from O(n²) to O(n) for generation phase

```rust
pub struct KVCache<B: Backend> {
    k: Tensor<B, 3>,  // (batch, seq_len, hidden)
    v: Tensor<B, 3>,
    seq_len: usize,
}

impl<B: Backend> KVCache<B> {
    pub fn append(&mut self, new_k: Tensor<B, 3>, new_v: Tensor<B, 3>) {
        // Instead of:
        //   attention(Q, K, V)  // K,V computed from scratch
        // 
        // Do:
        //   attention(Q, [cached_K, new_K], [cached_V, new_V])
        
        self.k = Tensor::cat([self.k.clone(), new_k], 1);
        self.v = Tensor::cat([self.v.clone(), new_v], 1);
        self.seq_len += 1;
    }
    
    pub fn get_kv(&self) -> (Tensor<B, 3>, Tensor<B, 3>) {
        (self.k.clone(), self.v.clone())
    }
}
```

**Performance Impact:**
```
Without KV Cache:
  Forward pass for token 100: Compute K,V for all 100 tokens
  Complexity: O(100²) = 10,000 operations
  
With KV Cache:
  Forward pass for token 100: Compute K,V for only new token
  Complexity: O(100) = 100 operations
  Speedup: 100x on attention computation!
```

**Implementation Time:** 2-3 hours  
**Performance Boost:** 1.5-2x speedup on generation

### 2.2 Flash Attention

**What:** Reduce memory accesses in attention computation (I/O aware algorithm)

**Reference:** [Flash Attention Paper](https://arxiv.org/abs/2205.14135)

```rust
/// Standard attention: O(n²) memory access
pub fn attention_naive<B: Backend>(
    q: Tensor<B, 3>,
    k: Tensor<B, 3>,
    v: Tensor<B, 3>,
) -> Tensor<B, 3> {
    let scores = q.matmul(&k.transpose(-2, -1));  // (batch, n_heads, seq_len, seq_len)
    let scores = scores / (head_dim as f32).sqrt();
    let attn = scores.softmax(-1);
    let output = attn.matmul(&v);
    output
}

/// Flash Attention: O(n) memory access (tiling + fused computation)
pub fn attention_flash<B: Backend>(
    q: Tensor<B, 3>,
    k: Tensor<B, 3>,
    v: Tensor<B, 3>,
) -> Tensor<B, 3> {
    // Key insight: Process in tiles to keep intermediate results in fast memory
    
    let block_size = 64;  // Tune for GPU cache
    let num_blocks = (seq_len + block_size - 1) / block_size;
    
    let mut output = Tensor::zeros([batch, n_heads, seq_len, head_dim]);
    
    for block_idx in 0..num_blocks {
        let start = block_idx * block_size;
        let end = (start + block_size).min(seq_len);
        
        let q_block = q.index_select(2, start..end);  // (batch, heads, block_size, head_dim)
        let k_block = k.index_select(2, 0..=end);     // (batch, heads, end, head_dim)
        let v_block = v.index_select(2, 0..=end);
        
        // Compute attention for this block (stays in fast memory)
        let scores = q_block.matmul(&k_block.transpose(-2, -1));
        let scores = scores / (head_dim as f32).sqrt();
        let attn = scores.softmax(-1);
        let block_output = attn.matmul(&v_block);
        
        // Write output
        output.index_set(2, start..end, block_output);
    }
    
    output
}
```

**Performance Impact:**
```
Memory Bandwidth Limited System:
  Naive Attention: 
    Compute: 2n²hd operations
    Memory: n²hd reads/writes
    Ratio: 2 ops per byte (memory bottleneck!)
    
  Flash Attention:
    Compute: 2n²hd operations (same)
    Memory: nhd + n²/B reads/writes (B = block size)
    Ratio: 2B ops per byte (100x more efficient!)
```

**Implementation Time:** 3-4 hours  
**Performance Boost:** 1.5-2x speedup on attention (30-40% of forward pass)

### 2.3 Operator Fusion

**What:** Combine multiple GPU operations into one to reduce memory movement

**Example:** Instead of separate RMSNorm + Matmul, fuse into one kernel

```rust
// Before (naive): 2 GPU kernels
let hidden = hidden.clone();
let normalized = rms_norm(&hidden);  // Kernel 1: writes to memory
let output = matmul(&normalized);     // Kernel 2: reads from memory

// After (fused): 1 GPU kernel
let output = rms_norm_then_matmul(&hidden);  // Kernel 1: keep in registers
```

**Operations to Fuse:**
1. RMSNorm + Linear projection
2. Attention + Output projection
3. SiLU + Linear (gate activation)
4. Layer norm + attention

**Performance Impact:**
```
Memory Bandwidth: ~1TB/s
Computation: ~100 TFLOPS

Without fusion: 10 memory trips = 10TB/s bandwidth (bottleneck!)
With fusion: 1 memory trip = 100GB/s (now compute-bound, much better!)
```

**Implementation Time:** 2-3 hours (use burn-rs's fusion capabilities)  
**Performance Boost:** 1.2-1.5x speedup on forward pass

---

## Layer 3: Batch Processing (Day 3-4)

### 3.1 Request Batching

**What:** Process multiple requests together (amortize compute overhead)

```rust
pub struct BatchedInference<B: Backend> {
    max_batch_size: usize,
    queue: Arc<Mutex<Vec<InferenceRequest>>>,
}

impl<B: Backend> BatchedInference<B> {
    pub async fn generate_batch(
        &self,
        requests: Vec<InferenceRequest>,
    ) -> Result<Vec<String>> {
        // Process multiple requests in parallel on GPU
        
        let batch_size = requests.len();
        let max_seq_len = requests.iter().map(|r| r.tokens.len()).max().unwrap();
        
        // Pad sequences to same length
        let tokens = requests.iter()
            .map(|r| pad_sequence(&r.tokens, max_seq_len))
            .collect::<Vec<_>>();
        
        // Stack into batch tensor
        let batch_tokens = Tensor::stack(tokens);  // (batch, seq_len)
        
        // Single forward pass for all requests
        let batch_embeddings = self.embed(batch_tokens);
        let batch_output = self.forward(batch_embeddings);
        
        // Split results
        let results = batch_output.unbind(0);  // [(seq_len, hidden)]
        
        Ok(requests.into_iter().zip(results)
            .map(|(req, output)| decode(output))
            .collect())
    }
}
```

**Performance Impact:**
```
Request 1: Process alone
  GPU utilization: 20% (waiting for compute)
  Throughput: 30 t/s
  
Batch 8 requests together:
  GPU utilization: 90% (full parallelism)
  Throughput: 30 * 8 * 0.85 = 204 t/s (overhead: ~15%)
```

**Why Batching Works:**
- GPU has 1000+ cores
- Single request can't use them all
- Batch of 8+ requests → full GPU utilization

**Implementation Time:** 3-4 hours  
**Performance Boost:** 5-8x if batching 8+ requests

### 3.2 Pipelined Generation

**What:** While generating next token, prefetch future tokens in parallel

```rust
pub async fn generate_pipelined(
    &self,
    prompt: &str,
    n_tokens: usize,
) -> String {
    let mut output = String::new();
    let mut kv_cache = KVCache::new();
    
    let tokens = self.tokenize(prompt)?;
    let mut input = tokens.clone();
    
    for _ in 0..n_tokens {
        // Stage 1: Forward pass (on GPU)
        let task1 = tokio::spawn({
            let backend = self.clone();
            let input = input.clone();
            async move {
                backend.forward_with_cache(&input, &mut kv_cache).await
            }
        });
        
        // Stage 2: Sampling (on CPU, in parallel)
        // While GPU is computing, sample on CPU for next token
        let logits = task1.await??;
        let next_token = self.sample(&logits)?;
        
        output.push_str(&self.decode(&[next_token])?);
        input = vec![next_token];
    }
    
    Ok(output)
}
```

**Performance Impact:**
```
Sequential (current):
  GPU (1ms) → Softmax+Sample (0.1ms) → GPU (1ms) → ...
  Total per token: 1.1ms = 909 t/s potential

Pipelined:
  GPU (1ms) ────┐
                ├─ Softmax+Sample (0.1ms, overlapped)
  GPU (1ms) ────┘
  Total per token: ~1ms = 1000 t/s potential
  
Speedup: ~10% latency improvement
```

**Implementation Time:** 1-2 hours  
**Performance Boost:** 1.05-1.1x latency improvement

---

## Layer 4: Quantization (Optional, for Ultra High Performance)

### 4.1 INT8 Quantization

**What:** Reduce precision to 8-bit integers (4x smaller, 2x faster)

```rust
/// Quantize weights to INT8
pub fn quantize_weights(weights: &Tensor) -> (Tensor, Tensor) {
    // Find min/max
    let max_val = weights.abs().max().item();
    let min_val = weights.abs().min().item();
    
    // Scale to [-128, 127]
    let scale = 127.0 / max_val;
    let quantized = (weights * scale).cast::<i8>();
    
    (quantized, Tensor::from([scale]))
}

/// Dequantize during forward pass
pub fn dequantize_matmul(
    input: Tensor,
    weights_int8: Tensor,
    scale: f32,
) -> Tensor {
    let weights = weights_int8.cast::<f32>() / scale;
    input.matmul(&weights)
}
```

**Performance Impact:**
```
FP32 weights (4 bytes each):
  Mistral-7B: 7B * 4 = 28GB weights
  Memory bandwidth required: 28GB / latency
  
INT8 weights (1 byte each):
  Mistral-7B: 7B * 1 = 7GB weights
  Memory bandwidth required: 7GB / latency
  4x reduction in memory traffic!

Expected speedup: 1.5-2x (accounting for dequantization cost)
```

**Tradeoff:** Slight accuracy loss (usually <1% for transformers)

**Implementation Time:** 2-3 hours  
**Performance Boost:** 1.5-2x faster  
**Accuracy Impact:** <1% loss

---

## Implementation Roadmap (4-Day Schedule)

### Day 1: GPU Backend Core (8 hours)
- [ ] 0:00-1:00  Setup burn-rs dependencies
- [ ] 1:00-3:00  Weight loader from SafeTensors
- [ ] 3:00-5:00  GPU transformer layers (attention, FFN)
- [ ] 5:00-7:00  Main backend implementation
- [ ] 7:00-8:00  Basic testing
- **Target:** 50-100 t/s (naive)

### Day 2: Optimization Part 1 (8 hours)
- [ ] 0:00-2:00  KV Cache implementation + testing
- [ ] 2:00-4:00  Flash Attention implementation
- [ ] 4:00-6:00  Operator fusion setup
- [ ] 6:00-8:00  Benchmark and validate
- **Target:** 100-130 t/s

### Day 3: Optimization Part 2 (8 hours)
- [ ] 0:00-2:00  Request batching infrastructure
- [ ] 2:00-3:00  Pipelined generation
- [ ] 3:00-5:00  INT8 quantization (optional)
- [ ] 5:00-7:00  Comprehensive benchmarking
- [ ] 7:00-8:00  Documentation and tuning
- **Target:** 130-150+ t/s

### Day 4: Validation & Scaling (4-6 hours)
- [ ] 0:00-1:00  Test on Mistral-7B
- [ ] 1:00-2:00  Performance profiling
- [ ] 2:00-3:00  Edge case testing
- [ ] 3:00-4:00  Documentation and commit
- **Target:** Sustained 150+ t/s

---

## Expected Performance Growth

```
Day 1:  Naive GPU Backend
        TinyLlama-1.1B: 50-100 t/s
        
Day 2:  + KV Cache + Flash Attention
        TinyLlama-1.1B: 100-130 t/s (+30%)
        
Day 3:  + Batching + Quantization
        TinyLlama-1.1B: 130-150+ t/s (+20%)
        
Day 4:  Validation & Scaling
        Mistral-7B:     100-150+ t/s (same throughput!)
        Multiple backends: 150+ t/s sustained
```

---

## Critical Code Locations

### KV Cache (High Priority)
```rust
// File: src/inference/gpu_models/kv_cache.rs
pub struct KVCache<B: Backend> {
    pub k: Tensor<B, 3>,
    pub v: Tensor<B, 3>,
    pub position: usize,
}

impl<B: Backend> KVCache<B> {
    pub fn update(&mut self, k: Tensor<B, 3>, v: Tensor<B, 3>) {
        // Append new tokens to cache
    }
}
```

### Flash Attention (High Priority)
```rust
// File: src/inference/gpu_models/flash_attention.rs
pub fn flash_attention<B: Backend>(
    q: Tensor<B, 3>,
    k: Tensor<B, 3>,
    v: Tensor<B, 3>,
) -> Tensor<B, 3> {
    // Tiled attention computation
}
```

### Request Batching (Medium Priority)
```rust
// File: src/inference/batched_inference.rs
pub struct BatchedBackend<B: Backend> {
    queue: Arc<Mutex<Vec<Request>>>,
    batch_size: usize,
}
```

---

## Performance Monitoring

### Metrics to Track
```bash
# Throughput (tokens/sec)
./real-benchmark --format safetensors-gpu --runs 10

# Latency (TTFT, TpT)
./real-benchmark --verbose --format safetensors-gpu

# GPU Utilization
nvidia-smi dmon  # NVIDIA
mtl_utilization # Apple

# Memory Usage
nvidia-smi --query-gpu=memory.used,memory.total
```

### Profiling Commands
```bash
# PyTorch profiler equivalent
cargo bench --bench gpu_inference

# Flamegraph (CPU bottlenecks)
cargo flamegraph --bin real-benchmark

# GPU profiling
nsys profile ./target/release/real-benchmark
```

---

## Optimization Priority (Do in This Order)

### Tier 1: Essential (Must Do)
1. **KV Cache** - 1.5-2x speedup, critical for generation
2. **Flash Attention** - 1.5-2x speedup on attention (30% of compute)
3. **Operator Fusion** - 1.2-1.5x speedup on forward pass

### Tier 2: Important (Should Do)
4. **Request Batching** - 5-8x for batch=8 (if concurrent requests)
5. **Pipelined Generation** - 1.05-1.1x latency improvement

### Tier 3: Optional (Nice To Have)
6. **INT8 Quantization** - 1.5-2x if accuracy acceptable
7. **Speculative Decoding** - 1.2-1.5x if you can train draft model

---

## Estimated Final Performance

### Single Token Generation (Batch=1)
```
TinyLlama-1.1B:
  Naive GPU:        100 t/s
  + KV Cache:       150 t/s
  + Flash Att:      200 t/s
  + Fusion:         240 t/s (1ms per token!)
  
Mistral-7B:
  Naive GPU:        60 t/s (6x model)
  + KV Cache:       90 t/s
  + Flash Att:      120 t/s
  + Fusion:         150 t/s (6.6ms per token)
```

### Batch Processing (Batch=8)
```
TinyLlama-1.1B:
  Batch throughput: 240 * 8 * 0.85 = 1632 t/s!
  
Mistral-7B:
  Batch throughput: 150 * 8 * 0.85 = 1020 t/s!
```

---

## Success Criteria for 150+ t/s

✅ **Single Request**
- TinyLlama-1.1B: ≥150 t/s
- Mistral-7B: ≥100 t/s

✅ **Batch Processing**
- Batch=4: ≥300 t/s
- Batch=8: ≥500 t/s

✅ **Latency**
- TTFT: <100ms (prompt processing)
- TpT: <7ms (per-token latency)

✅ **Quality**
- Accuracy maintained (full precision)
- No numerical instabilities
- Proper attention masking

✅ **Stability**
- Handles edge cases (empty prompt, long sequences)
- Memory cleanup (no leaks)
- Error handling (GPU OOM, etc.)

---

## Testing Strategy

### Unit Tests (Day 2-3)
```rust
#[test]
fn test_kv_cache_correctness() {
    // Verify KV cache produces same results as full computation
}

#[test]
fn test_flash_attention_numerical_stability() {
    // Check flash attention matches standard attention
}

#[test]
fn test_batching_consistency() {
    // Single request vs batch should give same results
}
```

### Integration Tests (Day 3-4)
```rust
#[test]
fn test_tinyllama_150tps() {
    // Measure throughput ≥150 t/s
}

#[test]
fn test_mistral_100tps() {
    // Measure throughput ≥100 t/s
}

#[test]
fn test_concurrent_requests() {
    // Multiple requests processed correctly
}
```

### Benchmark Suite (Day 4)
```bash
# Compare implementations
./real-benchmark --format safetensors-gpu \
                  --runs 20 \
                  --output results-final.csv

# Analyze results
python3 analyze_benchmarks.py results-final.csv
```

---

## Risk Mitigation

### Risk: GPU Out of Memory
**Mitigation:** Implement gradient checkpointing + dynamic batching
```rust
if gpu_memory_available() < required {
    reduce_batch_size();
}
```

### Risk: Numerical Instability
**Mitigation:** Keep full precision until final layer
```rust
let mut hidden = input;  // FP32
// ... compute ...
let output = hidden.cast::<f32>();  // Result in FP32
```

### Risk: Slower Than GGUF
**Mitigation:** If optimization doesn't work, fall back to GGUF
```rust
if safetensors_performance < gguf_performance {
    use_gguf_backend();
}
```

---

## Tools & Benchmarks We'll Create

1. **gpu-safetensors-bench.rs** - Specialized GPU benchmarking
2. **kv_cache_benchmark.rs** - Measure KV cache speedup
3. **attention_benchmark.rs** - Compare Flash Attention vs naive
4. **batching_benchmark.rs** - Measure batch scaling
5. **analyze_benchmarks.py** - Visualize performance graphs

---

## Final Architecture After Phase 4

```
InferenceRequest
    ↓
RequestQueue
    ↓
BatchingEngine (request batching)
    ↓
GPUSafeTensorsBackend
    ├─ ForwardPass
    │  ├─ Embedding
    │  ├─ TransformerLayer (fused)
    │  │  ├─ RMSNorm (fused with next op)
    │  │  ├─ FlashAttention (with KV cache)
    │  │  ├─ SiLU+Matmul (fused)
    │  │  └─ Residual
    │  └─ OutputProj
    │
    ├─ KVCache (persistent across tokens)
    ├─ TokenSampler
    └─ PipelinedExecution (async GPU/CPU)
    
    ↓
Output (150+ t/s)
```

---

## Conclusion

**150+ t/s is absolutely achievable** through:
1. Proper GPU backend (50-100 t/s)
2. KV cache + Flash Attention (100-130 t/s)
3. Batching + quantization (130-150+ t/s)

The math is clear:
- GPU has 100+ TFLOPS
- LLM inference needs ~2T operations per token
- 100T / 2T = 50 t/s theoretical minimum
- With optimizations: 150-300 t/s single request
- With batching: 1000+ t/s sustained

**Timeline:** 4 days  
**Effort:** 30-40 hours  
**Payoff:** 5-10x performance improvement

**Let's build it!**
