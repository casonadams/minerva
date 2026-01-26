# Maximum Throughput Optimization Strategy
## GPT-OSS 20B - Aiming for 500+ tokens/sec

**Date:** January 25, 2026  
**Goal:** Achieve 300-500 tokens/sec on your Mac  
**Current Status:** Foundation ready, about to implement optimization layers

---

## Performance Bottleneck Analysis

### Expected Bottlenecks (in order of impact)

1. **Matrix Multiplication (60% of compute time)**
   - Current: ndarray matmul (single-threaded)
   - Optimization: BLAS/SIMD parallelization
   - Target: 10-50x improvement

2. **Attention Computation (25% of compute time)**
   - Current: Naive implementation O(N²)
   - Optimization: Flash Attention, KV caching
   - Target: 5-20x improvement

3. **Quantization Overhead (10% of compute time)**
   - Current: Per-layer dequantization
   - Optimization: Vectorized dequant, block-level caching
   - Target: 2-5x improvement

4. **Memory Bandwidth (5% of compute time)**
   - Current: Naive tensor access
   - Optimization: Operator fusion, loop tiling
   - Target: 2-3x improvement

---

## Optimization Roadmap (Fast Path)

### CRITICAL PATH - Focus on These First

#### Phase 1: Core Inference (6-8 hours)
```
1. GGUF tensor loading + dequantization    [2-3h] ← BLOCKING
2. GQA attention implementation             [2-3h] ← BLOCKING  
3. End-to-end forward pass                  [1-2h] ← BLOCKING
4. Initial throughput benchmark             [1h]
```

**Expected Results After Phase 1:**
- Single token: 10-30 t/s (naive)
- Batch of 20: 50-100 t/s
- Load time: 30-60s

#### Phase 2: Fast Attention (4-6 hours)
```
1. Flash Attention or approximation         [3-4h]
2. KV cache for generation                  [1-2h]
3. Benchmark with caching enabled           [1h]
```

**Expected Results After Phase 2:**
- Single token: 30-50 t/s
- Batch of 20: 100-200 t/s
- Reduction: 50% computation time

#### Phase 3: Quantization Optimization (3-4 hours)
```
1. SIMD dequantization kernels              [2h]
2. Block-level caching                      [1-2h]
3. Benchmark improvement                    [1h]
```

**Expected Results After Phase 3:**
- Single token: 50-100 t/s
- Batch of 20: 150-300 t/s
- Reduction: 30% per-layer overhead

#### Phase 4: Advanced Optimization (4-5 hours)
```
1. Request batching scheduler               [2h]
2. Operator fusion (attention + mlp)        [2-3h]
3. Memory layout optimization               [1h]
```

**Expected Results After Phase 4:**
- Single token: 100+ t/s
- Batch of 20: 250-400 t/s
- Amortization: 2-3x from batching

#### Phase 5: Final Push (2-3 hours)
```
1. Speculative decoding (if time permits)   [2-3h]
2. Final kernel tuning                      [1h]
```

**Final Target:**
- Single token: 150+ t/s
- Batch of 20: 300-500+ t/s

---

## Technical Optimization Details

### 1. GGUF Tensor Loading (2-3 hours)

**Current Status:** Header parsing works ✓

**What's Needed:**
```rust
// Parse tensor data section
fn load_tensor_data(reader: &mut BufReader, tensor_info: &TensorInfo) -> Vec<u8>

// Dequantize MXFP4 blocks
fn dequant_mxfp4(quantized: &[u8]) -> Vec<f32> {
    // MXFP4: 4-bit mantissa + shared exponent per block
    // Block size: 32 elements
    // Very fast with SIMD: 4 bytes → 128 bytes per operation
}

// Dequantize Q8 (used in critical layers)
fn dequant_q8(quantized: &[u8], scale: f32) -> Vec<f32>
```

**Performance Impact:**
- Load time: 30-60 seconds
- Throughput impact: Minimal (one-time cost)
- Memory overhead: 0% (stays quantized in memory)

**SIMD Opportunity:**
- Vectorize block decompression (4x speedup)
- Parallel shard loading (2-3x speedup)

---

### 2. GQA Attention (2-3 hours)

**GPT-OSS Architecture:**
- 64 query heads, 8 key-value heads
- KV sharing: 8 different KV, repeated 8x for Q
- Memory: 8x less for KV cache

**Implementation Strategy:**

```rust
// Current naive: queries @ keys → (batch, 64, 64, seqlen)
// Better: use KV broadcasting

fn attention_gqa(
    queries: &Array3<f32>,      // (seq, 64, head_dim)
    keys: &Array3<f32>,         // (seq, 8, head_dim)  
    values: &Array3<f32>,       // (seq, 8, head_dim)
    kv_cache: &KVCache,
) -> Array2<f32> {
    // Step 1: Repeat KV heads 8x to match query heads
    let keys = keys.broadcast((seq, 64, head_dim))?;
    let values = values.broadcast((seq, 64, head_dim))?;
    
    // Step 2: Normal attention (but 8x smaller KV cache!)
    attention(queries, keys, values)
}
```

**Performance Impact:**
- KV cache: 8x smaller (saves bandwidth)
- Attention: Same O(N²) but fewer parameters
- Speed: 2-3x improvement from better cache locality

---

### 3. Flash Attention (3-4 hours)

**Key Insight:** Reduce memory accesses from O(N) to O(1)

**Standard Attention Memory Pattern:**
```
Read Q (64*64*4 = 16KB)
Read K (64*64*4 = 16KB)
    Compute QK^T (64x64 matmul)
    Write intermediate (64x64*4 = 16KB)
Read QK^T softmax results (16KB)
Read V (64*64*4 = 16KB)
    Compute Attention @ V (64x64 matmul)
    Write output (16KB)
Total: 100KB+ memory moves
```

**Flash Attention Pattern:**
```
for block in seq_length {
    Load Q block (4KB)
    Load K block (4KB)
    Load V block (4KB)
    Compute locally (compute-bound)
    Write output (4KB)
}
Total: 4KB working set (cache-fit!)
10-20x reduction in memory bandwidth
```

**Simplified Implementation:**
```rust
fn flash_attention_block(
    q_block: &Array2<f32>,    // (block_size, head_dim)
    k_block: &Array2<f32>,    // (block_size, head_dim)
    v_block: &Array2<f32>,    // (block_size, head_dim)
) -> Array2<f32> {
    // Do everything in-register
    let qk = q_block @ k_block.t();     // (block, block)
    let probs = softmax(qk);             // (block, block)
    let out = probs @ v_block;           // (block, head_dim)
    out
}
```

**Performance Impact:**
- Bandwidth: 10-20x reduction
- Speed: 5-20x improvement for long sequences
- Critical for batching

---

### 4. KV Cache (1-2 hours)

**Current:** Placeholder that doesn't cache

**What's Needed:**
```rust
pub struct KVCache {
    k_cache: Array3<f32>,  // (seq_len, 8, head_dim)
    v_cache: Array3<f32>,  // (seq_len, 8, head_dim)
    current_len: usize,
}

impl KVCache {
    pub fn append(&mut self, k_new: Array2<f32>, v_new: Array2<f32>) {
        // Append new K, V from this token
        // Don't recompute all previous K, V
    }
    
    pub fn get(&self, layer: usize) -> (Array3<f32>, Array3<f32>) {
        // Return cached K, V from all previous layers
    }
}
```

**Performance Impact:**
- Generation time: O(N) → O(1) per token
- Speed: 20-50x improvement for generation
- Memory: Grows with sequence length (2 * seq * 8 * head_dim)

---

### 5. Batch Processing (2-3 hours)

**Naive Batching Problem:**
```
Batch size 1:  1 seq × 64×64 matmul = 64k ops
Batch size 10: 10 seq × 64×64 matmul = 640k ops
Issue: 10x more compute, but cache misses increase too
```

**Smart Batching:**
```rust
pub struct BatchRequest {
    token_ids: Vec<usize>,
    seq_len: usize,
    priority: i32,  // For scheduling
}

pub fn forward_batch(
    backend: &Backend,
    requests: &[BatchRequest],
) -> Vec<Array2<f32>> {
    // Group requests by sequence length
    // Process in blocks (64 tokens/batch)
    // Amortize attention computation
    
    let mut results = Vec::new();
    
    for batch in requests.chunks(BATCH_SIZE) {
        // Pack all batch sequences into single tensor
        // One matmul per layer serves all sequences
        // Unpack results
        results.extend(per_request_outputs);
    }
    
    results
}
```

**Performance Impact:**
- Compute: 2-4x amortization
- Memory bandwidth: 3-5x better utilization
- Practical: 250-500 t/s with batch of 20+

---

## Implementation Priority Order

### Must Do (Blocking everything else)
1. **GGUF Tensor Loading** (2-3h) ← START HERE
2. **GQA Attention** (2-3h) ← THEN THIS
3. **Forward Pass** (1-2h) ← THEN TEST

### Very High Value
4. **Flash Attention** (3-4h) ← 5-20x speed
5. **KV Cache** (1-2h) ← 20-50x for generation
6. **Batch Processing** (2-3h) ← 2-4x amortization

### High Value (if time permits)
7. **Quantization Optimization** (3-4h) ← Reduce dequant overhead
8. **Speculative Decoding** (2-3h) ← 2-3x for decoding

### Polish (if time permits)
9. **Memory Layout** (1-2h) ← Better cache behavior
10. **Kernel Tuning** (1-2h) ← Final 10-20% gains

---

## Baseline Comparison

### Current (Estimated - Not Implemented Yet)
```
Load Time:              30-60s (GGUF), 60-90s (SafeTensors)
Memory:                 ~12GB
Single Token:           5-10 t/s (very slow, naive implementation)
Batch(20):              20-50 t/s (naive batching)
```

### Target (After Phase 1-3)
```
Load Time:              30-60s (cached)
Memory:                 ~13GB (KV cache for seq_len=100)
Single Token:           50-100 t/s (with GQA + Flash Attn)
Batch(20):              200-400 t/s (with batching)
```

### Theoretical Maximum
```
Load Time:              30s (parallel GGUF loading)
Memory:                 ~14GB (KV cache for longer sequences)
Single Token:           150+ t/s (optimized kernels)
Batch(20):             400-500+ t/s (full optimization)
```

---

## Performance Measurement Plan

### Benchmarks to Run

```bash
# 1. Model loading
time cargo run --release --bin load-model

# 2. Single forward pass (1 token)
time cargo run --release --bin forward-single-token

# 3. Generation (100 tokens)
time cargo run --release --bin generate-100-tokens

# 4. Batch inference (20 requests × 100 tokens)
time cargo run --release --bin batch-100-tokens

# 5. Throughput measurement
cargo run --release --bin throughput-benchmark -- --batch-size 20 --length 100
```

### Metrics to Track

For each benchmark:
- **Load Time** (ms)
- **Forward Pass Time** (ms)
- **Tokens/Second** (t/s)
- **Memory Peak** (GB)
- **Memory Average** (GB)

---

## Key Optimization Techniques Summary

| Technique | Speedup | Complexity | Priority |
|-----------|---------|-----------|----------|
| GGUF dequant | 1x (baseline) | Low | 1 |
| GQA attention | 2-3x | Medium | 2 |
| Flash Attention | 5-20x | High | 3 |
| KV Cache | 20-50x | Medium | 4 |
| Batch Processing | 2-4x | High | 5 |
| Quantization SIMD | 3-5x | High | 6 |
| Speculative Decode | 2-3x | Very High | 7 |
| **Combined** | **300-500x** | - | - |

---

## Success Criteria

### Phase 1 Success (Core Inference)
- [ ] GGUF tensors load completely (all 459 tensors)
- [ ] Forward pass produces valid logits
- [ ] Single forward: < 1 second
- [ ] Batch of 20: < 5 seconds
- [ ] No memory crashes
- [ ] Benchmark tool shows results

### Phase 2 Success (Fast Attention)
- [ ] Flash Attention implemented
- [ ] KV cache reduces recomputation
- [ ] Generation mode: 20+ t/s (per token)
- [ ] Batch: 100+ t/s
- [ ] Memory stable

### Phase 3 Success (Batching)
- [ ] Batch scheduler working
- [ ] 10-20 concurrent requests
- [ ] Batch throughput: 200+ t/s
- [ ] Load balanced

### Final Success (500+ t/s)
- [ ] Batch of 20: 300-500+ t/s
- [ ] Single token: 50-100 t/s
- [ ] Consistent performance
- [ ] No memory leaks

---

## Quick Win Opportunities

If running short on time, focus on these high-value items:

1. **Parallelize Matrix Operations** (1-2h, 2-5x)
   - Use rayon for parallel iteration
   - SIMD matmul with ndarray built-in parallelization

2. **KV Cache** (1-2h, 20-50x for generation)
   - Biggest bang for buck
   - Simple to implement
   - Massive impact on second token onwards

3. **Flash Attention Approximation** (2-3h, 3-10x)
   - Block-level computation
   - Simpler than full Flash Attention
   - Still massive speedup

---

## Dependencies & Crates

### For Fast Linear Algebra
```toml
ndarray = "0.15"           # Already have - supports parallel
ndarray-linalg = "0.16"    # Add - BLAS/LAPACK support
openblas-src = "0.10"      # Add - fast BLAS library
rayon = "1.7"              # Add - data parallelism
```

### For Profiling
```toml
pprof = "0.13"             # Add - flamegraph profiling
criterion = "0.5"          # Add - benchmarking framework
```

---

## Conclusion

**The path to 500+ t/s is clear:**

1. **Implement core inference** (GGUF + GQA + forward) → 50-100 t/s
2. **Add fast attention** (Flash Attn + KV cache) → 200-300 t/s
3. **Enable batching** (batch scheduler) → 300-500 t/s

**Total effort: 15-20 hours of focused work**
**Expected result: 300-500 tokens/second**

Ready to start Phase 1? 🚀
