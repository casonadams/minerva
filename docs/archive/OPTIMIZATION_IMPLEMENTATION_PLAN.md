# GPU Optimization Implementation Plan

**Status:** Starting Phase 1  
**Goal:** Optimize from 2-10 t/s → 100-500 t/s with 4-8K context  
**Timeline:** This session

---

## Phase 1: Complete Tensor Loading (2-3 hours)

### What We Need
Load actual GGUF tensor data and wire into inference pipeline.

### Current State
- ✅ GGUF header parsing works
- ✅ Metadata extraction works
- ✅ Tensor headers detected
- ❌ Actual tensor data NOT loaded
- ❌ Forward pass is a stub

### Implementation Steps

#### Step 1.1: Implement Dequantization Kernels
**File:** `src-tauri/src/inference/gpu/gguf_loader.rs`

Add these functions:
```rust
// MXFP4 dequantization (4-bit, 1 exponent per 32 values)
pub fn dequant_mxfp4(quantized: &[u8]) -> Vec<f32> {
    // Convert MXFP4 4-bit values to f32
    // Block size: 32 values = 16 bytes
    // Format: [4-bit mantissa] [shared exponent]
}

// Q8 dequantization (8-bit, simple scaling)
pub fn dequant_q8(quantized: &[u8], scale: f32) -> Vec<f32> {
    // Convert 8-bit quantized values to f32
    // Simply: (i8 as f32) * scale
}
```

**Estimated time:** 30-45 minutes

#### Step 1.2: Implement Tensor Header Reading
**Already mostly done** in `tool_optimized_loader.rs`

Need to extend to:
- Parse all 459 tensor headers
- Extract: name, shape, dtype, file offset
- Organize into data structure

**Estimated time:** 15 minutes

#### Step 1.3: Implement Tensor Data Loading
**File:** `src-tauri/src/inference/gpu/gguf_loader.rs`

Add function:
```rust
pub fn load_tensor_data(
    file_path: &Path,
    tensor_header: &GGUFTensorHeader,
) -> MinervaResult<Vec<f32>> {
    // 1. Seek to file offset
    // 2. Read quantized bytes
    // 3. Dequantize based on dtype
    // 4. Return f32 vector
}
```

**Estimated time:** 45-60 minutes

#### Step 1.4: Organize Tensors by Layer
**File:** `src-tauri/src/inference/gpu/backend.rs`

Create structure:
```rust
pub struct LoadedTensors {
    pub embedding: Vec<f32>,
    pub lm_head: Vec<f32>,
    pub layers: Vec<LayerTensors>,
}

pub struct LayerTensors {
    pub attn_q: Vec<f32>,
    pub attn_k: Vec<f32>,
    pub attn_v: Vec<f32>,
    pub attn_out: Vec<f32>,
    pub mlp_gate: Vec<f32>,
    pub mlp_up: Vec<f32>,
    pub mlp_down: Vec<f32>,
    pub norm_attn: Vec<f32>,
    pub norm_mlp: Vec<f32>,
}
```

**Estimated time:** 30 minutes

### Testing Phase 1
```bash
# Build
cargo build --release

# Test tensor loading
cargo test --lib inference::gpu::gguf_loader::tests::test_load_gpt_oss_20b_gguf -- --ignored

# Should output:
# ✓ Loaded 459 tensors
# ✓ Embedding shape correct
# ✓ All 24 layers present
# ✓ Shapes match expected
```

---

## Phase 2: Implement Forward Pass (2-3 hours)

### What We Need
Wire loaded tensors into a working inference pipeline.

### Implementation Steps

#### Step 2.1: Reshape Tensors to ndarrays
```rust
pub fn reshape_to_2d(data: Vec<f32>, shape: (usize, usize)) -> Array2<f32> {
    Array2::from_shape_vec(shape, data).unwrap()
}
```

#### Step 2.2: Implement Full Forward Pass
**File:** `src-tauri/src/inference/gpu/backend.rs`

```rust
pub fn forward_pass(
    &self,
    input_ids: &[u32],
    kv_cache: Option<&mut KVCacheOptimized>,
) -> MinervaResult<Array2<f32>> {
    // 1. Embedding lookup
    let mut hidden = embedding_lookup(input_ids, &self.embedding);
    
    // 2. Process through 24 layers
    for (layer_idx, layer) in self.layers.iter().enumerate() {
        // 2a. Attention norm
        hidden = rms_norm(&hidden, &self.norm_attn_weights[layer_idx]);
        
        // 2b. Attention
        let attn_out = gqa_attention(
            &hidden,
            &layer.attn_q,
            &layer.attn_k,
            &layer.attn_v,
            &layer.attn_out,
            kv_cache,
            layer_idx,
        );
        
        // 2c. Residual connection
        hidden = hidden + attn_out;
        
        // 2d. FFN norm
        hidden = rms_norm(&hidden, &self.norm_mlp_weights[layer_idx]);
        
        // 2e. MLP
        let mlp_out = mlp_forward(
            &hidden,
            &layer.mlp_gate,
            &layer.mlp_up,
            &layer.mlp_down,
        );
        
        // 2f. Residual connection
        hidden = hidden + mlp_out;
    }
    
    // 3. Final norm
    hidden = rms_norm(&hidden, &self.norm_weight);
    
    // 4. Logits
    let logits = matmul(&hidden, &self.lm_head);
    
    Ok(logits)
}
```

**Estimated time:** 90-120 minutes

#### Step 2.3: Create Generation Loop
```rust
pub fn generate(
    &mut self,
    input_ids: Vec<u32>,
    max_tokens: usize,
) -> MinervaResult<Vec<u32>> {
    let mut generated = input_ids.clone();
    let mut kv_cache = KVCacheOptimized::new(self.config.num_hidden_layers);
    
    for _ in 0..max_tokens {
        // Forward pass on last token only (with cache)
        let logits = self.forward_pass(&[*generated.last().unwrap()], Some(&mut kv_cache))?;
        
        // Sample next token
        let next_token = sample_top_k(&logits, 40, 0.9);
        generated.push(next_token);
    }
    
    Ok(generated)
}
```

**Estimated time:** 30-45 minutes

### Testing Phase 2
```bash
# Test forward pass
cargo test --lib inference::gpu::backend::tests::test_forward_pass

# Should output:
# ✓ Output shape correct: (1, 201088)
# ✓ Output values in valid range
# ✓ No NaNs or Infs
# ✓ Logits look reasonable
```

---

## Phase 3: Integrate KV Cache (1-2 hours)

### What We Have
- ✅ KV cache data structure (already implemented)
- ✅ Append/get operations
- ❌ Not integrated into forward pass

### Implementation Steps

#### Step 3.1: Wire KV Cache into Attention
**Modify:** `src-tauri/src/inference/gpu/attention_kernel.rs`

```rust
pub fn gqa_attention_with_cache(
    queries: &Array2<f32>,
    keys: &Array2<f32>,
    values: &Array2<f32>,
    cache: &mut KVCacheOptimized,
    layer_idx: usize,
) -> Array2<f32> {
    // 1. Append current K,V to cache
    cache.append(keys, values, layer_idx);
    
    // 2. Get cached K,V from all previous tokens
    let (cached_k, cached_v) = cache.get(layer_idx);
    
    // 3. Compute attention: Q @ (all K) -> weights
    // 4. Apply weights to (all V) -> output
    
    // This avoids re-computing attention for old tokens!
    attention_computation(queries, cached_k, cached_v)
}
```

**Estimated time:** 45-60 minutes

#### Step 3.2: Test KV Cache Integration
```rust
#[test]
fn test_cache_improves_speed() {
    // 1. First pass: 100 tokens without cache
    let time_without = measure_time(|| forward_pass(100, None));
    
    // 2. Second pass: 1 token with cache
    let time_with_cache = measure_time(|| forward_pass(1, Some(&mut cache)));
    
    // 3. Expect: time_with_cache << time_without
    // Realistic: 10x speedup
    assert!(time_with_cache < time_without / 8);
}
```

**Estimated time:** 30 minutes

---

## Phase 4: Measure and Profile (1-2 hours)

### Create Measurement Framework

**File:** `src-tauri/src/bin/inference-benchmark.rs`

```rust
fn main() {
    let model = load_gpt_oss_20b()?;
    
    // Benchmark 1: Single token
    let start = Instant::now();
    let logits = model.forward_pass(&[1234])?;
    println!("Single token: {}ms", start.elapsed().as_millis());
    
    // Benchmark 2: Sequence without cache
    let start = Instant::now();
    let logits = model.forward_pass(&input_ids)?;
    println!("Full sequence (no cache): {}ms", start.elapsed().as_millis());
    
    // Benchmark 3: Sequence with cache
    let start = Instant::now();
    let mut cache = KVCacheOptimized::new(24);
    for token in &input_ids {
        model.forward_pass(&[*token], Some(&mut cache))?;
    }
    println!("Full sequence (with cache): {}ms", start.elapsed().as_millis());
    
    // Benchmark 4: Generation
    let start = Instant::now();
    let generated = model.generate(input_ids, 100)?;
    let elapsed = start.elapsed().as_secs_f32();
    println!("Generated {} tokens in {:.2}s = {:.1} t/s",
        generated.len(), elapsed, generated.len() as f32 / elapsed);
}
```

**Estimated time:** 45-60 minutes

### Profile with Instruments
```bash
# On macOS, use Instruments.app to profile:
# - Memory usage
# - CPU time per operation
# - Memory bandwidth
# - Cache misses
```

---

## Phase 5: Apply Flash Attention (2-3 hours)

### Current State
- ✅ Flash Attention kernel already implemented
- ❌ Not integrated into forward pass

### Implementation
Replace naive attention with Flash Attention:

```rust
// Before:
let attn_output = naive_attention(Q, K, V);

// After:
let attn_output = flash_attention_approx(Q, K, V, block_size=64);
```

**Expected improvement:** 3-5x speedup

---

## Success Criteria

### Phase 1 (Tensor Loading)
- [ ] Load all 459 tensors from GGUF
- [ ] Dequantization works correctly
- [ ] Shapes match expected dimensions
- [ ] Test: `test_load_gpt_oss_20b_gguf` passes

### Phase 2 (Forward Pass)
- [ ] Single token forward pass works
- [ ] Output shape: (1, 201088)
- [ ] Values are reasonable (not all zeros or NaNs)
- [ ] Test: `test_forward_pass` passes

### Phase 3 (KV Cache)
- [ ] KV cache integration works
- [ ] Generation produces valid tokens
- [ ] Cache provides >5x speedup
- [ ] Test: `test_cache_improves_speed` passes

### Phase 4 (Benchmarking)
- [ ] Single token: <100ms
- [ ] Generate 100 tokens: <10 seconds
- [ ] Throughput: >10 tokens/second

### Phase 5 (Flash Attention)
- [ ] Flash Attention integrated
- [ ] Single token: <50ms
- [ ] Generate 100 tokens: <5 seconds  
- [ ] Throughput: >20 tokens/second

---

## Rollback Strategy

If any phase fails:
1. Keep previous working version in git
2. Can always revert: `git checkout <previous-commit>`
3. Incremental commits after each phase success

---

## Timeline Estimate

```
Phase 1: 2-3 hours
Phase 2: 2-3 hours
Phase 3: 1-2 hours
Phase 4: 1-2 hours
Phase 5: 2-3 hours
─────────────────
Total:   8-13 hours

Realistic with debugging: 12-15 hours
```

---

## Let's Start!

Beginning with **Phase 1: GGUF Tensor Loading**

Next steps:
1. Implement dequantization kernels
2. Implement tensor data loading
3. Test with actual GGUF file
4. Measure load time

Let's go! 🚀
