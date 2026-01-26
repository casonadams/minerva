# Phase 4: GPU Backend for SafeTensors Implementation Plan

**Decision:** Add GPU backend (burn-rs) for native SafeTensors execution  
**Timeline:** 2-3 days  
**Target:** 50-100 t/s SafeTensors throughput with full precision  
**Status:** Planning Phase

---

## Architecture Overview

### Current State
```
GGUF → llama_cpp_backend (GPU) → 30-100 t/s ✓
SafeTensors → pure_rust_backend (CPU) → 18-20 t/s (fake) ✗
```

### Target State
```
GGUF → llama_cpp_backend (GPU) → 30-100 t/s ✓
SafeTensors → gpu_safetensors_backend (burn-rs) → 50-100 t/s ✓
[Fallback] → pure_rust_backend (SIMD) → 15-25 t/s ✓
```

---

## Phase 4A: Dependency Setup (Day 1 Morning - 1 hour)

### Add burn-rs to Cargo.toml
```toml
[dependencies]
# GPU Inference Framework
burn = { version = "0.13", features = [
    "backend-ndarray",      # CPU fallback
    "backend-wgpu",         # WebGPU (cross-platform)
    "backend-cuda",         # NVIDIA GPU (optional)
    "backend-metal",        # Apple Metal (optional)
] }
```

### Create New Module Structure
```
src/inference/
├── gpu_safetensors_backend.rs      [NEW - 150-200 lines]
├── gpu_models/                     [NEW - model definitions]
│   ├── transformer.rs              [Attention, FFN, Embedding]
│   ├── llama.rs                    [LLaMA-specific layers]
│   └── mistral.rs                  [Mistral-specific layers]
└── safetensors_loader.rs           [NEW - Load weights from SafeTensors]
```

### Update Module Exports
```rust
// src/inference/mod.rs
pub mod gpu_safetensors_backend;
pub mod gpu_models;
pub mod safetensors_loader;
```

---

## Phase 4B: GPU SafeTensors Backend Implementation (Day 1 Afternoon - 4-5 hours)

### Step 1: Weight Loader Module

**File:** `src/inference/safetensors_loader.rs` (100-120 lines)

```rust
use safetensors::SafeTensors;
use burn::tensor::Tensor;
use burn::backend::Backend;
use std::path::Path;

pub struct SafeTensorsLoader;

impl SafeTensorsLoader {
    /// Load weights from SafeTensors file into Burn tensors
    pub fn load<B: Backend>(
        path: &Path,
        device: &B::Device,
    ) -> Result<TransformerWeights<B>> {
        let buffer = std::fs::read(path)?;
        let safetensors = SafeTensors::deserialize(&buffer)?;
        
        let mut weights = TransformerWeights::new();
        
        // Load embedding weights
        if let Some(embedding) = safetensors.tensor("model.embed_tokens.weight") {
            weights.embedding = Self::tensor_from_safetensors::<B>(embedding, device)?;
        }
        
        // Load attention weights (for each layer)
        for layer_idx in 0..32 {  // Adjust for model
            let prefix = format!("model.layers.{}", layer_idx);
            
            // Q, K, V projections
            if let Some(q_weight) = safetensors.tensor(&format!("{}.self_attn.q_proj.weight", prefix)) {
                weights.layers[layer_idx].q_proj = Self::tensor_from_safetensors::<B>(q_weight, device)?;
            }
            // ... same for K, V
            
            // FFN weights
            if let Some(gate) = safetensors.tensor(&format!("{}.mlp.gate_proj.weight", prefix)) {
                weights.layers[layer_idx].gate = Self::tensor_from_safetensors::<B>(gate, device)?;
            }
            // ... same for up_proj, down_proj
        }
        
        // Load layer norms and final weights
        // ...
        
        Ok(weights)
    }
    
    fn tensor_from_safetensors<B: Backend>(
        sf_tensor: &safetensors::tensor::TensorView,
        device: &B::Device,
    ) -> Result<Tensor<B, 2>> {
        let shape = sf_tensor.shape();
        let data: Vec<f32> = sf_tensor.data().iter().copied().collect();
        
        Tensor::from_floats(data, device).reshape(shape)
    }
}

#[derive(Clone)]
pub struct TransformerWeights<B: Backend> {
    pub embedding: Tensor<B, 2>,
    pub layers: Vec<LayerWeights<B>>,
    pub norm: Tensor<B, 1>,
    pub lm_head: Tensor<B, 2>,
}

#[derive(Clone)]
pub struct LayerWeights<B: Backend> {
    pub q_proj: Tensor<B, 2>,
    pub k_proj: Tensor<B, 2>,
    pub v_proj: Tensor<B, 2>,
    pub o_proj: Tensor<B, 2>,
    pub gate: Tensor<B, 2>,
    pub up_proj: Tensor<B, 2>,
    pub down_proj: Tensor<B, 2>,
    pub norm1: Tensor<B, 1>,
    pub norm2: Tensor<B, 1>,
}
```

### Step 2: GPU Model Definitions

**File:** `src/inference/gpu_models/transformer.rs` (120-150 lines)

```rust
use burn::{
    module::Module,
    tensor::Tensor,
    backend::Backend,
};

#[derive(Module)]
pub struct TransformerLayer<B: Backend> {
    pub self_attention: MultiHeadAttention<B>,
    pub feed_forward: FeedForward<B>,
    pub norm1: LayerNorm<B>,
    pub norm2: LayerNorm<B>,
}

impl<B: Backend> TransformerLayer<B> {
    pub fn forward(
        &self,
        x: Tensor<B, 2>,
        cache: Option<&mut KVCache<B>>,
    ) -> Result<Tensor<B, 2>> {
        // Pre-norm architecture (like Llama/Mistral)
        let norm_x = self.norm1.forward(x.clone());
        let attn_out = self.self_attention.forward(norm_x, cache)?;
        let x = x.add(&attn_out);  // Residual
        
        // FFN block
        let norm_x = self.norm2.forward(x.clone());
        let ffn_out = self.feed_forward.forward(norm_x)?;
        Ok(x.add(&ffn_out))  // Residual
    }
}

#[derive(Module)]
pub struct MultiHeadAttention<B: Backend> {
    pub q_proj: Linear<B>,
    pub k_proj: Linear<B>,
    pub v_proj: Linear<B>,
    pub o_proj: Linear<B>,
    pub num_heads: usize,
    pub head_dim: usize,
}

impl<B: Backend> MultiHeadAttention<B> {
    pub fn forward(
        &self,
        x: Tensor<B, 2>,
        cache: Option<&mut KVCache<B>>,
    ) -> Result<Tensor<B, 2>> {
        let batch_size = x.shape()[0];
        let seq_len = x.shape()[1];
        
        // Project to Q, K, V
        let q = self.q_proj.forward(x.clone());
        let k = self.k_proj.forward(x.clone());
        let v = self.v_proj.forward(x);
        
        // Reshape for multi-head: (batch, seq, hidden) → (batch, seq, heads, head_dim)
        let q = q.reshape([batch_size, seq_len, self.num_heads, self.head_dim]);
        let k = k.reshape([batch_size, seq_len, self.num_heads, self.head_dim]);
        let v = v.reshape([batch_size, seq_len, self.num_heads, self.head_dim]);
        
        // KV Cache: reuse past keys/values
        let (k, v) = if let Some(cache) = cache {
            cache.update(&k, &v)?
        } else {
            (k, v)
        };
        
        // Attention scores
        let scores = q.matmul(&k.transpose(-2, -1)); // (batch, seq, heads, seq)
        let scores = scores / (self.head_dim as f32).sqrt();
        let attn_weights = scores.softmax(-1);
        
        // Apply attention to values
        let output = attn_weights.matmul(&v);
        
        // Reshape back: (batch, heads, seq, head_dim) → (batch, seq, hidden)
        let output = output.reshape([batch_size, seq_len, self.num_heads * self.head_dim]);
        
        Ok(self.o_proj.forward(output))
    }
}

#[derive(Module)]
pub struct FeedForward<B: Backend> {
    pub gate_proj: Linear<B>,
    pub up_proj: Linear<B>,
    pub down_proj: Linear<B>,
}

impl<B: Backend> FeedForward<B> {
    pub fn forward(&self, x: Tensor<B, 2>) -> Result<Tensor<B, 2>> {
        let gate = self.gate_proj.forward(x.clone()).silu();  // SiLU activation
        let up = self.up_proj.forward(x);
        let combined = gate * up;  // Element-wise product (like LLaMA)
        Ok(self.down_proj.forward(combined))
    }
}

pub struct KVCache<B: Backend> {
    pub k: Tensor<B, 3>,  // (batch, seq, hidden)
    pub v: Tensor<B, 3>,
}

impl<B: Backend> KVCache<B> {
    pub fn update(
        &mut self,
        new_k: &Tensor<B, 3>,
        new_v: &Tensor<B, 3>,
    ) -> Result<(Tensor<B, 3>, Tensor<B, 3>)> {
        // Append new tokens to cache
        self.k = Tensor::cat([self.k.clone(), new_k.clone()], 1);
        self.v = Tensor::cat([self.v.clone(), new_v.clone()], 1);
        Ok((self.k.clone(), self.v.clone()))
    }
}
```

### Step 3: Main GPU Backend

**File:** `src/inference/gpu_safetensors_backend.rs` (150-180 lines)

```rust
use crate::error::{MinervaError, MinervaResult};
use crate::inference::inference_backend_trait::{GenerationParams, InferenceBackend};
use crate::inference::safetensors_loader::{SafeTensorsLoader, TransformerWeights};
use crate::inference::gpu_models::TransformerLayer;
use burn::backend::Wgpu;  // or Cuda/Metal
use burn::tensor::Tensor;
use std::path::Path;
use std::sync::Arc;
use tokio::sync::Mutex;

type B = Wgpu;  // GPU backend (can be switched to Cuda/Metal)

pub struct GPUSafeTensorsBackend {
    weights: Arc<Mutex<Option<TransformerWeights<B>>>>,
    tokenizer: Arc<Mutex<Option<LLaMATokenizer>>>,
    device: B::Device,
    n_ctx: usize,
}

impl GPUSafeTensorsBackend {
    pub fn new() -> Self {
        let device = B::Device::default();  // Auto-detect GPU
        Self {
            weights: Arc::new(Mutex::new(None)),
            tokenizer: Arc::new(Mutex::new(None)),
            device,
            n_ctx: 0,
        }
    }
}

impl InferenceBackend for GPUSafeTensorsBackend {
    fn load_model(&mut self, path: &Path, n_ctx: usize) -> MinervaResult<()> {
        // Load SafeTensors and convert to GPU tensors
        let weights = SafeTensorsLoader::load::<B>(path, &self.device)
            .map_err(|e| MinervaError::LoadFailed(format!("Failed to load weights: {}", e)))?;
        
        let mut w = self.weights.blocking_lock();
        *w = Some(weights);
        self.n_ctx = n_ctx;
        
        Ok(())
    }

    fn generate(&self, prompt: &str, params: GenerationParams) -> MinervaResult<String> {
        let runtime = tokio::runtime::Handle::current();
        runtime.block_on(self.generate_async(prompt, params))
    }
}

impl GPUSafeTensorsBackend {
    async fn generate_async(
        &self,
        prompt: &str,
        params: GenerationParams,
    ) -> MinervaResult<String> {
        let weights = self.weights.lock().await;
        let weights = weights.as_ref()
            .ok_or_else(|| MinervaError::NotLoaded)?;
        
        // Tokenize input
        let token_ids = self.tokenize(prompt)?;
        let tokens = Tensor::from_data(
            Data::from(token_ids.iter().map(|&t| t as f32).collect::<Vec<_>>()),
            &self.device,
        ).unsqueeze::<2>([0]);  // Add batch dimension
        
        // Embedding
        let mut hidden = weights.embedding.index_select(&tokens, 0);
        
        // Forward pass through layers
        let mut kv_cache = KVCache::new();
        for layer_weights in &weights.layers {
            let layer = TransformerLayer::from_weights(layer_weights);
            hidden = layer.forward(hidden, Some(&mut kv_cache))?;
        }
        
        // Final norm and output
        hidden = weights.norm.forward(hidden);
        let logits = weights.lm_head.matmul(&hidden.transpose(-2, -1));
        
        // Sample next token
        let next_token = self.sample_token(&logits, params.temperature, params.top_p)?;
        
        let mut result = self.tokenize_to_string(&token_ids)?;
        for _ in 0..params.max_tokens {
            // Continue generation (incremental decode)
            let new_token = self.forward_single(&next_token, weights)?;
            result.push_str(&self.decode(&[new_token])?);
        }
        
        Ok(result)
    }
    
    fn sample_token(
        &self,
        logits: &Tensor<B, 3>,
        temperature: f32,
        top_p: f32,
    ) -> MinervaResult<u32> {
        // Standard sampling with temperature and top-p
        // Returns next token ID
        todo!()
    }
}
```

---

## Phase 4C: Testing & Optimization (Day 2 - 4-5 hours)

### Step 1: Unit Tests

**File:** `src/inference/gpu_safetensors_backend.rs` (at bottom)

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_safetensors() {
        let backend = GPUSafeTensorsBackend::new();
        let path = Path::new("../models/tinyllama-1.1b-safetensors/model.safetensors");
        
        if !path.exists() {
            return;  // Skip if model not available
        }
        
        assert!(backend.load_model(path, 512).is_ok());
    }

    #[test]
    fn test_generate() {
        let backend = GPUSafeTensorsBackend::new();
        let path = Path::new("../models/tinyllama-1.1b-safetensors/model.safetensors");
        
        if !path.exists() {
            return;
        }
        
        backend.load_model(path, 512).unwrap();
        let params = GenerationParams {
            max_tokens: 50,
            temperature: 0.7,
            top_p: 0.9,
        };
        
        let result = backend.generate("Hello", params);
        assert!(result.is_ok());
        assert!(!result.unwrap().is_empty());
    }
}
```

### Step 2: Integration Tests

**File:** `tests/gpu_safetensors_integration.rs`

```rust
#[tokio::test]
async fn test_tinyllama_gpu_inference() {
    let backend = GPUSafeTensorsBackend::new();
    backend.load_model(
        Path::new("models/tinyllama-1.1b-safetensors/model.safetensors"),
        512,
    ).unwrap();
    
    let result = backend.generate(
        "What is AI?",
        GenerationParams {
            max_tokens: 100,
            temperature: 0.7,
            top_p: 0.9,
        },
    ).unwrap();
    
    assert!(result.len() > 20);
}
```

### Step 3: Benchmark

**Update:** `src/bin/real-benchmark.rs`

```rust
// Add GPU backend to benchmarking
match backend {
    "gguf" => { /* existing */ },
    "safetensors-gpu" => {
        // Time GPU SafeTensors backend
        let gpu_backend = GPUSafeTensorsBackend::new();
        gpu_backend.load_model(model_path, 2048)?;
        
        let start = Instant::now();
        let output = gpu_backend.generate(prompt, params)?;
        let elapsed = start.elapsed();
    },
    _ => {},
}
```

---

## Phase 4D: Optimization (Day 3 - 2-3 hours)

### KV Cache Optimization
```rust
// Already included in KVCache design
// Reuse past computations for faster incremental generation
```

### Batch Processing
```rust
// Support multiple sequences in parallel
pub fn generate_batch(
    &self,
    prompts: Vec<&str>,
    params: GenerationParams,
) -> MinervaResult<Vec<String>> {
    // Process multiple sequences together
    // Amortizes attention computation
}
```

### Memory Management
```rust
// Implement gradient checkpointing for large models
// Support offloading to CPU if GPU memory limited
```

---

## Expected Results After Implementation

### Performance

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| **Throughput** | 18-20 t/s | 50-100 t/s | 2.5-5x |
| **TTFT** | 80-100ms | 40-60ms | 1.5-2x |
| **TpT** | 40-50ms | 15-25ms | 1.6-2x |
| **Memory** | 2.0GB | 2.0GB | Same |
| **Precision** | Full (fake) | Full (real) | Accurate |

### Comparison

```
After GPU Backend Implementation:
  GGUF:        30-100 t/s  ✓
  SafeTensors: 50-100 t/s  ✓ (NEW!)
  Pure Rust:   15-25 t/s   ✓ (fallback)
```

---

## Implementation Checklist

### Day 1
- [ ] Add burn-rs to Cargo.toml (30 min)
- [ ] Create module structure (30 min)
- [ ] Implement SafeTensorsLoader (1 hour)
- [ ] Implement TransformerLayer models (2 hours)
- [ ] Implement GPUSafeTensorsBackend (2 hours)

### Day 2
- [ ] Unit tests (1 hour)
- [ ] Integration tests (1 hour)
- [ ] Fix compilation errors (1-2 hours)
- [ ] Benchmark implementation (1 hour)
- [ ] Initial performance tuning (1 hour)

### Day 3
- [ ] KV cache optimization (30 min)
- [ ] Batch processing support (1 hour)
- [ ] Memory optimizations (1 hour)
- [ ] Final benchmarking and analysis (30 min)
- [ ] Documentation (30 min)

---

## Risk Mitigation

### Potential Issues & Solutions

| Issue | Solution |
|-------|----------|
| burn-rs learning curve | Study existing examples, start simple |
| Weight format mismatch | Verify with print statements, compare with llama.cpp |
| GPU out of memory | Implement gradient checkpointing, reduce batch size |
| Compilation errors | Use cargo check iteratively |
| Slow first run | CUDA/Metal compilation overhead, normal |

---

## Success Criteria

✅ **GPU backend compiles without errors**
✅ **Loads TinyLlama-1.1B SafeTensors successfully**
✅ **Generates text (quality doesn't matter for Phase 4)**
✅ **Benchmarks show 50-100 t/s throughput**
✅ **Maintains 100% standards compliance**
✅ **Zero unsafe code**
✅ **< 200 lines per file**

---

## Phase 5: What Comes After

1. **Mistral-7B Testing** (1 day)
   - Download full Mistral-7B SafeTensors
   - Benchmark with GPU backend
   - Compare GGUF vs GPU backends

2. **Competitive Analysis** (1-2 days)
   - Compare against ollama, llama.cpp, vLLM
   - Identify further optimizations

3. **Production Hardening** (2-3 days)
   - Error recovery
   - Concurrent requests
   - Resource monitoring

---

## Conclusion

Adding a proper GPU backend for SafeTensors is the **right architectural decision**. It:
- Gives full precision (no quantization)
- Provides 50-100 t/s performance
- Works across GPU platforms (CUDA, Metal, DirectML)
- Sets foundation for production deployment
- Aligns with industry best practices

**Estimated total effort:** 8-10 hours over 2-3 days  
**Expected payoff:** 5x performance improvement + architectural correctness

Ready to implement!
