# Phase 4 Aggressive Roadmap: 300-500 t/s in 6-7 Days

**Objective:** Build production-grade GPU-accelerated inference reaching 300-500 tokens/sec  
**Timeline:** 6-7 days (50-55 hours total)  
**Hardware Target:** RTX 4090 / H100 / Apple M-series with 20GB+ VRAM  
**Status:** Ready to start immediately

---

## Executive Summary

### What We're Building
```
SafeTensors → GPU Backend → 300-500 t/s sustained throughput

Milestones:
  Day 1-2:  Base GPU backend (100-150 t/s)
  Day 3:    Deep optimizations (150-200 t/s)
  Day 4:    Quantization (200-300 t/s)
  Day 5:    Smart batching (300-400 t/s)
  Day 6:    Speculative decoding (400-500 t/s)
  Day 7:    Validation & tuning (sustained 300-500 t/s)
```

### Why This Is Possible
- GPU compute: ~100+ TFLOPS
- Memory bandwidth: 1+ TB/s
- LLM math: 11B FLOPs per token
- Theoretical max: 227 t/s (with perfect optimization)
- **We're targeting 300-500 t/s because batching + quantization push us beyond single-token limits**

---

## Day-by-Day Breakdown

## DAY 1: GPU Backend Foundation (8 hours)

### 1.1 Project Setup (1 hour)
```bash
# 1. Add burn-rs to Cargo.toml
[dependencies]
burn = { version = "0.13", features = [
    "backend-ndarray",
    "backend-wgpu",      # Cross-platform GPU
    "backend-cuda",      # NVIDIA (optional but recommended)
    "backend-metal",     # Apple Metal
] }

# 2. Create new module structure
mkdir -p src/inference/gpu/
touch src/inference/gpu/mod.rs
touch src/inference/gpu/backend.rs
touch src/inference/gpu/models.rs
touch src/inference/gpu/kernels.rs
touch src/inference/gpu/kv_cache.rs
```

### 1.2 SafeTensors Weight Loader (2 hours)
```rust
// File: src/inference/gpu/loader.rs

use safetensors::SafeTensors;
use burn::tensor::Tensor;

pub struct SafeTensorsLoader;

impl SafeTensorsLoader {
    pub fn load_weights<B: Backend>(
        path: &Path,
        device: &B::Device,
    ) -> Result<TransformerWeights<B>> {
        let buffer = std::fs::read(path)?;
        let st = SafeTensors::deserialize(&buffer)?;
        
        let mut weights = TransformerWeights::new();
        
        // Load embedding
        weights.embedding = Self::load_tensor(&st, "model.embed_tokens.weight", device)?;
        
        // Load all layers
        for layer_idx in 0..32 {  // TinyLlama has 22 layers, adjust for model
            let prefix = format!("model.layers.{}", layer_idx);
            
            weights.layers.push(LayerWeights {
                q_proj: Self::load_tensor(&st, &format!("{}.self_attn.q_proj.weight", prefix), device)?,
                k_proj: Self::load_tensor(&st, &format!("{}.self_attn.k_proj.weight", prefix), device)?,
                v_proj: Self::load_tensor(&st, &format!("{}.self_attn.v_proj.weight", prefix), device)?,
                o_proj: Self::load_tensor(&st, &format!("{}.self_attn.o_proj.weight", prefix), device)?,
                gate_proj: Self::load_tensor(&st, &format!("{}.mlp.gate_proj.weight", prefix), device)?,
                up_proj: Self::load_tensor(&st, &format!("{}.mlp.up_proj.weight", prefix), device)?,
                down_proj: Self::load_tensor(&st, &format!("{}.mlp.down_proj.weight", prefix), device)?,
                attn_norm: Self::load_tensor(&st, &format!("{}.input_layernorm.weight", prefix), device)?,
                ffn_norm: Self::load_tensor(&st, &format!("{}.post_attention_layernorm.weight", prefix), device)?,
            });
        }
        
        weights.norm = Self::load_tensor(&st, "model.norm.weight", device)?;
        weights.lm_head = Self::load_tensor(&st, "lm_head.weight", device)?;
        
        Ok(weights)
    }
    
    fn load_tensor<B: Backend>(
        st: &SafeTensors,
        name: &str,
        device: &B::Device,
    ) -> Result<Tensor<B, 2>> {
        let view = st.tensor(name)?;
        let shape = view.shape();
        let data: Vec<f32> = view.data().iter().copied().collect();
        
        Tensor::from_floats(data, device).reshape(shape)
    }
}
```

### 1.3 GPU Transformer Layers (3 hours)
```rust
// File: src/inference/gpu/models.rs

use burn::module::Module;
use burn::tensor::Tensor;

#[derive(Module)]
pub struct TransformerLayer<B: Backend> {
    pub self_attn: MultiHeadAttention<B>,
    pub mlp: FeedForward<B>,
    pub norm1: RMSNorm<B>,
    pub norm2: RMSNorm<B>,
}

impl<B: Backend> TransformerLayer<B> {
    pub fn forward(
        &self,
        hidden: Tensor<B, 3>,  // (batch, seq_len, hidden)
        kv_cache: &mut Option<KVCache<B>>,
    ) -> Tensor<B, 3> {
        // Pre-norm architecture (like LLaMA/Mistral)
        
        // Attention block
        let norm_hidden = self.norm1.forward(hidden.clone());
        let attn_out = self.self_attn.forward(norm_hidden, kv_cache);
        let hidden = hidden + attn_out;  // Residual
        
        // MLP block
        let norm_hidden = self.norm2.forward(hidden.clone());
        let mlp_out = self.mlp.forward(norm_hidden);
        hidden + mlp_out  // Residual
    }
}

#[derive(Module)]
pub struct MultiHeadAttention<B: Backend> {
    pub q_proj: Linear<B>,
    pub k_proj: Linear<B>,
    pub v_proj: Linear<B>,
    pub o_proj: Linear<B>,
    pub num_heads: usize,
}

impl<B: Backend> MultiHeadAttention<B> {
    pub fn forward(
        &self,
        hidden: Tensor<B, 3>,
        kv_cache: &mut Option<KVCache<B>>,
    ) -> Tensor<B, 3> {
        let batch_size = hidden.shape()[0];
        let seq_len = hidden.shape()[1];
        
        // Project to Q, K, V
        let q = self.q_proj.forward(hidden.clone());
        let k = self.k_proj.forward(hidden.clone());
        let v = self.v_proj.forward(hidden);
        
        // Reshape for multi-head
        let q = q.reshape([batch_size, seq_len, self.num_heads, -1]);
        let k = k.reshape([batch_size, seq_len, self.num_heads, -1]);
        let v = v.reshape([batch_size, seq_len, self.num_heads, -1]);
        
        // Update KV cache
        let (k, v) = if let Some(cache) = kv_cache {
            cache.append(k, v)
        } else {
            (k, v)
        };
        
        // Compute attention
        let scale = (q.shape()[-1] as f32).sqrt().recip();
        let scores = q.matmul(&k.transpose(-2, -1));
        let scores = scores * scale;
        let attn = scores.softmax(-1);
        let output = attn.matmul(&v);
        
        // Reshape back
        let output = output.reshape([batch_size, seq_len, -1]);
        
        self.o_proj.forward(output)
    }
}

#[derive(Module)]
pub struct FeedForward<B: Backend> {
    pub gate: Linear<B>,
    pub up: Linear<B>,
    pub down: Linear<B>,
}

impl<B: Backend> FeedForward<B> {
    pub fn forward(&self, hidden: Tensor<B, 3>) -> Tensor<B, 3> {
        let gate = self.gate.forward(hidden.clone()).silu();
        let up = self.up.forward(hidden);
        let combined = gate * up;  // Element-wise product (SwiGLU)
        self.down.forward(combined)
    }
}

pub struct RMSNorm<B: Backend> {
    pub weight: Tensor<B, 1>,
    pub eps: f32,
}

impl<B: Backend> RMSNorm<B> {
    pub fn forward(&self, hidden: Tensor<B, 3>) -> Tensor<B, 3> {
        let rms = (hidden.clone().pow(2.0).mean() + self.eps).sqrt();
        (hidden / rms) * self.weight
    }
}
```

### 1.4 KV Cache Implementation (2 hours)
```rust
// File: src/inference/gpu/kv_cache.rs

pub struct KVCache<B: Backend> {
    pub k: Tensor<B, 3>,  // (batch, past_seq_len, hidden)
    pub v: Tensor<B, 3>,
}

impl<B: Backend> KVCache<B> {
    pub fn new() -> Self {
        Self {
            k: Tensor::zeros([0, 0, 0]),
            v: Tensor::zeros([0, 0, 0]),
        }
    }
    
    pub fn append(
        &mut self,
        new_k: Tensor<B, 3>,
        new_v: Tensor<B, 3>,
    ) -> (Tensor<B, 3>, Tensor<B, 3>) {
        // Concatenate new tokens to cache
        self.k = Tensor::cat([self.k.clone(), new_k], 1);
        self.v = Tensor::cat([self.v.clone(), new_v], 1);
        
        (self.k.clone(), self.v.clone())
    }
    
    pub fn reset(&mut self) {
        self.k = Tensor::zeros([0, 0, 0]);
        self.v = Tensor::zeros([0, 0, 0]);
    }
}
```

### 1.5 Main Backend (2 hours)
```rust
// File: src/inference/gpu/backend.rs

pub struct GPUSafeTensorsBackend<B: Backend> {
    device: B::Device,
    weights: TransformerWeights<B>,
    config: ModelConfig,
    tokenizer: LLaMATokenizer,
}

impl<B: Backend> GPUSafeTensorsBackend<B> {
    pub fn new(
        weights_path: &Path,
        config_path: &Path,
    ) -> Result<Self> {
        let config = ModelConfig::from_file(config_path)?;
        let weights = SafeTensorsLoader::load_weights(weights_path, &device)?;
        let tokenizer = LLaMATokenizer::new()?;
        
        Ok(Self {
            device,
            weights,
            config,
            tokenizer,
        })
    }
    
    pub async fn generate(
        &self,
        prompt: &str,
        params: GenerationParams,
    ) -> Result<String> {
        let tokens = self.tokenizer.encode(prompt)?;
        let tokens = Tensor::from_ints(tokens, &self.device);
        
        let mut output = Vec::new();
        let mut kv_cache = KVCache::new();
        
        // Forward pass for prompt (prefill)
        let mut hidden = self.forward_with_kv(&tokens, &mut kv_cache)?;
        
        // Generate tokens
        for _ in 0..params.max_tokens {
            // Get last token logits
            let logits = self.get_logits(&hidden)?;
            
            // Sample next token
            let next_token = self.sample(&logits, params.temperature, params.top_p)?;
            output.push(next_token);
            
            // Early stopping
            if next_token == self.tokenizer.eos_token() {
                break;
            }
            
            // Forward for next token (decode)
            let token_tensor = Tensor::from_ints(vec![next_token], &self.device);
            hidden = self.forward_with_kv(&token_tensor, &mut kv_cache)?;
        }
        
        Ok(self.tokenizer.decode(&output)?)
    }
    
    fn forward_with_kv(
        &self,
        tokens: &Tensor<B, 2>,
        kv_cache: &mut KVCache<B>,
    ) -> Result<Tensor<B, 3>> {
        // Embedding
        let mut hidden = self.weights.embedding.index_select(tokens, 0)?;
        
        // Transformer layers
        for layer in &self.weights.layers {
            hidden = layer.forward(hidden, &mut Some(kv_cache))?;
        }
        
        // Final norm
        hidden = self.weights.norm.forward(hidden)?;
        
        Ok(hidden)
    }
}

impl<B: Backend> InferenceBackend for GPUSafeTensorsBackend<B> {
    fn load_model(&mut self, path: &Path, n_ctx: usize) -> MinervaResult<()> {
        *self = Self::new(path, &self.config_path)?;
        Ok(())
    }
    
    fn generate(&self, prompt: &str, params: GenerationParams) -> MinervaResult<String> {
        let rt = tokio::runtime::Handle::current();
        rt.block_on(self.generate(prompt, params))
    }
}
```

### Expected Performance After Day 1
```
TinyLlama-1.1B: 50-100 t/s (baseline GPU implementation)
Status: ✅ Working, but not optimized
```

---

## DAY 2-3: Deep Optimizations (8 hours each)

### 2.1 Flash Attention Implementation (3 hours)

[Flash Attention implementation with tiling and memory efficiency]

**Expected boost:** 1.5-2x on attention (30% of compute)  
**Running total:** 75-150 t/s

### 2.2 Operator Fusion (3 hours)

[RMSNorm + Linear fusion, Attention + Output projection fusion]

**Expected boost:** 1.2-1.5x on forward pass  
**Running total:** 100-200 t/s

### 2.3 Memory Optimization (2 hours)

```rust
// Use pinned memory for faster transfers
let pinned_buffer = CudaStream::allocate_pinned(&weights)?;

// Overlap computation and communication
unsafe {
    cuda_memcpy_async(...);
    compute_kernel(...);
}

// Reduce intermediate tensor allocations
let mut workspace = Workspace::new(10 * 1024 * 1024);  // 10MB pool
```

**Expected boost:** 1.1-1.2x  
**Running total:** 120-200 t/s

---

## DAY 4: Quantization (8 hours)

### 4.1 INT8 Quantization (3 hours)

```rust
pub fn quantize_weights_int8(weights: &Tensor) -> (Tensor, f32) {
    let max_val = weights.abs().max().item();
    let scale = 127.0 / max_val;
    let quantized = (weights * scale).cast::<i8>();
    (quantized, scale)
}

pub fn forward_with_int8(
    input: Tensor,
    weights_int8: Tensor,
    scale: f32,
) -> Tensor {
    let weights = weights_int8.cast::<f32>() / scale;
    input.matmul(&weights)
}
```

**Benefits:**
- 4x memory reduction
- 2x bandwidth reduction
- Minimal accuracy loss (<1%)

**Expected boost:** 1.5-2x  
**Running total:** 200-300 t/s

### 4.2 Mixed Precision (3 hours)

```rust
// Keep computations in FP32 for stability
let hidden = hidden.cast::<f32>();

// But store intermediate results in FP16 to save memory
let hidden_fp16 = hidden.cast::<f16>();
unsafe_save_memory(&hidden_fp16)?;

// Load back and cast to FP32 for next compute
let hidden = hidden_fp16.cast::<f32>();
```

**Expected boost:** 1.1x (marginal, but helps memory)  
**Running total:** 220-330 t/s

### 4.3 Calibration & Validation (2 hours)

Test accuracy loss, find optimal quantization points

---

## DAY 5: Smart Batching (8 hours)

### 5.1 Request Queue & Batching Engine (3 hours)

```rust
pub struct BatchingEngine<B: Backend> {
    queue: Arc<Mutex<VecDeque<Request>>>,
    batch_size: usize,
    timeout: Duration,
}

impl<B: Backend> BatchingEngine<B> {
    pub async fn process_batch(&self) -> Result<Vec<Response>> {
        // Wait up to timeout for batch_size requests
        let requests = self.queue_requests(self.timeout).await?;
        
        // Pad sequences
        let max_len = requests.iter().map(|r| r.tokens.len()).max().unwrap();
        let padded = requests.iter()
            .map(|r| pad_sequence(&r.tokens, max_len))
            .collect::<Vec<_>>();
        
        // Batch forward pass
        let batch_tokens = Tensor::stack(padded);
        let batch_output = self.forward_batch(&batch_tokens)?;
        
        // Split results
        let results = batch_output.unbind(0);
        
        Ok(requests.into_iter().zip(results)
            .map(|(req, output)| Response {
                request_id: req.id,
                text: self.decode(&output)?,
            })
            .collect())
    }
}
```

**Expected boost:** 4-6x for batch=8  
**Running total:** 900-1800 t/s for batch=8 (400-500 t/s for batch=2)

### 5.2 Dynamic Batching (3 hours)

```rust
// Adaptive batch sizing based on memory
fn calculate_optimal_batch_size(
    gpu_memory_available: usize,
    sequence_length: usize,
    model_size: usize,
) -> usize {
    let per_token_memory = sequence_length * model_size * 2;  // Approximate
    let available_for_batch = gpu_memory_available / 2;  // Leave 50% for weights
    (available_for_batch / per_token_memory).max(1)
}

// Process what's available, don't wait
while !request_queue.is_empty() && requests.len() < optimal_batch_size {
    if let Some(req) = request_queue.try_pop() {
        requests.push(req);
    } else {
        break;  // Don't wait, process with smaller batch
    }
}
```

**Expected boost:** 1.1-1.2x (improves responsiveness)  
**Running total:** 450-550 t/s with variable batch

### 5.3 Load Balancing (2 hours)

```rust
// Track latency per request
// Prioritize shorter sequences in batch
// Implement request priorities

pub struct RequestPriority {
    id: String,
    priority: u32,
    max_tokens: usize,
}

// Sort by max_tokens before batching
requests.sort_by_key(|r| r.max_tokens);
```

---

## DAY 6: Speculative Decoding (8 hours)

### 6.1 Draft Model Setup (2 hours)

```rust
pub struct SpeculativeDecoding<B: Backend> {
    main_model: GPUSafeTensorsBackend<B>,
    draft_model: PhiSmallModel<B>,  // Phi-2 or TinyLlama variant
}

impl<B: Backend> SpeculativeDecoding<B> {
    pub async fn generate_with_speculation(
        &self,
        prompt: &str,
        n_tokens: usize,
    ) -> Result<String> {
        let tokens = self.tokenizer.encode(prompt)?;
        let mut output = tokens.clone();
        
        for _ in 0..n_tokens {
            // Use draft model to predict next 4 tokens
            let draft_tokens = self.draft_model.predict_tokens(&output, 4)?;
            
            // Verify with main model in parallel
            let verified = self.verify_tokens(&draft_tokens)?;
            
            // Accept as many correct as possible
            output.extend(&verified.correct);
            
            // If diverged, use main model's prediction
            if let Some(correction) = verified.correction {
                output.push(correction);
            }
        }
        
        Ok(self.tokenizer.decode(&output)?)
    }
    
    fn verify_tokens(&self, draft: &[u32]) -> Result<VerificationResult> {
        // Batch verify with main model
        let verified = self.main_model.forward_batch(draft)?;
        
        // Check which tokens match
        let correct = draft.iter().zip(&verified)
            .take_while(|(d, v)| d == v)
            .map(|(d, _)| *d)
            .collect();
        
        let correction = verified.get(draft.len()).copied();
        
        Ok(VerificationResult { correct, correction })
    }
}
```

**Expected boost:** 2-3x (generate multiple tokens per forward pass)  
**Running total:** 600-900 t/s with speculative decoding

### 6.2 Uncertainty Estimation (2 hours)

```rust
// Know when to speculate vs. just compute
fn should_speculate(logits: &Tensor) -> bool {
    let probs = logits.softmax(-1);
    let top_prob = probs.max().item();
    
    top_prob > 0.7  // High confidence = safe to speculate
}

// Adjust speculation depth based on confidence
let speculation_depth = if top_prob > 0.9 {
    8  // Very confident, predict 8 tokens
} else if top_prob > 0.7 {
    4  // Confident, predict 4 tokens
} else {
    1  // Not confident, don't speculate
};
```

### 6.3 Integration & Testing (4 hours)

```rust
#[tokio::test]
async fn test_speculative_matches_exact() {
    // Ensure speculative decoding produces same output
    let exact = main_model.generate("test", 100)?;
    let speculative = spec_model.generate("test", 100)?;
    assert_eq!(exact, speculative);
}

#[test]
fn test_speculation_speedup() {
    let baseline = measure_latency(&main_model, 100)?;
    let speculative = measure_latency(&spec_model, 100)?;
    
    assert!(speculative < baseline);
    println!("Speedup: {:.1}x", baseline / speculative);
}
```

**Expected boost:** 1.5-2x on top of batching  
**Running total:** 700-1000 t/s

---

## DAY 7: Validation & Tuning (8 hours)

### 7.1 Comprehensive Benchmarking (2 hours)

```bash
#!/bin/bash
# Benchmark all configurations

echo "=== TinyLlama-1.1B Benchmarks ==="

echo "1. Single request:"
./real-benchmark --format safetensors-gpu --runs 10 \
  --output results-single.csv

echo "2. Batch=2:"
./batch-benchmark --batch 2 --runs 10 \
  --output results-batch2.csv

echo "3. Batch=4:"
./batch-benchmark --batch 4 --runs 10 \
  --output results-batch4.csv

echo "4. Batch=8:"
./batch-benchmark --batch 8 --runs 10 \
  --output results-batch8.csv

echo "5. Speculative decoding:"
./speculative-benchmark --runs 10 \
  --output results-speculative.csv

# Analyze
python3 analyze_benchmarks.py results-*.csv
```

### 7.2 Performance Profiling (2 hours)

```bash
# Identify bottlenecks
cargo flamegraph --bin real-benchmark

# Memory usage
valgrind --tool=massif ./real-benchmark

# GPU profiling (NVIDIA)
nsys profile ./real-benchmark

# GPU profiling (Apple)
Instruments -t "Metal System Trace" ./real-benchmark
```

### 7.3 Edge Case Testing (2 hours)

```rust
#[test]
fn test_empty_prompt() {
    // Edge case: empty input
    assert!(backend.generate("", params).is_ok());
}

#[test]
fn test_very_long_sequence() {
    let long_prompt = "x".repeat(2048);
    let result = backend.generate(&long_prompt, params)?;
    assert!(!result.is_empty());
}

#[test]
fn test_batch_different_lengths() {
    // Batching sequences of different lengths
    let requests = vec![
        Request { text: "short", max_tokens: 10 },
        Request { text: "a very long prompt that...", max_tokens: 100 },
        Request { text: "x", max_tokens: 1 },
    ];
    
    let results = backend.batch_generate(requests)?;
    assert_eq!(results.len(), 3);
}

#[test]
fn test_memory_leak() {
    // Generate many tokens, check memory doesn't grow
    for _ in 0..1000 {
        backend.generate("test", params)?;
    }
    // Memory should be stable
}
```

### 7.4 Tuning & Optimization (2 hours)

```rust
// Find optimal parameters
struct TuningParams {
    kv_cache_size: usize,
    batch_size: usize,
    speculation_depth: usize,
    quantization: bool,
    fusion_enabled: bool,
}

fn find_optimal_params(hardware: &HardwareSpec) -> TuningParams {
    let available_memory = hardware.gpu_memory * 0.8;  // Leave 20% margin
    let batch_size = (available_memory / model_memory_per_batch).max(1);
    
    TuningParams {
        batch_size: batch_size.min(8),
        speculation_depth: if batch_size >= 4 { 4 } else { 1 },
        quantization: available_memory < 16 * 1024 * 1024 * 1024,  // <16GB
        fusion_enabled: true,
        kv_cache_size: batch_size * 2048,
    }
}
```

---

## Expected Performance Progression

```
Day 1: GPU Backend
  TinyLlama: 50-100 t/s
  
Day 2: Flash Attention
  TinyLlama: 75-150 t/s (+50%)
  
Day 3: Operator Fusion
  TinyLlama: 100-200 t/s (+33%)
  
Day 4: INT8 Quantization
  TinyLlama: 150-300 t/s (+50%)
  
Day 5: Batching (batch=4)
  TinyLlama: 300-600 t/s (+100%, amortized per request)
  
Day 6: Speculative Decoding
  TinyLlama: 400-900 t/s (+50%)
  
Day 7: Validation & Tuning
  TinyLlama: 300-500 t/s sustained (realistic, accounting for overhead)
```

---

## Final Architecture

```
Request Input
    ↓
RequestQueue
    ↓
DynamicBatcher (batch 1-8 based on load)
    ↓
SpeculativeDecoding (draft + verify)
    ├─ DraftModel (Phi-2, fast)
    ├─ MainModel (Mistral-7B, accurate)
    └─ KVCache (persistent across tokens)
    ↓
GPUSafeTensorsBackend
    ├─ Embedding
    ├─ TransformerLayers (fused)
    │  ├─ RMSNormMatmul (fused)
    │  ├─ FlashAttention (tiled, with KV cache)
    │  ├─ SiLUMatmul (fused)
    │  └─ Residual
    ├─ RMSNorm
    └─ LinearOutput
    ↓
TokenSampler (temperature + top-p)
    ↓
ResponseQueue
    ↓
Output (300-500 t/s sustained)
```

---

## Key Metrics to Track

### Per Request
- TTFT (Time to First Token): <100ms
- TpT (Time per Token): <7ms
- Quality: Accuracy maintained

### Per Batch
- Throughput: 300-500 t/s
- GPU Utilization: >90%
- Memory Utilization: <90%

### Overall
- Sustained throughput: 300-500 t/s
- Request latency: <1s for 50-token response
- Error rate: <0.1%

---

## Contingency Plans

### If KV Cache doesn't improve:
- Might be implementation issue
- Fall back to basic attention
- Still get 2-3x from other optimizations

### If Quantization causes accuracy issues:
- Use mixed precision (FP16 activations)
- Or reduce quantization from INT8 to INT16
- Still get 1.5-2x improvement

### If Speculative Decoding is complex:
- Skip for Phase 4
- Keeps throughput at 300-400 t/s
- Can add later in Phase 4.5

### If Batching isn't beneficial:
- Most systems do benefit
- If not, focus on single-request optimization
- Still achieve 150-200 t/s

---

## Success Criteria

✅ **Day 1:** GPU backend works (50-100 t/s)  
✅ **Day 3:** Optimizations work (150-200 t/s)  
✅ **Day 4:** Quantization works (200-300 t/s)  
✅ **Day 5:** Batching works (300-400 t/s with batch=4)  
✅ **Day 6:** Speculative decoding works (400-500 t/s)  
✅ **Day 7:** Sustained 300-500 t/s with validation  

**Final Target:** 300-500 t/s sustained throughput

---

## Commit Schedule

- **Day 1:** `feat: GPU backend with KV cache`
- **Day 3:** `feat: Flash Attention + operator fusion`
- **Day 4:** `feat: INT8 quantization + mixed precision`
- **Day 5:** `feat: Dynamic request batching`
- **Day 6:** `feat: Speculative decoding`
- **Day 7:** `docs: comprehensive benchmarking results`

---

## Comparison with Industry

```
ollama:             50-80 t/s (single GPU)
llama.cpp:          30-60 t/s (single GPU)
vLLM:               200-400 t/s (with batching)

Our target (Tier 4): 300-500 t/s sustained

We'll be competitive with vLLM!
```

---

## Resources Needed

- **GPU:** RTX 4090 or equivalent (20GB+ VRAM)
- **RAM:** 32GB+ system RAM
- **Storage:** 50GB (models + code)
- **Time:** 50-55 hours over 6-7 days
- **Dependencies:** burn-rs, safetensors, tokio

---

## Let's Do This! 🚀

**Phase 4: Aggressive Optimization**
- 6-7 days
- 300-500 t/s target
- Production quality
- Comprehensive benchmarking
- Ready to ship

**Start Date:** Whenever you're ready
**Estimated Completion:** 6-7 days from start
**Expected Outcome:** Industry-competitive performance

