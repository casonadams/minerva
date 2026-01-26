# Rust MLX Implementation: Step-by-Step Guide

**Status:** Ready to build native Rust MLX  
**Timeline:** 14-20 hours for complete implementation  
**Target Performance:** 200-300 t/s throughput  

---

## Before You Start: Prerequisites

### What You Need
- Rust 1.70+ (we have this)
- macOS 12+ (we have this)
- 16GB RAM minimum (we have this)
- ~2 hours uninterrupted for Phase 1

### Crates to Add to Cargo.toml
```toml
[dependencies]
# Existing (reuse)
ndarray = "0.15"
rand = "0.8"

# New (for MLX)
safetensors = "0.3"
serde_json = "1.0"
tokio = { version = "1.0", features = ["full"] }

# Metal (Phase 5 only)
metal = "0.27"  # Rust bindings for Metal
metalbuild = "0.1"  # Build script for shaders
```

### Directory Structure to Create
```bash
mkdir -p src-tauri/src/inference/mlx_native
mkdir -p shaders

# Create files
touch src-tauri/src/inference/mlx_native/mod.rs
touch src-tauri/src/inference/mlx_native/loader.rs
touch src-tauri/src/inference/mlx_native/config.rs
touch src-tauri/src/inference/mlx_native/unified_memory.rs
touch src-tauri/src/inference/mlx_native/kv_quantization.rs
touch src-tauri/src/inference/mlx_native/compute_graph.rs
touch src-tauri/src/inference/mlx_native/metal_backend.rs
touch shaders/kernels.metal
```

---

## Phase 1: MLX Model Loader (2-3 hours)

**Goal:** Load SafeTensors models from disk

### Step 1.1: Create Module Structure

**File:** `src-tauri/src/inference/mlx_native/mod.rs`

```rust
pub mod loader;
pub mod config;
pub mod unified_memory;
pub mod kv_quantization;
pub mod compute_graph;
pub mod metal_backend;

pub use loader::{MLXModel, load_mlx_model};
pub use config::GPTOSSConfig;
```

**File:** `src-tauri/src/inference/mlx_native/config.rs`

```rust
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GPTOSSConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub max_position_embeddings: usize,
    pub rope_theta: f32,
    pub initializer_range: f32,
    pub rms_norm_eps: f32,
    pub use_cache: bool,
}

impl Default for GPTOSSConfig {
    fn default() -> Self {
        Self {
            vocab_size: 201088,
            hidden_size: 2880,
            intermediate_size: 7168,
            num_hidden_layers: 24,
            num_attention_heads: 64,
            num_key_value_heads: 8,
            max_position_embeddings: 4096,
            rope_theta: 10000.0,
            initializer_range: 0.02,
            rms_norm_eps: 1e-6,
            use_cache: true,
        }
    }
}
```

### Step 1.2: Implement Model Loader

**File:** `src-tauri/src/inference/mlx_native/loader.rs`

```rust
use std::path::Path;
use ndarray::{Array1, Array2};
use safetensors::SafeTensors;
use serde_json::json;
use crate::error::{MinervaError, MinervaResult};
use super::config::GPTOSSConfig;

#[derive(Debug, Clone)]
pub struct MLXLayerWeights {
    pub attn_q: Array2<f32>,
    pub attn_k: Array2<f32>,
    pub attn_v: Array2<f32>,
    pub attn_out: Array2<f32>,
    pub mlp_gate: Array2<f32>,
    pub mlp_up: Array2<f32>,
    pub mlp_down: Array2<f32>,
    pub norm_attn: Array1<f32>,
    pub norm_mlp: Array1<f32>,
}

#[derive(Debug, Clone)]
pub struct MLXModel {
    pub embedding: Array2<f32>,
    pub lm_head: Array2<f32>,
    pub layers: Vec<MLXLayerWeights>,
    pub norm_final: Array1<f32>,
    pub config: GPTOSSConfig,
}

pub fn load_mlx_model(path: &Path) -> MinervaResult<MLXModel> {
    // 1. Load all SafeTensors files
    let weights = load_safetensors(path)?;
    
    // 2. Extract embedding
    let embedding = extract_tensor_2d(&weights, "model.embed_tokens.weight")?;
    
    // 3. Extract LM head
    let lm_head = extract_tensor_2d(&weights, "lm_head.weight")?;
    
    // 4. Extract final norm
    let norm_final = extract_tensor_1d(&weights, "model.norm.weight")?;
    
    // 5. Extract per-layer weights
    let mut layers = Vec::new();
    for layer_idx in 0..24 {
        let layer_weights = MLXLayerWeights {
            attn_q: extract_tensor_2d(&weights, 
                &format!("model.layers.{}.self_attn.q_proj.weight", layer_idx))?,
            attn_k: extract_tensor_2d(&weights,
                &format!("model.layers.{}.self_attn.k_proj.weight", layer_idx))?,
            attn_v: extract_tensor_2d(&weights,
                &format!("model.layers.{}.self_attn.v_proj.weight", layer_idx))?,
            attn_out: extract_tensor_2d(&weights,
                &format!("model.layers.{}.self_attn.o_proj.weight", layer_idx))?,
            mlp_gate: extract_tensor_2d(&weights,
                &format!("model.layers.{}.mlp.gate_proj.weight", layer_idx))?,
            mlp_up: extract_tensor_2d(&weights,
                &format!("model.layers.{}.mlp.up_proj.weight", layer_idx))?,
            mlp_down: extract_tensor_2d(&weights,
                &format!("model.layers.{}.mlp.down_proj.weight", layer_idx))?,
            norm_attn: extract_tensor_1d(&weights,
                &format!("model.layers.{}.input_layernorm.weight", layer_idx))?,
            norm_mlp: extract_tensor_1d(&weights,
                &format!("model.layers.{}.post_attention_layernorm.weight", layer_idx))?,
        };
        layers.push(layer_weights);
    }
    
    Ok(MLXModel {
        embedding,
        lm_head,
        layers,
        norm_final,
        config: GPTOSSConfig::default(),
    })
}

fn load_safetensors(path: &Path) -> MinervaResult<SafeTensors> {
    // Load index.json to find shard files
    let index_path = path.join("model.safetensors.index.json");
    let index_content = std::fs::read_to_string(&index_path)
        .map_err(|e| MinervaError::IOError(format!("Failed to read index: {}", e)))?;
    let index: serde_json::Value = serde_json::from_str(&index_content)?;
    
    // Extract weight file mapping
    let weights_map = index["weight_map"].as_object()
        .ok_or_else(|| MinervaError::ParseError("Invalid index.json".into()))?;
    
    // Find unique shard files
    let mut shard_files = std::collections::HashSet::new();
    for shard in weights_map.values() {
        if let Some(s) = shard.as_str() {
            shard_files.insert(s.to_string());
        }
    }
    
    // Load all shards in parallel
    let mut all_tensors = std::collections::HashMap::new();
    let runtime = tokio::runtime::Runtime::new()?;
    
    runtime.block_on(async {
        let mut tasks = vec![];
        for shard_file in shard_files {
            let shard_path = path.join(&shard_file);
            tasks.push(tokio::spawn(async move {
                load_shard_file(&shard_path)
            }));
        }
        
        for task in tasks {
            let shard_tensors = task.await???;
            all_tensors.extend(shard_tensors);
        }
        
        Ok::<_, MinervaError>(())
    })?;
    
    // Combine into single SafeTensors object
    let data = safetensors::serialize(&all_tensors)?;
    let st = SafeTensors::deserialize(&data)?;
    Ok(st)
}

fn load_shard_file(path: &Path) -> MinervaResult<std::collections::HashMap<String, Vec<u8>>> {
    let bytes = std::fs::read(path)?;
    let st = SafeTensors::deserialize(&bytes)?;
    
    let mut result = std::collections::HashMap::new();
    for tensor_name in st.names() {
        let tensor = st.tensor(tensor_name)?;
        result.insert(tensor_name.to_string(), tensor.data().to_vec());
    }
    
    Ok(result)
}

fn extract_tensor_2d(weights: &SafeTensors, name: &str) -> MinervaResult<Array2<f32>> {
    let tensor = weights.tensor(name)
        .map_err(|_| MinervaError::ParseError(format!("Missing tensor: {}", name)))?;
    
    let data: Vec<f32> = tensor.data().chunks(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();
    
    let shape = tensor.shape();
    Array2::from_shape_vec((shape[0], shape[1]), data)
        .map_err(|_| MinervaError::ParseError("Invalid shape".into()))
}

fn extract_tensor_1d(weights: &SafeTensors, name: &str) -> MinervaResult<Array1<f32>> {
    let tensor = weights.tensor(name)
        .map_err(|_| MinervaError::ParseError(format!("Missing tensor: {}", name)))?;
    
    let data: Vec<f32> = tensor.data().chunks(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();
    
    Array1::from_vec(data)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    
    #[test]
    fn test_load_mlx_gpt_oss_20b() {
        let model_path = PathBuf::from(
            "~/.cache/mlx-models/gpt-oss-20b-MXFP4-Q8"
        ).canonicalize().unwrap();
        
        // Skip if model not available
        if !model_path.exists() {
            println!("Model not found at {:?}, skipping test", model_path);
            return;
        }
        
        let model = load_mlx_model(&model_path).expect("Failed to load model");
        
        // Verify structure
        assert_eq!(model.embedding.shape()[0], 201088);
        assert_eq!(model.embedding.shape()[1], 2880);
        assert_eq!(model.layers.len(), 24);
        assert_eq!(model.lm_head.shape()[0], 201088);
    }
}
```

### Step 1.3: Test Model Loading

```bash
cd src-tauri

# Add to Cargo.toml if not present
cargo add safetensors
cargo add serde_json

# Build
cargo build --release

# Test (will skip if model not available)
cargo test --lib inference::mlx_native::loader
```

### Expected Output
```
test inference::mlx_native::loader::tests::test_load_mlx_gpt_oss_20b ... ok

Loaded model in ~200ms
Embedding shape: (201088, 2880) ✓
Layers: 24 ✓
LM Head shape: (201088, 2880) ✓
```

### If Tests Fail

**Error: "Module 'safetensors' not found"**
```bash
cargo add safetensors@0.3
```

**Error: "Missing tensor"**
- Check model path exists
- Check model is actually gpt-oss-20b-MXFP4-Q8
- Print available tensor names for debugging

**Error: "Invalid shape"**
- Tensor might be quantized (needs dequant)
- Check safetensors metadata for dtype

---

## Phase 2: Unified Memory (1-2 hours)

**After Phase 1 works, start this.**

**File:** `src-tauri/src/inference/mlx_native/unified_memory.rs`

```rust
use std::sync::Arc;
use std::sync::Mutex;
use ndarray::{Array1, Array2};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Device {
    CPU,
    GPU,
}

#[derive(Debug, Clone)]
pub struct MLXArray {
    data: Arc<Mutex<Vec<f32>>>,
    shape: (usize, usize),
    device: Device,
}

impl MLXArray {
    pub fn new(shape: (usize, usize), data: Vec<f32>, device: Device) -> Self {
        MLXArray {
            data: Arc::new(Mutex::new(data)),
            shape,
            device,
        }
    }
    
    pub fn to_device(&self, target: Device) -> Self {
        if self.device == target {
            return self.clone();
        }
        
        let data = self.data.lock().unwrap().clone();
        MLXArray {
            data: Arc::new(Mutex::new(data)),
            shape: self.shape,
            device: target,
        }
    }
    
    pub fn shape(&self) -> (usize, usize) {
        self.shape
    }
    
    pub fn data(&self) -> Vec<f32> {
        self.data.lock().unwrap().clone()
    }
}
```

---

## Phase 3-5: Continue Implementation

*After Phase 1-2 work, implement remaining phases following the detailed specs in `MLX_RUST_NATIVE_STRATEGY.md`*

---

## Testing Strategy

### Test Each Phase

```bash
# Phase 1: Model loading
cargo test --lib inference::mlx_native::loader

# Phase 2: Unified memory
cargo test --lib inference::mlx_native::unified_memory

# Phase 3: KV quantization
cargo test --lib inference::mlx_native::kv_quantization

# Phase 4: Compute graphs
cargo test --lib inference::mlx_native::compute_graph

# Phase 5: Metal backend
cargo test --lib inference::mlx_native::metal_backend

# All MLX tests
cargo test --lib inference::mlx_native
```

---

## Integration with OpenAI API

After Phase 1 complete, wire to API:

**File:** `src-tauri/src/api/inference.rs`

```rust
use crate::inference::mlx_native::load_mlx_model;

pub async fn generate_with_mlx(
    prompt: String,
    max_tokens: usize,
) -> MinervaResult<String> {
    let model = load_mlx_model(Path::new("~/.cache/mlx-models/gpt-oss-20b-MXFP4-Q8"))?;
    
    // Tokenize prompt
    let input_ids = tokenize(&prompt);
    
    // Generate
    let mut generated = input_ids.clone();
    for _ in 0..max_tokens {
        let logits = model.forward(&generated)?;
        let next_token = sample_top_k(&logits);
        generated.push(next_token);
    }
    
    // Detokenize
    Ok(detokenize(&generated))
}
```

---

## Performance Benchmarking

After each phase, measure:

```bash
# Phase 1 benchmark
cargo run --release --bin mlx-phase1-benchmark

# Phase 2 benchmark
cargo run --release --bin mlx-phase2-benchmark

# etc...
```

---

## Success Checkpoints

### Phase 1 (Model Loading) ✓
- Model loads without errors
- Correct tensor shapes
- Memory usage ~12GB

### Phase 2 (Unified Memory) ✓
- Data transfers work
- GPU/CPU transparent to user
- No correctness loss

### Phase 3 (KV Quantization) ✓
- Quantization reduces size 8x
- Dequantization < 1% accuracy loss
- Memory usage for 8K context < 2GB

### Phase 4 (Compute Graphs) ✓
- Graph builds in < 100ms
- Optimization finds 2-5x speedups
- Correctness preserved

### Phase 5 (Metal GPU) ✓
- Metal kernels compile
- GPU operations 5-20x faster
- End-to-end speedup measured

---

## Git Workflow

After each phase:

```bash
# Phase 1
git add src-tauri/src/inference/mlx_native/loader.rs
git add src-tauri/src/inference/mlx_native/config.rs
git commit -m "feat(mlx): implement model loader for SafeTensors"

# Phase 2
git add src-tauri/src/inference/mlx_native/unified_memory.rs
git commit -m "feat(mlx): implement unified memory model"

# etc...
```

---

## Troubleshooting

### Build Errors

**Error: "Can't find safetensors"**
```bash
cargo add safetensors@0.3
cargo clean
cargo build --release
```

**Error: "Can't find tokio"**
```bash
cargo add tokio --features full
```

### Runtime Errors

**Error: "Model not found"**
- Download from HuggingFace: `mlx-community/gpt-oss-20b-MXFP4-Q8`
- Place in `~/.cache/mlx-models/`

**Error: "Out of memory"**
- Phase 1 alone uses ~12GB
- Need 16GB total for safety margin
- Phase 3 (quantization) reduces to ~2GB

**Error: "Tensor shape mismatch"**
- Check SafeTensors version matches model
- Print tensor metadata for debugging

---

## You're Ready!

Everything is prepared for Rust MLX implementation.

**Next steps:**
1. Create module structure (5 min)
2. Implement Phase 1 loader (2-3 hours)
3. Test thoroughly
4. Commit
5. Proceed to Phase 2

**Let's build the fastest MLX implementation on macOS!** 🚀

