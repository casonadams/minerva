# Real Model Testing Guide - Phase 3 with Actual Mistral 7B

This guide shows how to download real Mistral 7B models and run actual benchmarks (not mock data).

## Quick Start

### Option 1: Download All Models (Parallel, ~40GB total, 1-2 hours)

```bash
# Create download script
cat > download-models.sh << 'EOF'
#!/bin/bash
set -e

mkdir -p models/{mistral-7b-gguf,mistral-7b-safetensors,mistral-7b-mlx}

echo "Downloading GGUF Q4_K_M (~4.8GB)..."
curl -L -o models/mistral-7b-gguf/Mistral-7B.Q4_K_M.gguf \
  https://huggingface.co/TheBloke/Mistral-7B-GGUF/resolve/main/Mistral-7B.Q4_K_M.gguf &

echo "Downloading SafeTensors (~13GB)..."
curl -L -o models/mistral-7b-safetensors/model.safetensors \
  https://huggingface.co/mistralai/Mistral-7B/resolve/main/model.safetensors &

curl -L -o models/mistral-7b-safetensors/config.json \
  https://huggingface.co/mistralai/Mistral-7B/resolve/main/config.json &

curl -L -o models/mistral-7b-safetensors/tokenizer.json \
  https://huggingface.co/mistralai/Mistral-7B/resolve/main/tokenizer.json &

echo "Downloading MLX (~13GB)..."
curl -L -o models/mistral-7b-mlx/model.safetensors \
  https://huggingface.co/mlx-community/Mistral-7B/resolve/main/model.safetensors &

curl -L -o models/mistral-7b-mlx/config.json \
  https://huggingface.co/mlx-community/Mistral-7B/resolve/main/config.json &

curl -L -o models/mistral-7b-mlx/tokenizer.json \
  https://huggingface.co/mlx-community/Mistral-7B/resolve/main/tokenizer.json &

wait
echo "All downloads complete!"
ls -lh models/*/
EOF

chmod +x download-models.sh
./download-models.sh
```

### Option 2: Download One Format at a Time

**GGUF Only (fastest for testing, ~4.8GB):**
```bash
mkdir -p models/mistral-7b-gguf
curl -L -o models/mistral-7b-gguf/Mistral-7B.Q4_K_M.gguf \
  https://huggingface.co/TheBloke/Mistral-7B-GGUF/resolve/main/Mistral-7B.Q4_K_M.gguf
```

**SafeTensors Only (~13GB):**
```bash
mkdir -p models/mistral-7b-safetensors
curl -L -o models/mistral-7b-safetensors/model.safetensors \
  https://huggingface.co/mistralai/Mistral-7B/resolve/main/model.safetensors
curl -L -o models/mistral-7b-safetensors/config.json \
  https://huggingface.co/mistralai/Mistral-7B/resolve/main/config.json
curl -L -o models/mistral-7b-safetensors/tokenizer.json \
  https://huggingface.co/mistralai/Mistral-7B/resolve/main/tokenizer.json
```

**MLX Only (~13GB):**
```bash
mkdir -p models/mistral-7b-mlx
curl -L -o models/mistral-7b-mlx/model.safetensors \
  https://huggingface.co/mlx-community/Mistral-7B/resolve/main/model.safetensors
curl -L -o models/mistral-7b-mlx/config.json \
  https://huggingface.co/mlx-community/Mistral-7B/resolve/main/config.json
curl -L -o models/mistral-7b-mlx/tokenizer.json \
  https://huggingface.co/mlx-community/Mistral-7B/resolve/main/tokenizer.json
```

## Verify Models Downloaded Successfully

Once downloads are complete, verify with:

```bash
# Check GGUF
cd src-tauri
cargo run --release --bin verify-gguf -- --model-path ../models/mistral-7b-gguf/Mistral-7B.Q4_K_M.gguf

# Check SafeTensors
cargo run --release --bin verify-safetensors \
  --model-path ../models/mistral-7b-safetensors/model.safetensors \
  --config-path ../models/mistral-7b-safetensors/config.json

# Check MLX
cargo run --release --bin verify-mlx \
  --model-path ../models/mistral-7b-mlx/model.safetensors \
  --config-path ../models/mistral-7b-mlx/config.json
```

Expected output for all:
```
✓ File exists
✓ Valid [FORMAT] file detected
✓ VERIFICATION PASSED
```

## Test Individual Backends

### Test GGUF Backend (llama_cpp)

Once models are verified, you can test actual inference by updating the benchmark to use the real llama_cpp backend:

```bash
cd src-tauri

# Build real inference test
cargo test --lib inference::llama_cpp_backend -- --nocapture

# Or create integration test
cargo run --release --example test_gguf_inference
```

### Test SafeTensors Backend (Pure Rust)

```bash
cd src-tauri

# Load and test the SafeTensors model
cargo test --lib inference::pure_rust_backend -- --nocapture

# Or run inference
cargo run --release --example test_safetensors_inference
```

### Test MLX Backend (Apple Silicon)

```bash
cd src-tauri

# Load and test MLX model
cargo test --lib inference::mlx_backend -- --nocapture

# Or run inference
cargo run --release --example test_mlx_inference
```

## Real Benchmark Results

Once models are verified, update `minerva-bench` to use actual backends:

```bash
cd src-tauri

# Run GGUF benchmark (actual inference, no mocking)
cargo run --release --bin minerva-bench -- --format gguf --runs 3 --output ../REAL_GGUF_RESULTS.csv

# Run SafeTensors benchmark
cargo run --release --bin minerva-bench -- --format safetensors --runs 3 --output ../REAL_SAFETENSORS_RESULTS.csv

# Run MLX benchmark
cargo run --release --bin minerva-bench -- --format mlx --runs 3 --output ../REAL_MLX_RESULTS.csv
```

Results will include:
- **Real inference latency** (not simulated)
- **Actual memory usage**
- **True throughput** with optimization
- **Backend-specific behavior** (quantization effects, etc.)

## Expected Real Results vs Mock

### GGUF (llama_cpp with Quantization)
**Mock Baseline:** 38.4 t/s  
**Expected Real:** 10-20 t/s
- Model computation adds 50-70% overhead
- Memory bandwidth becomes bottleneck
- GPU utilization affects scaling

### SafeTensors (Pure Rust, CPU)
**Mock Baseline:** 22.1 t/s  
**Expected Real:** 2-5 t/s
- Large model computation dominates
- Memory bound on CPU
- Linear with token count

### MLX (Apple Silicon, Metal)
**Mock Baseline:** 22.1 t/s  
**Expected Real:** 10-30 t/s  
**With optimization:** 30-50 t/s
- Metal acceleration not fully utilized in baseline
- M3 Pro/Max can leverage GPU
- Batching will improve performance

### Mock (Reference)
**Result:** 32.3 t/s (unchanged)
- Baseline for comparison only

## Creating Integration Tests

Create `src-tauri/examples/test_gguf_inference.rs`:

```rust
use minerva::inference::llama_cpp_backend::LlamaCppBackend;
use minerva::inference::inference_backend_trait::InferenceBackend;
use std::path::Path;

fn main() {
    let mut backend = LlamaCppBackend::new();
    
    // Load real model
    match backend.load_model(Path::new("models/mistral-7b-gguf/Mistral-7B.Q4_K_M.gguf"), 512) {
        Ok(_) => println!("✓ Model loaded successfully"),
        Err(e) => {
            eprintln!("✗ Failed to load: {}", e);
            return;
        }
    }
    
    // Test inference
    let prompt = "What is machine learning?";
    let params = GenerationParams {
        max_tokens: 100,
        temperature: 0.7,
        top_p: 0.9,
    };
    
    match backend.generate(prompt, params) {
        Ok(response) => println!("Generated: {}", response),
        Err(e) => eprintln!("Inference failed: {}", e),
    }
}
```

## Troubleshooting

### Download Fails or Redirects

If curl redirects to an HTML page instead of downloading:

1. **Use wget instead:**
```bash
wget -O models/mistral-7b-gguf/Mistral-7B.Q4_K_M.gguf \
  "https://huggingface.co/TheBloke/Mistral-7B-GGUF/resolve/main/Mistral-7B.Q4_K_M.gguf"
```

2. **Add HuggingFace token:**
```bash
# First login
huggingface-cli login

# Then download
curl -H "Authorization: Bearer $(cat ~/.huggingface/token)" \
  -L -o models/mistral-7b-gguf/Mistral-7B.Q4_K_M.gguf \
  "https://huggingface.co/TheBloke/Mistral-7B-GGUF/resolve/main/Mistral-7B.Q4_K_M.gguf"
```

3. **Use LFS:**
```bash
git clone https://huggingface.co/TheBloke/Mistral-7B-GGUF
cd Mistral-7B-GGUF
git lfs install
git lfs pull --include="Mistral-7B.Q4_K_M.gguf"
```

### Verification Fails

Check file sizes:
```bash
# GGUF should be ~4.8GB
ls -lh models/mistral-7b-gguf/

# SafeTensors should be ~13GB
ls -lh models/mistral-7b-safetensors/model.safetensors
```

If files are <100MB, download failed and got HTML redirect.

### Backend Loading Fails

1. Ensure all required files exist (model + config + tokenizer)
2. Check file formats match backend expectations
3. Verify sufficient memory available for model
4. Check GPU support for acceleration backends

## Cleanup

To remove downloaded models and free up space:

```bash
# Remove just the models
rm -rf models/mistral-7b-*

# Or keep only one format
rm -rf models/mistral-7b-safetensors models/mistral-7b-mlx
```

## Next Steps

1. **Download models** using script above
2. **Verify with** verify-gguf, verify-safetensors, verify-mlx
3. **Run real benchmarks** with minerva-bench
4. **Compare results** against mock baselines
5. **Optimize** based on actual performance data

---

**Total Setup Time:** 1-2 hours (mostly download time)  
**Disk Space:** 40GB total (30GB for real models)  
**Expected Real Results:** Will show actual backend performance without simulation
