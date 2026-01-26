# Quick Real Model Testing Guide

Download small models for rapid testing of all 3 backends.

## Quick Start (5 minutes)

### Option 1: GGUF Only (Fastest)
```bash
mkdir -p models/tinyllama-1.1b-gguf
curl -L -o models/tinyllama-1.1b-gguf/TinyLlama-1.1B-Chat-v1.0.Q4_K_M.gguf \
  "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"

# Verify
cd src-tauri
cargo run --release --bin verify-gguf -- --model-path ../models/tinyllama-1.1b-gguf/TinyLlama-1.1B-Chat-v1.0.Q4_K_M.gguf
```

Expected: ✓ VERIFICATION PASSED

### Option 2: All Three Formats (20 minutes)
```bash
./download-test-models.sh tinyllama

# Verify all three
cd src-tauri
cargo run --release --bin verify-gguf -- --model-path ../models/tinyllama-1.1b-gguf/TinyLlama-1.1B-Chat-v1.0.Q4_K_M.gguf
cargo run --release --bin verify-safetensors -- --model-path ../models/tinyllama-1.1b-safetensors/model.safetensors
cargo run --release --bin verify-mlx -- --model-path ../models/tinyllama-1.1b-mlx/model.safetensors
```

## Model Sizes & Download Times

### TinyLlama-1.1B (Recommended for Testing)
| Format | Size | Time | Link |
|--------|------|------|------|
| GGUF Q4 | 640MB | 5 min | TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF |
| SafeTensors | 2.2GB | 15 min | TinyLlama/TinyLlama-1.1B-Chat-v1.0 |
| MLX | 2.2GB | 15 min | mlx-community/TinyLlama-1.1B-Chat-v1.0 |
| **Total** | **~5GB** | **~20 min** | |

### Phi-2 (3.3B, if you want something bigger)
| Format | Size | Time | Link |
|--------|------|------|------|
| GGUF Q4 | 2GB | 15 min | TheBloke/phi-2-GGUF |
| SafeTensors | 4.8GB | 30 min | microsoft/phi-2 |
| MLX | 4.8GB | 30 min | mlx-community/phi-2 |
| **Total** | **~11GB** | **~75 min** | |

## Run Tests After Download

### Verify Models Load
```bash
cd src-tauri

# Test GGUF
cargo run --release --bin verify-gguf -- \
  --model-path ../models/tinyllama-1.1b-gguf/TinyLlama-1.1B-Chat-v1.0.Q4_K_M.gguf \
  --verbose

# Test SafeTensors
cargo run --release --bin verify-safetensors -- \
  --model-path ../models/tinyllama-1.1b-safetensors/model.safetensors \
  --config-path ../models/tinyllama-1.1b-safetensors/config.json \
  --verbose

# Test MLX
cargo run --release --bin verify-mlx -- \
  --model-path ../models/tinyllama-1.1b-mlx/model.safetensors \
  --config-path ../models/tinyllama-1.1b-mlx/config.json \
  --verbose
```

### Run Actual Benchmarks
```bash
cd src-tauri

# GGUF benchmark (use actual llama_cpp_backend)
cargo run --release --bin minerva-bench -- \
  --format gguf --runs 5 --output ../TINYLLAMA_GGUF_REAL.csv

# SafeTensors benchmark (pure_rust_backend)
cargo run --release --bin minerva-bench -- \
  --format safetensors --runs 5 --output ../TINYLLAMA_SAFETENSORS_REAL.csv

# MLX benchmark (mlx_backend)
cargo run --release --bin minerva-bench -- \
  --format mlx --runs 5 --output ../TINYLLAMA_MLX_REAL.csv
```

## Expected Real Results

With actual model inference (not synthetic), expect:

### TinyLlama-1.1B (1.1B parameters)
- **GGUF:** 50-100 t/s (GPU quantized)
- **SafeTensors:** 10-20 t/s (CPU, will use synthetic until fixed)
- **MLX:** 50-100 t/s (Apple Silicon optimized)

*Note: Much faster than Mistral-7B due to 6x smaller model*

### Phi-2 (3.3B parameters)
- **GGUF:** 30-60 t/s 
- **SafeTensors:** 5-10 t/s
- **MLX:** 30-60 t/s

## Troubleshooting

### Download Stuck?
Add resume flag to continue:
```bash
curl -L -C - -o models/tinyllama-1.1b-gguf/TinyLlama-1.1B-Chat-v1.0.Q4_K_M.gguf ...
#          ^ resume from byte position
```

### Model Verification Fails?
Check file size (should not be 29 bytes):
```bash
ls -lh models/tinyllama-1.1b-gguf/
# Should show ~640MB, not 29B
```

### Inference Fails?
- Ensure all required files exist (model + config + tokenizer)
- Check file wasn't corrupted during download
- Verify model format matches backend

## Compare with Mock Results

Previous mock benchmarks (synthetic computation):
- GGUF: 38.4 t/s
- SafeTensors: 22.1 t/s
- MLX: 22.1 t/s

Real results will show:
- ✅ GGUF similar or better (uses real weights)
- ⚠️ SafeTensors much slower (real weights, CPU-bound)
- ✅ MLX faster with Metal optimization

## Next Steps After Testing

1. Compare mock vs real results
2. Identify which backend performs best for your use case
3. Plan Phase 4 optimizations based on actual bottlenecks
4. Consider SafeTensors fix (synthetic → real weights)

---

**Time to test:** 20-30 minutes total
**Disk space:** 5GB for TinyLlama, 11GB for Phi-2
**Result:** Real performance data for optimization planning
