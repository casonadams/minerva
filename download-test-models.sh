#!/bin/bash
# Download small models for quick testing across all 3 formats
# TinyLlama-1.1B is ideal: 640MB GGUF, 2.2GB SafeTensors/MLX
# Total: ~5GB, ~20 minutes download time

set -e

MODEL=${1:-tinyllama}  # tinyllama or phi2
OUTPUT_DIR=${2:-./models}

echo "=========================================="
echo "Downloading Test Models"
echo "=========================================="
echo ""

mkdir -p "$OUTPUT_DIR"

if [ "$MODEL" = "tinyllama" ] || [ "$MODEL" = "all" ]; then
    echo "Downloading TinyLlama-1.1B (smallest, fastest)"
    echo ""
    
    # GGUF - TheBloke quantized version (~640MB)
    mkdir -p "$OUTPUT_DIR/tinyllama-1.1b-gguf"
    echo "1. TinyLlama GGUF Q4 (~640MB, ~5 min)..."
    curl -L -C - -o "$OUTPUT_DIR/tinyllama-1.1b-gguf/TinyLlama-1.1B-Chat-v1.0.Q4_K_M.gguf" \
        "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf" 2>&1 &
    GGUF_PID=$!
    
    # SafeTensors - Microsoft original (~2.2GB)
    mkdir -p "$OUTPUT_DIR/tinyllama-1.1b-safetensors"
    echo "2. TinyLlama SafeTensors (~2.2GB, ~15 min)..."
    curl -L -C - -o "$OUTPUT_DIR/tinyllama-1.1b-safetensors/model.safetensors" \
        "https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0/resolve/main/model.safetensors" 2>&1 &
    SAFE_PID=$!
    
    curl -L -C - -o "$OUTPUT_DIR/tinyllama-1.1b-safetensors/config.json" \
        "https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0/resolve/main/config.json" 2>&1 &
    
    curl -L -C - -o "$OUTPUT_DIR/tinyllama-1.1b-safetensors/tokenizer.json" \
        "https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0/resolve/main/tokenizer.json" 2>&1 &
    
    # MLX - mlx-community optimized (~2.2GB)
    mkdir -p "$OUTPUT_DIR/tinyllama-1.1b-mlx"
    echo "3. TinyLlama MLX (~2.2GB, ~15 min)..."
    curl -L -C - -o "$OUTPUT_DIR/tinyllama-1.1b-mlx/model.safetensors" \
        "https://huggingface.co/mlx-community/TinyLlama-1.1B-Chat-v1.0/resolve/main/model.safetensors" 2>&1 &
    MLX_PID=$!
    
    curl -L -C - -o "$OUTPUT_DIR/tinyllama-1.1b-mlx/config.json" \
        "https://huggingface.co/mlx-community/TinyLlama-1.1B-Chat-v1.0/resolve/main/config.json" 2>&1 &
    
    curl -L -C - -o "$OUTPUT_DIR/tinyllama-1.1b-mlx/tokenizer.json" \
        "https://huggingface.co/mlx-community/TinyLlama-1.1B-Chat-v1.0/resolve/main/tokenizer.json" 2>&1 &
    
    wait $GGUF_PID $SAFE_PID $MLX_PID 2>/dev/null || true
    echo ""
fi

if [ "$MODEL" = "phi2" ] || [ "$MODEL" = "all" ]; then
    echo "Downloading Phi-2 (3.3B, balanced)"
    echo ""
    
    # GGUF - TheBloke quantized (~2GB)
    mkdir -p "$OUTPUT_DIR/phi-2-gguf"
    echo "1. Phi-2 GGUF Q4 (~2GB, ~15 min)..."
    curl -L -C - -o "$OUTPUT_DIR/phi-2-gguf/phi-2.Q4_K_M.gguf" \
        "https://huggingface.co/TheBloke/phi-2-GGUF/resolve/main/phi-2.Q4_K_M.gguf" 2>&1 &
    GGUF_PID=$!
    
    # SafeTensors - Microsoft (~4.8GB)
    mkdir -p "$OUTPUT_DIR/phi-2-safetensors"
    echo "2. Phi-2 SafeTensors (~4.8GB, ~30 min)..."
    curl -L -C - -o "$OUTPUT_DIR/phi-2-safetensors/model.safetensors" \
        "https://huggingface.co/microsoft/phi-2/resolve/main/model.safetensors" 2>&1 &
    SAFE_PID=$!
    
    curl -L -C - -o "$OUTPUT_DIR/phi-2-safetensors/config.json" \
        "https://huggingface.co/microsoft/phi-2/resolve/main/config.json" 2>&1 &
    
    curl -L -C - -o "$OUTPUT_DIR/phi-2-safetensors/tokenizer.json" \
        "https://huggingface.co/microsoft/phi-2/resolve/main/tokenizer.json" 2>&1 &
    
    # MLX - mlx-community (~4.8GB)
    mkdir -p "$OUTPUT_DIR/phi-2-mlx"
    echo "3. Phi-2 MLX (~4.8GB, ~30 min)..."
    curl -L -C - -o "$OUTPUT_DIR/phi-2-mlx/model.safetensors" \
        "https://huggingface.co/mlx-community/phi-2/resolve/main/model.safetensors" 2>&1 &
    MLX_PID=$!
    
    curl -L -C - -o "$OUTPUT_DIR/phi-2-mlx/config.json" \
        "https://huggingface.co/mlx-community/phi-2/resolve/main/config.json" 2>&1 &
    
    curl -L -C - -o "$OUTPUT_DIR/phi-2-mlx/tokenizer.json" \
        "https://huggingface.co/mlx-community/phi-2/resolve/main/tokenizer.json" 2>&1 &
    
    wait $GGUF_PID $SAFE_PID $MLX_PID 2>/dev/null || true
    echo ""
fi

echo "=========================================="
echo "Download Summary"
echo "=========================================="

if [ -d "$OUTPUT_DIR/tinyllama-1.1b-gguf" ]; then
    echo ""
    echo "TinyLlama-1.1B:"
    ls -lh "$OUTPUT_DIR"/tinyllama-1.1b-*/ 2>/dev/null || echo "  (downloading...)"
fi

if [ -d "$OUTPUT_DIR/phi-2-gguf" ]; then
    echo ""
    echo "Phi-2:"
    ls -lh "$OUTPUT_DIR"/phi-2-*/ 2>/dev/null || echo "  (downloading...)"
fi

echo ""
echo "Next steps:"
echo "1. Verify models: cargo run --release --bin verify-gguf -- --model-path $OUTPUT_DIR/tinyllama-1.1b-gguf/..."
echo "2. Run benchmarks: cargo run --release --bin minerva-bench -- --format gguf"
echo "3. Compare results: cat MISTRAL_BENCHMARK_RESULTS.csv"
