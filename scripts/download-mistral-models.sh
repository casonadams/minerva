#!/bin/bash

#############################################################################
# Mistral 7B Model Download Script
# Downloads Mistral 7B in three formats for benchmarking:
# 1. GGUF - llama.cpp backend
# 2. SafeTensors - Pure Rust backend  
# 3. MLX - Apple Silicon optimized backend
#############################################################################

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Directories
MODELS_DIR="${1:-.}/models"
MISTRAL_GGUF_DIR="$MODELS_DIR/mistral-7b-gguf"
MISTRAL_SAFETENSORS_DIR="$MODELS_DIR/mistral-7b-safetensors"
MISTRAL_MLX_DIR="$MODELS_DIR/mistral-7b-mlx"

# HuggingFace repositories
GGUF_REPO="https://huggingface.co/TheBloke/Mistral-7B-GGUF/resolve/main"
SAFETENSORS_REPO="https://huggingface.co/mistralai/Mistral-7B/resolve/main"
MLX_REPO="https://huggingface.co/mlx-community/Mistral-7B/resolve/main"

# Model files
GGUF_Q4="Mistral-7B.Q4_K_M.gguf"
GGUF_Q5="Mistral-7B.Q5_K_M.gguf"
SAFETENSORS_MODEL="model.safetensors"
MLX_MODEL="model.safetensors"

#############################################################################
# Helper Functions
#############################################################################

print_header() {
    echo -e "\n${BLUE}================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}================================================${NC}\n"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${YELLOW}ℹ $1${NC}"
}

download_file() {
    local url=$1
    local output=$2
    local description=$3
    
    print_info "Downloading $description..."
    print_info "URL: $url"
    print_info "Output: $output"
    
    if curl -L --progress-bar -o "$output" "$url"; then
        local size=$(du -h "$output" | cut -f1)
        print_success "Downloaded $description ($size)"
        return 0
    else
        print_error "Failed to download $description"
        return 1
    fi
}

verify_file() {
    local file=$1
    local description=$2
    
    if [ -f "$file" ]; then
        local size=$(du -h "$file" | cut -f1)
        print_success "Verified $description ($size)"
        return 0
    else
        print_error "File not found: $file"
        return 1
    fi
}

#############################################################################
# Main Script
#############################################################################

print_header "Mistral 7B Model Download Script"

echo -e "This script will download Mistral 7B in three formats:\n"
echo -e "  1. ${BLUE}GGUF${NC} (4-bit & 5-bit quantized)"
echo -e "     → llama.cpp backend"
echo -e "     → GPU acceleration with Metal/CUDA"
echo -e "     → Size: ~4.8GB (Q4) or ~6.3GB (Q5)\n"

echo -e "  2. ${BLUE}SafeTensors${NC} (Full precision)"
echo -e "     → Pure Rust backend"
echo -e "     → No C/CUDA dependencies"
echo -e "     → Portable, auditable"
echo -e "     → Size: ~13GB\n"

echo -e "  3. ${BLUE}MLX${NC} (Apple Silicon optimized)"
echo -e "     → MLX backend for Apple Silicon"
echo -e "     → Metal acceleration"
echo -e "     → Optimized for M1/M2/M3 chips"
echo -e "     → Size: ~13GB\n"

read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    print_error "Cancelled by user"
    exit 1
fi

# Create directories
print_header "Creating directories"
mkdir -p "$MISTRAL_GGUF_DIR"
mkdir -p "$MISTRAL_SAFETENSORS_DIR"
mkdir -p "$MISTRAL_MLX_DIR"
print_success "Directories created"

# Download GGUF models (llama.cpp backend)
print_header "Downloading GGUF Models (llama.cpp backend)"
print_info "GGUF is quantized format optimized for llama.cpp"
print_info "Q4_K_M = 4-bit quantization (recommended, ~4.8GB)"
print_info "Q5_K_M = 5-bit quantization (higher quality, ~6.3GB)"

download_file \
    "$GGUF_REPO/$GGUF_Q4" \
    "$MISTRAL_GGUF_DIR/$GGUF_Q4" \
    "GGUF Q4_K_M (~4.8GB)" || print_info "Q4 download failed"

download_file \
    "$GGUF_REPO/$GGUF_Q5" \
    "$MISTRAL_GGUF_DIR/$GGUF_Q5" \
    "GGUF Q5_K_M (~6.3GB)" || print_info "Q5 download optional, skipping"

# Download SafeTensors model (Pure Rust backend)
print_header "Downloading SafeTensors Model (Pure Rust backend)"
print_info "SafeTensors is a safe, fast format for loading weights"
print_info "Used by pure Rust backend without C/CUDA dependencies"
print_info "Full precision: ~13GB\n"

download_file \
    "$SAFETENSORS_REPO/$SAFETENSORS_MODEL" \
    "$MISTRAL_SAFETENSORS_DIR/$SAFETENSORS_MODEL" \
    "SafeTensors model (~13GB)" || print_info "SafeTensors download failed"

# Download MLX model (Apple Silicon backend)
print_header "Downloading MLX Model (Apple Silicon backend)"
print_info "MLX is optimized for Apple Neural Engine"
print_info "Best performance on M1/M2/M3 Macs with Metal acceleration"
print_info "Full precision: ~13GB\n"

download_file \
    "$MLX_REPO/$MLX_MODEL" \
    "$MISTRAL_MLX_DIR/$MLX_MODEL" \
    "MLX model (~13GB)" || print_info "MLX download failed"

# Download tokenizer and configs
print_header "Downloading Tokenizer and Config Files"

for dir in "$MISTRAL_SAFETENSORS_DIR" "$MISTRAL_MLX_DIR"; do
    print_info "Downloading tokenizer for $(basename $dir)..."
    
    if [ "$(basename $dir)" = "mistral-7b-mlx" ]; then
        repo="$MLX_REPO"
    else
        repo="$SAFETENSORS_REPO"
    fi
    
    download_file \
        "$repo/tokenizer.json" \
        "$dir/tokenizer.json" \
        "tokenizer.json for $(basename $dir)" || print_info "Tokenizer download failed"
    
    download_file \
        "$repo/config.json" \
        "$dir/config.json" \
        "config.json for $(basename $dir)" || print_info "Config download failed"
done

# Verification
print_header "Verification"

verify_file "$MISTRAL_GGUF_DIR/$GGUF_Q4" "GGUF Q4_K_M" || true
verify_file "$MISTRAL_SAFETENSORS_DIR/$SAFETENSORS_MODEL" "SafeTensors model" || true
verify_file "$MISTRAL_SAFETENSORS_DIR/tokenizer.json" "SafeTensors tokenizer" || true
verify_file "$MISTRAL_MLX_DIR/$MLX_MODEL" "MLX model" || true
verify_file "$MISTRAL_MLX_DIR/tokenizer.json" "MLX tokenizer" || true

# Summary
print_header "Download Complete"

echo -e "Models downloaded to: $MODELS_DIR\n"
echo -e "Directory structure:"
echo -e "  models/"
echo -e "  ├── mistral-7b-gguf/        [llama.cpp + GPU]"
ls -lh "$MISTRAL_GGUF_DIR" 2>/dev/null | tail -n +2 | awk '{print "  │   ├── " $9 " (" $5 ")"}' || echo "  │   ├── [files]"
echo -e "  │"
echo -e "  ├── mistral-7b-safetensors/ [Pure Rust]"
ls -lh "$MISTRAL_SAFETENSORS_DIR" 2>/dev/null | tail -n +2 | awk '{print "  │   ├── " $9 " (" $5 ")"}' || echo "  │   ├── [files]"
echo -e "  │"
echo -e "  └── mistral-7b-mlx/         [Apple Silicon + Metal]"
ls -lh "$MISTRAL_MLX_DIR" 2>/dev/null | tail -n +2 | awk '{print "      ├── " $9 " (" $5 ")"}' || echo "      ├── [files]"

print_success "Ready to run benchmarks!"
echo -e "\nNext steps:"
echo -e "  1. Compile release build:"
echo -e "     ${YELLOW}cargo build --release${NC}"
echo -e ""
echo -e "  2. Run benchmarks:"
echo -e "     ${YELLOW}cargo run --release --bin minerva-bench -- --all-formats${NC}"
echo -e ""
echo -e "  3. Results will be saved to:"
echo -e "     ${YELLOW}MISTRAL_BENCHMARK_REPORT.md${NC}"
echo -e ""
echo -e "Backend-specific commands:"
echo -e "  GGUF only:       ${YELLOW}cargo run --release --bin minerva-bench -- --format gguf${NC}"
echo -e "  SafeTensors:     ${YELLOW}cargo run --release --bin minerva-bench -- --format safetensors${NC}"
echo -e "  MLX (Apple):     ${YELLOW}cargo run --release --bin minerva-bench -- --format mlx${NC}"

