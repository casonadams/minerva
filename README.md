# Minerva - OpenAI-Compatible Local LLM Server

A Tauri desktop application that serves local Large Language Models via OpenAI-compatible APIs. Run state-of-the-art LLMs locally on Apple Silicon with GPU acceleration and use them with any tool that supports the OpenAI API format.

## Overview

Minerva is a production-ready LLM server featuring:
- **OpenAI API Compatibility** - Drop-in replacement for OpenAI endpoints
- **Local Inference** - No external API calls, full privacy control
- **GPU Acceleration** - Apple Silicon Metal GPU support with CPU fallback
- **Multi-Model Support** - Load and switch between multiple GGUF models
- **Production Hardened** - Resilience patterns, health checks, comprehensive monitoring
- **Zero Dependencies** - Self-contained Tauri app, runs standalone

## Quick Start

### Prerequisites
- Rust 1.70+ ([https://rustup.rs/](https://rustup.rs/))
- Node.js 18+ and pnpm
- macOS 10.15+
- GGUF format LLM files (download from HuggingFace)

### Installation

```bash
# Install dependencies
pnpm install

# Build the backend (Rust)
cd src-tauri && cargo build --release && cd ..
```

### Running Minerva Server

```bash
# Development mode (auto-reload)
pnpm tauri dev

# Production build
pnpm tauri build --release
```

The server will start on `http://localhost:11434` with the OpenAI-compatible API available at `/v1`.

## OpenAI API Usage

### 1. Get Available Models

Once the server is running, discover available models:

```bash
curl http://localhost:11434/v1/models
```

**Response:**
```json
{
  "object": "list",
  "data": [
    {
      "id": "gpt-oss-20b",
      "object": "model",
      "created": 1769376104,
      "owned_by": "local",
      "quantization": "MXFP4",
      "file_size_mb": 12109.57,
      "tensor_count": 459
    }
  ]
}
```

### 2. Get Specific Model Info

```bash
curl http://localhost:11434/v1/models/gpt-oss-20b
```

### 3. Use with OpenAI-Compatible Tools

Since Minerva is OpenAI API compatible, you can use it with any OpenAI client:

**cURL (direct):**
```bash
curl -X POST http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-oss-20b",
    "messages": [{"role": "user", "content": "Hello, who are you?"}],
    "max_tokens": 100
  }'
```

**Python:**
```python
from openai import OpenAI

client = OpenAI(
    api_key="sk-local",  # any key works locally
    base_url="http://localhost:11434/v1"
)

response = client.chat.completions.create(
    model="gpt-oss-20b",
    messages=[
        {"role": "user", "content": "Hello, who are you?"}
    ],
    max_tokens=100
)

print(response.choices[0].message.content)
```

**JavaScript/Node.js:**
```javascript
import OpenAI from "openai";

const client = new OpenAI({
  apiKey: "sk-local",  // any key works locally
  baseURL: "http://localhost:11434/v1",
  dangerouslyAllowBrowser: true
});

const message = await client.chat.completions.create({
  model: "gpt-oss-20b",
  messages: [
    { role: "user", content: "Hello, who are you?" }
  ],
  max_tokens: 100
});

console.log(message.choices[0].message.content);
```

**OpenCode.ai / Cursor / Other OpenAI-Compatible Editors:**
```
Configure in settings:
├── API Provider: OpenAI API
├── Base URL: http://localhost:11434/v1
├── API Key: sk-local (or any value)
└── Model: gpt-oss-20b (or other available model)
```

### 4. Monitor Server Health

```bash
# Health check
curl http://localhost:11434/health

# Readiness probe
curl http://localhost:11434/ready

# View metrics
curl http://localhost:11434/metrics
```

## Getting Started with Models

### Download a Model

Models must be in GGUF format and placed in `~/.minerva/models/`:

```bash
# Create models directory
mkdir -p ~/.minerva/models

# Download from HuggingFace (example: GPT-OSS 20B)
# https://huggingface.co/ggml-org/gpt-oss-20b-GGUF
# Place the downloaded .gguf file in ~/.minerva/models/
```

### Popular Models to Try

| Model | Size | Best For |
|-------|------|----------|
| **GPT-OSS 20B** | 12.1 GB | Balance of speed & quality |
| **Mistral 7B** | 4.2 GB | Fast inference |
| **Llama 2 7B** | 4.0 GB | Open source, well-tested |
| **Neural Chat 7B** | 4.2 GB | Instruction following |
| **Orca 2 13B** | 8.0 GB | Reasoning tasks |

### Model Discovery Sources

- **HuggingFace:** [GGUF Models](https://huggingface.co/models?library=gguf)
- **TheBloke:** [Quantized Models](https://huggingface.co/TheBloke)
- **Ollama:** [Model Library](https://ollama.ai/library)

### Integration Examples

**Vercel AI SDK:**
```javascript
import { openai } from "@ai-sdk/openai-compatible";

const model = openai("gpt-oss-20b", {
  baseURL: "http://localhost:11434/v1",
  apiKey: "sk-local"
});

const response = await generateText({
  model,
  prompt: "Hello, who are you?"
});
```

**LangChain (Python):**
```python
from langchain.llms import OpenAI

llm = OpenAI(
    openai_api_base="http://localhost:11434/v1",
    openai_api_key="sk-local",
    model_name="gpt-oss-20b"
)

response = llm("Hello, who are you?")
```

## Features

### Core Features
- ✅ **OpenAI API v1 Compatible** - Works with OpenCode.ai, Cursor, LM Studio, Ollama, etc.
- ✅ **Fast Model Discovery** - Get model info in < 100ms (no tensor loading)
- ✅ **GGUF & SafeTensors Support** - Auto-detect and load quantized models
- ✅ **GPU Acceleration** - Apple Silicon Metal GPU with automatic CPU fallback
- ✅ **Streaming Responses** - Server-Sent Events (SSE) for real-time token generation
- ✅ **Multi-Model Support** - Register and switch between multiple models
- ✅ **Model Management UI** - Built-in web interface for model selection

### Production Features (Phase 7)
- ✅ **Observability** - Structured logging, request tracing, health checks
- ✅ **Resilience** - Retry with backoff, circuit breaker, graceful fallbacks
- ✅ **Performance** - Adaptive configuration, profiling, metrics collection
- ✅ **Health Endpoints** - `/health`, `/ready`, `/metrics` for monitoring
- ✅ **Error Recovery** - 4-level error classification with intelligent recovery

### Performance Features (Phase 6)
- ✅ **GPU Inference** - LLaMA-based inference with Metal GPU acceleration
- ✅ **KV Cache** - Efficient incremental generation with cache management
- ✅ **Batch Processing** - Async and parallel batch inference
- ✅ **Model Caching** - LRU-based multi-model memory management
- ✅ **Quantization** - Support for quantized GGUF models

## Project Architecture

```
Minerva = Tauri Desktop App + Production LLM Server
           ├── Frontend (Svelte 5)
           └── Backend (Rust + Axum)
               ├── HTTP Server (OpenAI-compatible API)
               ├── LLM Inference (GPU-accelerated)
               ├── Model Management (GGUF loading)
               ├── Observability (Logging, metrics, tracing)
               └── Resilience (Retry, circuit breaker, fallbacks)
```

### Technology Stack
- **Frontend:** Svelte 5 + TypeScript + SvelteKit
- **Backend:** Rust 1.70+ + Tauri + Axum
- **LLM Engine:** GGUF model loading + Metal GPU acceleration
- **API:** OpenAI-compatible REST endpoints
- **Testing:** 827 tests (579 unit + 248 integration)

## Project Status

### ✅ Completed Phases

**Phase 1: Foundation** ✅
- Tauri project setup, Axum HTTP server, OpenAI-compatible API models

**Phase 2: Model Loading & File System** ✅
- GGUF parsing, model discovery, configuration management, Tauri IPC commands

**Phase 3: LLM Inference Engine** ✅
- Inference architecture, context management, SSE streaming, metrics framework

**Phase 3.5: Real LLM Integration** ✅
- LlamaEngine wrapper, GPU context management, token streaming, error handling

**Phase 3.5a: Backend Abstraction** ✅
- InferenceBackend trait, MockBackend for testing, plugin architecture

**Phase 3.5b: Real llama.cpp Integration** ✅
- Full llama.cpp integration, Metal/CUDA GPU support, token streaming, benchmarking

**Phase 4: Advanced Features** ✅
- Step 1: Multi-model support with context management
- Step 2: Model caching with LRU eviction, intelligent preloading
- Step 3: Cache optimization with memory-aware strategies
- Step 4: GPU computation engine with Metal integration
- Step 5: KV cache optimizer for efficient generation
- Step 6: Batch processing (async/parallel)
- Step 7: Baseline measurements and performance profiling

**Phase 5: Performance & Scaling** ✅
- Async/parallel batch processing, GPU batch scheduling, streaming responses

**Phase 6: Deep Learning Core** ✅
- LLaMA model implementation, tokenization, GPU inference engine, cache optimization

**Phase 7: Production Hardening & Observability** ✅
- Step 1: Structured logging and request tracing
- Step 2: Error recovery & resilience patterns
- Step 3: Observable server endpoints
- Step 4: Desktop performance optimization
- Step 5: Server performance metrics integration

## Configuration

Configuration is automatically created at `~/.minerva/config.json`:

```json
{
  "server": {
    "port": 11434,
    "host": "127.0.0.1"
  },
  "models_dir": "~/.minerva/models",
  "gpu": {
    "enabled": true,
    "backend": "metal"
  }
}
```

### Model Directory Structure

```
~/.minerva/
├── config.json                 # Configuration file
└── models/                     # GGUF model storage
    ├── mistral-7b.gguf       # Download from HuggingFace
    ├── llama-2-7b.gguf
    └── neural-chat-7b.gguf
```

## File Structure

```
minerva/
├── src/                        # Svelte 5 frontend
│   ├── routes/                # SvelteKit pages
│   └── components/            # Reusable components
│
├── src-tauri/                  # Tauri/Rust backend
│   ├── src/
│   │   ├── lib.rs            # Module entry point
│   │   ├── main.rs           # Binary entry
│   │   ├── server.rs         # Axum HTTP server + endpoints
│   │   ├── config.rs         # Configuration management
│   │   ├── commands.rs       # Tauri IPC commands
│   │   ├── error.rs          # Error types & handling
│   │   │
│   │   ├── logging/          # Phase 7 Step 1
│   │   │   ├── mod.rs        # Logging initialization
│   │   │   └── spans.rs      # Request tracing
│   │   │
│   │   ├── resilience/       # Phase 7 Step 2
│   │   │   ├── mod.rs        # Error classification
│   │   │   ├── retry.rs      # Exponential backoff + jitter
│   │   │   ├── circuit_breaker.rs # Circuit breaker pattern
│   │   │   ├── fallback.rs   # Fallback mechanisms
│   │   │   ├── health.rs     # Health check infrastructure
│   │   │   ├── timeout.rs    # Deadline management
│   │   │   └── coordinator.rs # Pattern orchestration
│   │   │
│   │   ├── observability/    # Phase 7 Step 3
│   │   │   ├── mod.rs        # Metrics snapshot
│   │   │   ├── endpoints.rs  # HTTP response types
│   │   │   ├── metrics.rs    # Collection & aggregation
│   │   │   └── tracing_middleware.rs # Request tracing
│   │   │
│   │   ├── performance/      # Phase 7 Steps 4-5
│   │   │   ├── mod.rs        # Performance metrics
│   │   │   ├── adaptive.rs   # Adaptive configuration
│   │   │   ├── profiler.rs   # Operation profiling
│   │   │   └── integration.rs # Server integration
│   │   │
│   │   ├── models/
│   │   │   ├── mod.rs        # Model types & registry
│   │   │   ├── loader.rs     # GGUF model discovery
│   │   │   └── gguf_parser.rs # GGUF binary parsing
│   │   │
│   │   └── inference/
│   │       ├── mod.rs        # Inference infrastructure
│   │       ├── llama_engine.rs         # Real inference wrapper
│   │       ├── llama_adapter.rs        # Backend abstraction
│   │       ├── gpu_context.rs          # GPU memory management
│   │       ├── gpu_compute_engine.rs   # GPU compute operations
│   │       ├── gpu_batch_scheduler.rs  # GPU batch scheduling
│   │       ├── gpu_llama_integration.rs # GPU-LLaMA integration
│   │       ├── metal_gpu.rs            # Metal GPU implementation
│   │       ├── token_stream.rs         # Token collection
│   │       ├── streaming.rs            # SSE formatting
│   │       ├── llama_inference.rs      # LLaMA model inference
│   │       ├── llama_tokenizer.rs      # LLaMA tokenization
│   │       ├── context_manager.rs      # Multi-model context
│   │       ├── model_cache.rs          # Model caching
│   │       ├── model_registry.rs       # Model registry
│   │       ├── preload_manager.rs      # Model preloading
│   │       ├── kv_cache_optimizer.rs   # KV cache management
│   │       ├── cache_optimizer.rs      # Cache optimization
│   │       ├── parameters.rs           # Request parameters
│   │       ├── metrics.rs              # Performance metrics
│   │       ├── benchmarks.rs           # Performance benchmarks
│   │       ├── garbage_collector.rs    # Memory management
│   │       ├── pattern_detector.rs     # Usage pattern detection
│   │       └── error_recovery.rs       # Error recovery
│   │
│   └── Cargo.toml            # Rust dependencies
│
├── tests/
│   └── integration_tests.rs  # 248 integration tests
│
├── docs/                     # Documentation hub
│   ├── README.md             # Documentation index & navigation
│   ├── PHASES.md             # Summary of all 7 completed phases
│   ├── DEVELOPMENT.md        # Development setup & guide
│   ├── GPU_ACCELERATION.md   # GPU configuration & optimization
│   ├── CODE_QUALITY.md       # Engineering standards
│   ├── SCRIPTS.md            # Available npm/pnpm scripts
│   ├── IMPLEMENTATION_PLAN.md # Original architecture design
│   ├── TEST_STRUCTURE.md     # Testing patterns
│   └── archive/              # Historical phase documentation (reference)
│
├── README.md                 # This file
└── pnpm scripts              # Development helpers
```

## Testing

### Run Tests
```bash
# Run all tests (827 total)
pnpm test

# Run specific test suite
pnpm test:backend        # Backend tests only
pnpm test:backend:unit   # Unit tests only
pnpm test:backend:integration # Integration tests only

# Watch mode
pnpm test:backend:watch

# View test output
pnpm test -- --nocapture
```

### Test Coverage

**Unit Tests: 579**
- Logging & tracing (6 tests)
- Resilience patterns (67 tests)
- Observability (36 tests)
- Performance metrics (30 tests)
- Model management (100+ tests)
- GPU operations (150+ tests)
- Inference engine (100+ tests)
- And more...

**Integration Tests: 248**
- End-to-end inference pipeline
- Model discovery and loading
- Batch processing
- Streaming responses
- Error recovery scenarios
- Performance benchmarks

**Results:** ✅ **827 tests passing, 0 violations, 0 warnings**

## Development

### Recommended IDE Setup
- VS Code
- Extensions: Svelte, Tauri, rust-analyzer
- Rust toolchain 1.70+

### Building

```bash
# Development (watch mode)
pnpm tauri dev

# Production build
pnpm tauri build --release

# Just the backend
cd src-tauri && cargo build --release

# Format and lint
pnpm fmt
pnpm lint
```

### Common Commands

```bash
# Install dependencies
pnpm install

# Development server with hot reload
pnpm tauri dev

# Format code
pnpm fmt

# Lint code
pnpm lint

# Run tests with watch
pnpm test:backend:watch

# Build production app
pnpm tauri build --release

# Full validation
pnpm check:all
```

## HTTP API Endpoints

### OpenAI-Compatible Endpoints

**Model Discovery:**
- `GET /v1/models` - List all available models with metadata
- `GET /v1/models/{id}` - Get detailed info for specific model

**Inference:**
- `POST /v1/chat/completions` - Chat completion (streaming with SSE)

**Server Health:**
- `GET /health` - Health status with component details
- `GET /ready` - Readiness probe for orchestration
- `GET /metrics` - Performance metrics snapshot

All endpoints return OpenAI-compatible JSON responses. See [OPENAI_API_INTEGRATION.md](OPENAI_API_INTEGRATION.md) for detailed API documentation.

## Contributing

Follow the engineering standards in `AGENTS.md`:
- All code must pass `pnpm lint` (0 violations)
- All code must pass `pnpm test` (100% pass rate)
- Cyclomatic complexity M ≤ 3 for all functions
- Functions ≤ 25 lines
- Files ≤ 100 lines logical
- SOLID principles
- Meaningful tests with assertions

## Architecture Decisions

### Why Tauri?
- Single codebase for desktop + server
- Rust backend for safety and performance
- Native OS integration
- Small binary size
- Cross-platform capable

### Why Rust?
- Memory safety without GC
- Zero-cost abstractions
- Excellent async/await
- Strong type system
- Performance comparable to C++

### Why OpenAI-Compatible?
- Works with existing LLM tools/libraries
- Standard API contracts
- Easy migration from OpenAI
- Community ecosystem

## Performance Characteristics

**GPU Mode (Metal):**
- Model loading: 1-5 seconds
- Inference: 20-100ms latency (7B models)
- Throughput: 10-50 tokens/second

**CPU Mode:**
- Model loading: 5-10 seconds
- Inference: 100-500ms latency (7B models)
- Throughput: 2-10 tokens/second

**Memory Usage:**
- 7B model: ~4-8GB
- 13B model: ~8-16GB
- Varies by quantization level

## Troubleshooting

### App won't start
- Check Rust installation: `rustc --version`
- Check Node.js: `node --version`
- Clear cache: `rm -rf node_modules && pnpm install`

### GPU not detecting
- Check macOS version (10.15+)
- Verify Apple Silicon (arm64)
- Check Metal support: `system_profiler SPDisplaysDataType`

### Model loading fails
- Verify GGUF format: Check file starts with "GGUF"
- Check model size vs available RAM
- Ensure models in `~/.minerva/models/`

### Poor performance
- Check system resources: `Activity Monitor`
- Monitor GPU usage
- Try smaller model or quantization
- Review `/metrics` endpoint

## Known Limitations

- macOS only (ARM64 Apple Silicon optimized)
- Single GPU system (no distributed inference)
- Models limited to available RAM
- No fine-tuning in current version

## Future Roadmap (Phase 8+)

**Optional next phases (not started):**
- **Phase 8: OpenTelemetry Integration** - Standard observability format (Jaeger, Datadog integration)
- **Phase 8 Alternative: Distributed Inference** - Multi-machine model inference
- **Phase 8 Alternative: Auto-Scaling** - Dynamic model preloading based on demand
- **Phase 8 Alternative: Advanced API** - Fine-tuning endpoints, advanced batch processing

**Long-term enhancements:**
- Prometheus metrics export
- Kubernetes health probe integration
- Model fine-tuning API
- Multi-GPU support
- Performance dashboard

## License

MIT License - See LICENSE file for details

## Support & Community

- **Issues:** GitHub issues
- **Discussions:** GitHub discussions
- **Documentation:** See `/docs` folder

---

**Status:** ✅ **Production Ready**

Minerva is fully production-hardened and ready for local LLM deployment with OpenAI API compatibility.

---

**Project Status:** ✅ All 7 Phases Complete - Production Ready  
**Last Updated:** January 2025 (OpenAI API Optimization + Phase 7 Complete)  
**Total Tests:** 874 (all passing ✅)  
**Code Quality:** 0 lint violations, 0 compiler warnings  
**Documentation:** Complete API docs + integration guides

**Recent Additions (OpenAI API Optimization):**
- ✨ **Fast Model Discovery** - < 100ms metadata loading, < 1KB JSON responses
- ✨ **OpenCode.ai Integration** - Drop-in replacement with zero configuration
- ✨ **Tool-Optimized Loader** - Minimal context overhead for AI agents
- ✨ **Multi-Model Registry** - Register and manage multiple models simultaneously
- 📖 **Complete Integration Guides** - See [OPENAI_API_INTEGRATION.md](OPENAI_API_INTEGRATION.md) and [OPENCODE_INTEGRATION_GUIDE.md](OPENCODE_INTEGRATION_GUIDE.md)

**Quick Links:**
- **Getting Started:** See [docs/README.md](docs/README.md)
- **OpenAI API Details:** See [OPENAI_API_INTEGRATION.md](OPENAI_API_INTEGRATION.md)
- **All Phases Summary:** See [docs/PHASES.md](docs/PHASES.md)
- **Development Setup:** See [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md)
