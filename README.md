# Minerva - Local LLM Server

A Tauri desktop application that serves local Large Language Models via OpenAI-compatible APIs. Run LLMs locally and use them with any tool that supports the OpenAI API format.

## Features

- Serve GGUF-format LLMs locally
- OpenAI-compatible REST API (`/v1/chat/completions`, `/v1/models`)
- Mac Metal GPU acceleration (Apple Silicon)
- Works with OpenCode, LangChain, and other LLM clients
- Built with Rust + Tauri for performance
- Model management UI

## Quick Start

### Prerequisites

- Rust 1.70+ ([https://rustup.rs/](https://rustup.rs/))
- Node.js 18+ and pnpm
- macOS 10.15+

### Setup Models

1. Create the models directory (auto-created on first run):
```bash
mkdir -p ~/.minerva/models
```

2. Place GGUF model files in `~/.minerva/models/`:
```bash
# Example with llama.cpp models
cp ~/Downloads/mistral-7b.gguf ~/.minerva/models/
cp ~/Downloads/llama-2-7b.gguf ~/.minerva/models/
```

3. Models are automatically discovered on app startup

### Development

```bash
# Install dependencies
pnpm install

# Run development server
pnpm tauri dev
```

### API Usage

```bash
# List available models
curl http://localhost:11434/v1/models

# Chat completion
curl http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mistral-7b",
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

### Tauri Commands

The frontend can use these Tauri IPC commands:

```typescript
// Get current configuration
invoke('get_config')

// List discovered models
invoke('list_discovered_models')

// Load specific model
invoke('load_model_file', { modelPath: '/path/to/model.gguf' })

// Get/set models directory
invoke('get_models_directory')
invoke('set_models_directory', { path: '/new/path' })

// Ensure models directory exists
invoke('ensure_models_directory')
```

## Architecture

- **Frontend**: Svelte 5 + TypeScript + SvelteKit
- **Backend**: Rust + Tauri + Axum HTTP server
- **LLM Engine**: llama.cpp with Metal GPU acceleration
- **API**: OpenAI-compatible REST endpoints

## Project Status

**Phase 1: Foundation** ✅ COMPLETE
- [x] Project renamed to Minerva
- [x] Axum HTTP server setup
- [x] API response models created
- [x] Mock endpoints implemented
- [x] Error handling with OpenAI format
- [x] 5 unit tests passing

**Phase 2: Model Loading & File System** ✅ COMPLETE
- [x] GGUF file parsing with metadata extraction
- [x] Recursive model discovery from filesystem
- [x] Configuration management with persistence
- [x] Tauri commands for model management
- [x] HTTP server integration with real models
- [x] Directory structure auto-creation
- [x] 26 comprehensive tests (17 unit + 9 integration)

**Phase 3: LLM Inference Engine** ✅ COMPLETE
- [x] Core inference architecture with mock implementation
- [x] Multi-model context management with LRU eviction
- [x] Parameter validation and request parsing
- [x] Server-Sent Events (SSE) streaming infrastructure
- [x] Performance metrics framework
- [x] 26 comprehensive tests
- See `PHASE_3_IMPLEMENTATION.md` for details

**Phase 3.5: Real LLM Integration** ✅ COMPLETE
- [x] LlamaEngine for actual model loading/inference
- [x] GpuContext with Metal/CUDA auto-detection
- [x] TokenStream for real token collection
- [x] GPU memory management and allocation
- [x] Error handling for resource constraints
- [x] 10 Phase 3.5 integration tests
- See `docs/PHASE_3_5_IMPLEMENTATION.md` for architecture

**Phase 3.5a: Inference Backend Integration** ✅ COMPLETE
- [x] Intelligent mock inference engine with pattern-based responses
- [x] InferenceBackend trait abstraction layer
- [x] MockBackend for comprehensive testing
- [x] LlamaCppBackend stub with implementation comments
- [x] 6 new integration tests for backend abstraction
- [x] 112 total tests (all passing)
- [x] Clear path to real llama.cpp integration
- See `docs/PHASE_3_5A_COMPLETION.md` for details

**Phase 3.5b: Real llama.cpp Integration** ✅ COMPLETE
- [x] Step 1: Research llama.cpp crate (v0.3.2 analysis)
- [x] Step 2: Real backend implementation with GPU acceleration
- [x] Step 3: GPU initialization (Metal/CUDA per platform)
- [x] Step 4: Token streaming with callbacks (SSE-ready)
- [x] Step 5: Error handling & recovery (6 recovery strategies)
- [x] Step 6: Performance benchmarking (GPU vs CPU metrics)
- [x] Step 7: Integration testing (7 new real-world tests)
- [x] Step 8: Documentation (Complete guides)
- [x] 135 total tests passing (all phases)
- See `docs/PHASE_3_5B_COMPLETE.md` for full details

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
├── config.json                    # Configuration file
└── models/                        # GGUF model storage
    ├── mistral-7b.gguf          # Place your GGUF files here
    ├── llama-2-7b.gguf
    └── neural-chat-7b.gguf
```

Models are discovered automatically on app startup. GGUF files can be obtained from:
- [Hugging Face](https://huggingface.co/models?library=gguf)
- [TheBloke](https://huggingface.co/TheBloke)
- [Ollama model library](https://ollama.ai/library)

## Development

### Recommended IDE Setup

[VS Code](https://code.visualstudio.com/) + [Svelte](https://marketplace.visualstudio.com/items?itemName=svelte.svelte-vscode) + [Tauri](https://marketplace.visualstudio.com/items?itemName=tauri-apps.tauri-vscode) + [rust-analyzer](https://marketplace.visualstudio.com/items?itemName=rust-lang.rust-analyzer).

### Building

```bash
# Development
pnpm tauri dev

# Production
pnpm tauri build --release
```

## Integration Examples

### OpenCode

```javascript
import { openai } from "@ai-sdk/openai-compatible";

const model = openai("your-model", {
  baseURL: "http://localhost:11434/v1"
});
```

### Python/LangChain

```python
from langchain.llms import OpenAI

llm = OpenAI(
  openai_api_base="http://localhost:11434/v1",
  openai_api_key="sk-dummy",
  model_name="your-model"
)
```

## File Structure

```
minerva/
├── src/                           # Svelte 5 frontend
│   └── routes/                   # SvelteKit pages
├── src-tauri/                     # Tauri/Rust backend
│   ├── src/
│   │   ├── lib.rs               # Entry point, Tauri setup
│   │   ├── main.rs              # Binary entry
│   │   ├── server.rs            # Axum HTTP server
│   │   ├── config.rs            # Configuration management
│   │   ├── commands.rs          # Tauri IPC commands
│   │   ├── error.rs             # Error handling
│   │   ├── models/
│   │   │   ├── mod.rs           # Model types & registry
│   │   │   ├── loader.rs        # GGUF discovery
│   │   │   └── gguf_parser.rs   # GGUF binary parsing
│   │   ├── inference/
│   │   │   ├── mod.rs              # Inference infrastructure
│   │   │   ├── llama_engine.rs     # Real inference wrapper
│   │   │   ├── llama_adapter.rs    # Backend abstraction layer
│   │   │   ├── gpu_context.rs      # GPU memory management
│   │   │   ├── token_stream.rs     # Token collection
│   │   │   ├── streaming.rs        # SSE formatting
│   │   │   ├── context_manager.rs  # Multi-model management
│   │   │   ├── parameters.rs       # Request validation
│   │   │   └── metrics.rs          # Performance tracking
│   └── Cargo.toml                  # Rust dependencies
├── tests/
│   └── integration_tests.rs         # 21 integration tests (Rust convention)
│       - model discovery tests
│       - inference pipeline tests
│       - backend abstraction tests
│       - parameter validation tests
│       - end-to-end workflow tests
├── docs/                            # Documentation
│   ├── PHASE_3_IMPLEMENTATION.md    # Phase 3 architecture
│   ├── PHASE_3_5_IMPLEMENTATION.md  # Phase 3.5 foundation
│   ├── PHASE_3_5A_COMPLETION.md     # Phase 3.5a completion
│   ├── PHASE_3_5B_PLAN.md           # Phase 3.5b implementation plan
│   ├── PHASE_3_5B_SESSION_SUMMARY.md # Phase 3.5b session 2 summary
│   ├── GPU_ACCELERATION.md          # GPU setup and tuning
│   ├── TEST_STRUCTURE.md            # Testing organization
│   ├── DEVELOPMENT.md               # Development guide
│   ├── SCRIPTS.md                   # Npm script reference
│   └── IMPLEMENTATION_PLAN.md        # Initial implementation plan
├── README.md                        # This file
└── pnpm scripts                     # Development helpers
```

## Testing

```bash
# Run all tests (135 total)
pnpm test

# Run with output
pnpm test:backend:watch

# Format and lint
pnpm fmt
pnpm lint

# Full validation
pnpm check:all
```

**Test Coverage:**
- Unit Tests: 101 tests (all modules)
- Integration Tests: 34 tests (organized by domain)
  - Model discovery (3 tests)
  - Inference engine (5 tests)
  - Token streaming (5 tests)
  - Backend abstraction (4 tests)
  - Parameter validation (4 tests)
  - GPU context (3 tests)
  - Error recovery (3 tests)
  - Performance tracking (4 tests)
  - End-to-end pipeline (3 tests)

Result: **135 tests passing, 0 warnings, 0 errors**

**Test Organization:**
- Unit tests in respective modules (src/*/mod.rs or mod.rs files)
- Integration tests in `tests/integration_tests.rs` (Rust convention)
- Run all tests: `pnpm test`
- Run unit tests only: `pnpm test:backend:unit`
- Run integration tests only: `pnpm test:backend:integration`

## Contributing

Follow engineering standards in `AGENTS.md`. All code must pass:
- Compilation with zero warnings
- All tests pass (pnpm test)
- Code complexity M ≤ 3
- Functions ≤ 25 lines
- Meaningful test assertions
- SOLID principles

## License

MIT
