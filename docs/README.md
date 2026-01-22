# Minerva Documentation

Complete documentation for the Minerva project - a local LLM server built with Tauri and Rust.

## Quick Navigation

### Project Overview & Planning
- **[IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md)** - Initial high-level implementation strategy
- **[DEVELOPMENT.md](DEVELOPMENT.md)** - Development guide and setup instructions

### Phase Documentation

#### Phase 1: Foundation ✅
- **[PHASE_1_COMPLETE.md](PHASE_1_COMPLETE.md)** - Foundation phase completion details

#### Phase 2: Model Loading ✅
- **[PHASE_2_PLAN.md](PHASE_2_PLAN.md)** - Model loading and file system implementation

#### Phase 3: Inference Engine ✅
- **[PHASE_3_PLAN.md](PHASE_3_PLAN.md)** - Phase 3 planning document
- **[PHASE_3_IMPLEMENTATION.md](PHASE_3_IMPLEMENTATION.md)** - Phase 3 architecture and implementation

#### Phase 3.5: Real LLM Integration ✅
- **[PHASE_3_5_IMPLEMENTATION.md](PHASE_3_5_IMPLEMENTATION.md)** - Foundation: LlamaEngine, GPU context, token streams

#### Phase 3.5a: Backend Abstraction ✅
- **[PHASE_3_5A_COMPLETION.md](PHASE_3_5A_COMPLETION.md)** - Backend trait, MockBackend, abstraction layer

#### Phase 3.5b: Real llama.cpp Integration 🔄
- **[PHASE_3_5B_PLAN.md](PHASE_3_5B_PLAN.md)** - 8-step implementation roadmap for GPU and real inference
- **[PHASE_3_5B_SESSION_SUMMARY.md](PHASE_3_5B_SESSION_SUMMARY.md)** - Session 2 completion: Real backend + thread safety

### Technical Guides

- **[GPU_ACCELERATION.md](GPU_ACCELERATION.md)** - GPU setup, optimization, platform-specific configuration
- **[TEST_STRUCTURE.md](TEST_STRUCTURE.md)** - Test organization, running tests, coverage details
- **[SCRIPTS.md](SCRIPTS.md)** - npm/pnpm script reference and usage

## Current Status

**103 Tests Passing** (82 unit + 21 integration)
- All code compilation with zero warnings
- 100% format compliance
- Zero clippy warnings

### Active Development

**Phase 3.5b: Real llama.cpp Integration**
- ✅ Step 1: Research
- ✅ Step 2: Real backend implementation (Session 2 COMPLETE)
- ⏳ Step 3-8: GPU, streaming, error handling, benchmarking

## File Organization

```
docs/
├── README.md (this file)
├── IMPLEMENTATION_PLAN.md
├── DEVELOPMENT.md
├── SCRIPTS.md
├── PHASE_1_COMPLETE.md
├── PHASE_2_PLAN.md
├── PHASE_3_PLAN.md
├── PHASE_3_IMPLEMENTATION.md
├── PHASE_3_5_IMPLEMENTATION.md
├── PHASE_3_5A_COMPLETION.md
├── PHASE_3_5B_PLAN.md
├── PHASE_3_5B_SESSION_SUMMARY.md
├── GPU_ACCELERATION.md
└── TEST_STRUCTURE.md
```

## Key Architecture Concepts

### Backend Abstraction
- **InferenceBackend trait** - Pluggable inference backend interface
- **LlamaCppBackend** - Real llama.cpp integration with GPU acceleration
- **MockBackend** - Testing backend with intelligent mocking

### Thread Safety
- Arc<Mutex<>> for shared model/session access
- Send + Sync traits enforced on backend implementations
- Safe concurrent inference support

### GPU Acceleration
- 40 layer GPU offloading (configurable)
- Metal auto-detection on macOS
- CUDA support on Linux/Windows
- CPU fallback always available

## Quick Links

- **Main README**: [../README.md](../README.md)
- **Source Code**: [../src-tauri/src/](../src-tauri/src/)
- **Tests**: [../tests/](../tests/)
- **Config**: [../src-tauri/Cargo.toml](../src-tauri/Cargo.toml)

## For New Contributors

1. Start with **[DEVELOPMENT.md](DEVELOPMENT.md)** for setup
2. Read **[PHASE_3_5B_PLAN.md](PHASE_3_5B_PLAN.md)** for current work
3. Review **[TEST_STRUCTURE.md](TEST_STRUCTURE.md)** for testing patterns
4. Check **[GPU_ACCELERATION.md](GPU_ACCELERATION.md)** for GPU details

## Engineering Standards

All code follows strict standards:
- **Complexity**: M ≤ 3 (cyclomatic) per function
- **Length**: Functions ≤ 25 lines, files ≤ 100 lines
- **Testing**: Meaningful assertions, ≥1 test per public method
- **Build**: Zero warnings, all tests passing
- **SOLID**: All 5 principles respected

See root `AGENTS.md` for complete standards.

## API Reference

### HTTP Endpoints
- `GET /v1/models` - List available models
- `POST /v1/chat/completions` - Chat completion with streaming
- `POST /v1/completions` - Text completion (stub)

### Tauri Commands
- `get_config()` - Get configuration
- `list_discovered_models()` - Discover GGUF files
- `load_model_file()` - Load specific model
- `get/set_models_directory()` - Manage model storage

## Common Tasks

### Running Tests
```bash
pnpm test              # All tests
pnpm test:backend:unit # Unit only
pnpm test:backend:integration # Integration only
```

### Format & Lint
```bash
pnpm fmt:backend       # Format Rust code
pnpm lint:backend      # Clippy linting
pnpm check:all         # Full validation
```

### Development
```bash
pnpm tauri dev         # Dev server
pnpm tauri build --release # Production build
```

## Contact & Support

For questions about specific phases or components:
- Check the relevant phase documentation
- Review test examples in `tests/`
- See inline code comments in source files

---

**Last Updated**: Phase 3.5b Session 2 (Real llama.cpp Integration)  
**Tests**: 103/103 passing ✅  
**Build**: Clean ✅
