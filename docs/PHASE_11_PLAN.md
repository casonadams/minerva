# Phase 11: REST API Decoupling & Headless Server

## Executive Summary

**Goal**: Decouple the Rust backend server from the GUI, enabling:
- Headless API server deployments (no GUI required)
- HTTP/REST API for all backend communication (instead of Tauri IPC)
- Independent CLI tool with `serve` subcommand
- Multi-client capable server (web, mobile, desktop apps)
- Zero GUI dependencies in server binary

**Current State** (Phase 10):
- REST API endpoints exist but are only accessible via Tauri
- Frontend uses IPC commands (`invoke()`) for backend communication
- Server is tightly coupled to the Tauri desktop application
- No standalone server binary possible

**Target State** (Phase 11):
- Standalone binary: `minerva serve --host 0.0.0.0 --port 3000`
- Frontend communicates via HTTP to localhost:3000
- Same server can be deployed headless in Docker/K8s
- CLI tool for model management, configuration, serving

---

## Architecture Overview

### Current (Phase 10)
```
┌─────────────────────────────────┐
│   Tauri Application             │
├─────────────────────────────────┤
│ Frontend (Svelte + TS)          │
│   ↓ IPC (invoke)                │
│ Backend Commands (Rust)         │
│   ↓ Direct calls                │
│ Inference Engine                │
│   ↓ Model execution             │
│ ML Models                        │
└─────────────────────────────────┘
```

### Desired (Phase 11)
```
┌──────────────────┐
│   Tauri Desktop  │
│   Frontend       │
│   ↓ HTTP        │
└──────────────────┘
         ↓
    localhost:5173
         ↓
┌──────────────────────────────────────┐
│   REST API Server (Port 3000)        │
│   Can run standalone OR in Tauri     │
├──────────────────────────────────────┤
│ /v1/chat/completions                 │
│ /v1/models (list, load, unload)      │
│ /health, /ready, /metrics            │
│ /v1/model/download (w/ progress)     │
├──────────────────────────────────────┤
│ Inference Engine                      │
│ Model Registry                        │
│ Rate Limiting                         │
│ Caching                               │
└──────────────────────────────────────┘
         ↑
    ┌────┴───────┐
    ↓            ↓
┌─────────┐  ┌─────────────┐
│  Web    │  │ Mobile App  │
│  Client │  │ (iOS/Andr.) │
└─────────┘  └─────────────┘
```

---

## Phase 11 Implementation Plan

### Day 1: CLI Framework & Serve Command

**Objective**: Create standalone CLI tool with serve subcommand

**New Files**:
- `src-tauri/src/bin/minerva.rs` (60 lines) - CLI entry point
- `src-tauri/src/cli/mod.rs` (10 lines) - CLI module exports
- `src-tauri/src/cli/serve.rs` (80 lines) - `serve` subcommand logic
- `src-tauri/src/cli/config_cmd.rs` (60 lines) - `config` subcommand
- `src-tauri/src/cli/model_cmd.rs` (70 lines) - `model` subcommand

**Key Features**:
```bash
# Serve on default port 3000
minerva serve

# Custom host/port
minerva serve --host 0.0.0.0 --port 8000

# With config file
minerva serve --config /etc/minerva/config.toml

# With models directory
minerva serve --models-dir /var/lib/minerva/models

# List/manage models
minerva model list
minerva model download <model-id> --dest /models/

# Show/edit configuration
minerva config show
minerva config init
```

**Tests**:
- 5 tests: CLI argument parsing, subcommand routing, config loading
- 3 tests: Default values, config overrides, validation

**Standards**:
- ✅ Each file ≤ 150 lines
- ✅ Functions ≤ 25 lines
- ✅ Tests for public APIs

---

### Day 2: HTTP Client for Frontend

**Objective**: Replace Tauri IPC with HTTP API calls

**New Files**:
- `src/lib/api/client.ts` (120 lines) - HTTP client with retries
- `src/lib/api/endpoints.ts` (80 lines) - Endpoint definitions
- `src/lib/api/types.ts` (60 lines) - API response types
- `src/lib/composables/useApi.ts` (100 lines) - Svelte composable

**Frontend Changes**:
- Replace `invoke('command')` with `api.post('/v1/chat/completions')`
- Add error handling, timeouts, retries
- Environmental config (dev: localhost:3000, prod: /api)
- WebSocket fallback for streaming (Phase 12)

**Key Features**:
```typescript
// Old Tauri way
const response = await invoke('infer_prompt', { prompt: 'Hi' });

// New HTTP way
const response = await api.post('/v1/chat/completions', {
  model: 'llama-2-7b',
  messages: [{ role: 'user', content: 'Hi' }]
});
```

**Testing**:
- 8 tests: Happy path, error handling, retries
- 3 tests: Timeout, offline mode, request cancellation
- 2 tests: Header injection, auth token support (ready for Phase 12)

---

### Day 3: Decouple Tauri from Core

**Objective**: Move inference logic out of Tauri

**Changes**:
- `src-tauri/src/lib.rs`: Remove Tauri-specific code from `run()`
- Create `src-tauri/src/main_gui.rs` - Tauri entry point (standalone)
- Create `src-tauri/src/main_cli.rs` - CLI entry point (standalone)
- Keep inference logic in `inference/` module (no Tauri dependencies)

**Architecture**:
```
src-tauri/src/
├── lib.rs (core library, no Tauri)
├── main_gui.rs (GUI mode: Tauri app)
├── main_cli.rs (CLI mode: bin/minerva)
├── bin/
│   └── minerva.rs (entry point for CLI)
├── inference/ (✅ Already independent)
├── middleware/ (✅ Already independent)
├── server/ (✅ Already independent)
└── commands/ (Tauri-specific, stays here)
```

**Cargo.toml Updates**:
```toml
[[bin]]
name = "minerva"
path = "src/bin/minerva.rs"

[features]
default = ["tauri"]
gui = ["tauri", "tauri-plugin-opener"]
cli = []
full = ["gui", "cli"]
```

**Tests**:
- 3 integration tests: Server startup, inference pipeline, model loading
- No Tauri imports in core library tests

---

### Day 4: API Protocol Unification

**Objective**: Ensure API works for both GUI and headless

**New Files**:
- `src-tauri/src/api_server/mod.rs` - Unified HTTP server
- `src-tauri/src/api_server/startup.rs` - Server initialization
- `src-tauri/src/api_server/routes.rs` - Route registration

**Key Points**:
- Single Router struct works in both modes
- Tauri mode: Starts on localhost:3000, only accessible locally
- CLI mode: Configurable host/port, accessible remotely
- Same endpoints, same validation, same rate limiting

**Features**:
- Server health checks
- Graceful shutdown (signal handling)
- Connection pooling for concurrent requests
- Structured logging with tracing

**Tests**:
- 10 integration tests: Route handling, concurrency, shutdown
- Load testing validation (1000 req/sec targets)

---

### Day 5: Streaming & WebSocket Support (Foundation)

**Objective**: Prepare for WebSocket streaming (Phase 12 implementation)

**New Files**:
- `src-tauri/src/api_server/streaming.rs` (80 lines) - SSE/WS handlers
- `src-tauri/src/api_server/ws.rs` (100 lines) - WebSocket types

**Features**:
- Upgrade HTTP streaming routes to support WS
- Keep SSE as fallback for browsers
- Frame-based protocol for streaming responses
- Error handling and reconnection support

**Frontend Ready**:
```typescript
// Detect server capability at startup
async function detectStreamingMode() {
  const res = await fetch('/v1/capabilities');
  const { supports_websocket } = await res.json();
  useWebSocket = supports_websocket;
}
```

**Tests**:
- 5 tests: Stream initiation, frame parsing, error recovery
- 3 tests: WebSocket upgrade negotiation, close handling

---

### Day 6: Configuration Management

**Objective**: Unified config for CLI and GUI modes

**New Files**:
- `src-tauri/src/config/server_config.rs` (90 lines) - Server config
- `src-tauri/src/config/mod.rs` (20 lines) - Config exports

**Features**:
```toml
[server]
host = "127.0.0.1"
port = 3000
workers = 4
mode = "api"  # "api" or "gui"

[models]
directory = "/models"
max_loaded = 3
download_timeout = "600s"

[inference]
default_model = "llama-2-7b"
temperature = 0.7
top_p = 1.0
max_tokens = 2048
```

**Loading Order**:
1. Built-in defaults
2. `/etc/minerva/config.toml` (system-wide)
3. `~/.minerva/config.toml` (user)
4. `./minerva.toml` (current directory)
5. Environment variables (`MINERVA_*`)
6. CLI flags (highest priority)

**Tests**:
- 8 tests: Config loading, merging, validation
- 3 tests: Environment variable override

---

### Day 7: Testing & Validation

**Objective**: Comprehensive testing for decoupled architecture

**Test Suites**:

**1. Unit Tests** (existing, enhanced):
- Middleware: 30 tests (rate limiting, validation)
- Inference: 850 tests (existing)
- Server handlers: 20 new tests

**2. Integration Tests**:
- API server startup: 5 tests
- HTTP endpoint handling: 10 tests
- Client-server communication: 10 tests
- Concurrent request handling: 5 tests

**3. E2E Tests** (new):
- GUI → API communication: 5 tests
- CLI → API communication: 3 tests
- Headless server operation: 5 tests
- Model loading via API: 3 tests

**4. Load Tests**:
- 100 concurrent requests
- 1000 req/sec throughput target
- Rate limiting enforcement
- Memory stability (30 min burn test)

**5. Deployment Tests**:
- Docker build and run
- Kubernetes pod startup
- Configuration via env vars
- Health check functionality

---

### Day 8: Documentation & Release

**Objective**: Complete documentation for decoupled system

**New Documentation**:
- `docs/CLI.md` - CLI usage guide
- `docs/ARCHITECTURE.md` - System architecture
- `docs/MIGRATION_GUIDE.md` - IPC → HTTP migration
- `docs/QUICKSTART_HEADLESS.md` - Headless deployment

**Updates**:
- Update `docs/API.md` with WebSocket endpoints
- Update `docs/DEPLOYMENT.md` with CLI examples
- Create `docs/PERFORMANCE.md` with benchmarks

**Release Artifacts**:
- Binary: `minerva` (standalone CLI)
- Binary: `minerva-gui` (Tauri app)
- Docker image: `minerva:latest`
- Helm chart: `minerva/` directory

---

## Feature Comparison: Phase 10 vs Phase 11

| Feature | Phase 10 | Phase 11 |
|---------|----------|----------|
| REST API | ✅ (Tauri-bound) | ✅ (Standalone) |
| CLI Tool | ❌ | ✅ |
| Headless Server | ❌ | ✅ |
| HTTP Endpoints | ✅ (IPC only) | ✅ (Full REST) |
| Rate Limiting | ✅ | ✅ |
| Authentication | ❌ | ❌ (Phase 12) |
| WebSocket | ❌ | 🔄 (Foundation) |
| Server Binary | ❌ | ✅ |
| Docker Support | ❌ | ✅ |
| Kubernetes | ❌ | ✅ |
| GUI Mode | ✅ | ✅ |
| Web Client | ❌ | 🔄 (Ready for) |

---

## Technical Decisions

### 1. Single Code Base or Separate?
**Decision**: Single code base with feature flags
- Pros: Shared inference logic, easier maintenance
- Cons: Build complexity
- Solution: `features = ["gui", "cli"]` in Cargo.toml

### 2. IPC or Drop Completely?
**Decision**: Keep IPC for fast GUI-server communication
- Tauri GUI → localhost:3000 via HTTP (or keep IPC for performance)
- HTTP always available for external clients
- Phase 12: Optional authentication middleware

### 3. Port & Binding
**Decision**: CLI uses port 3000, GUI uses localhost-only
- CLI: `0.0.0.0:3000` (accept all interfaces)
- GUI: `127.0.0.1:3000` (localhost only, more secure)
- Configurable via `--host` flag

### 4. Configuration File Location
**Decision**: XDG Base Directory spec (Linux/Mac/Windows compatible)
- Linux: `~/.config/minerva/config.toml`
- macOS: `~/Library/Application Support/minerva/config.toml`
- Windows: `%APPDATA%\minerva\config.toml`

---

## Testing Strategy

### Unit Tests (Target: 50+ new tests)
- CLI argument parsing
- Configuration loading and merging
- Route registration
- Streaming protocol handling

### Integration Tests (Target: 30+ new tests)
- Full request/response cycles
- Rate limiting across HTTP connections
- Model loading via API
- Concurrent client handling

### E2E Tests (Target: 15+ new tests)
- Tauri GUI → API communication
- CLI tool → API communication
- Headless server operation
- Docker container startup

### Performance Tests
- Throughput: 1000+ req/sec
- Latency: p99 < 500ms
- Concurrent clients: 100+
- Memory: Stable over 30 minutes

---

## Success Criteria

✅ **Phase 11 Complete When**:

1. **Standalone Binary**
   - `minerva serve --help` shows usage
   - `minerva serve` starts API on port 3000
   - Server responds to `curl localhost:3000/health`

2. **CLI Tool**
   - `minerva model list` shows loaded models
   - `minerva model download` fetches models
   - `minerva config show` displays configuration
   - All commands have help text

3. **Frontend Decoupled**
   - Frontend makes HTTP calls instead of IPC
   - Works with both embedded server and external API
   - Environment variable controls API endpoint

4. **Tests**
   - 1,200+ total tests (1,065 from Phase 10 + 135 new)
   - 100% pass rate
   - Coverage: CLI, API, configuration

5. **Documentation**
   - API docs updated with HTTP examples
   - CLI usage guide complete
   - Architecture diagram documented
   - Headless deployment guide included

6. **Deployable**
   - Docker image builds successfully
   - Kubernetes manifest provided
   - Systemd service config included
   - Health checks work from external clients

---

## Estimated Effort

- **Days**: 8 (Days 1-8 of Phase 11)
- **Commits**: 8-10 focused commits
- **New Tests**: 135+ (target: 1,200 total)
- **New Code**: ~1,500 lines (CLI, config, tests)
- **Documentation**: 3 new guides + 2 updated guides

---

## Risk Mitigation

### Risk: Breaking GUI while refactoring
**Mitigation**: 
- Keep Tauri code in separate module
- Comprehensive integration tests
- Feature flags allow building GUI or CLI independently

### Risk: HTTP overhead vs IPC
**Mitigation**:
- Localhost HTTP is fast (< 1ms)
- Keep IPC as optional optimization (Phase 12)
- Benchmark before/after

### Risk: Configuration complexity
**Mitigation**:
- Sensible defaults work without config file
- Clear precedence rules (CLI > env > file > defaults)
- Validation and clear error messages

### Risk: Streaming changes break HTTP
**Mitigation**:
- SSE as fallback (always works)
- WebSocket as enhancement (Phase 12)
- Comprehensive streaming tests

---

## Next Steps

1. Review this plan with team
2. Identify any blockers or concerns
3. Begin Phase 11 Day 1: CLI Framework
4. Maintain test-driven development approach
5. Update documentation as we go

---

## Appendix: File Structure (Phase 11)

```
src-tauri/src/
├── lib.rs                    # Core library (no Tauri)
├── bin/
│   └── minerva.rs           # CLI binary entry point
├── main.rs                  # (Tauri entry point, untouched)
│
├── cli/                      # NEW: CLI module
│   ├── mod.rs
│   ├── serve.rs
│   ├── config_cmd.rs
│   └── model_cmd.rs
│
├── config/                   # NEW: Configuration module
│   ├── mod.rs
│   └── server_config.rs
│
├── api_server/               # NEW: HTTP server decoupling
│   ├── mod.rs
│   ├── startup.rs
│   ├── routes.rs
│   ├── streaming.rs
│   └── ws.rs
│
├── inference/                # ✅ Already independent
├── middleware/               # ✅ Already independent
├── server/                   # ✅ Already independent
└── commands/                 # Tauri-specific
```

---

*Last Updated: Phase 10 Day 8*
*Next Review: Phase 11 kickoff*
