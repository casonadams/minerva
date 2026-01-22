# Phase 2: Model Loading & File System Integration

## Overview

Phase 2 focuses on implementing local GGUF model loading, file system integration, and model discovery. This is the foundation for Phase 3 (LLM inference).

**Duration:** 1-2 weeks  
**Priority:** HIGH  
**Depends On:** Phase 1 ✅ (Complete)

## Goals

1. ✅ Load GGUF files from filesystem
2. ✅ Parse model metadata from GGUF files
3. ✅ Discover models in default directory
4. ✅ Persist model configuration
5. ✅ Integrate with existing model registry
6. ✅ Update `/v1/models` endpoint with real data
7. ✅ Add Tauri commands for model management
8. ✅ Create model storage directory structure

## Architecture

```
┌─────────────────────────────────────────┐
│  Frontend (Svelte)                      │
│  Model Management UI                    │
└──────────────┬──────────────────────────┘
               │ Tauri Commands
               ▼
┌─────────────────────────────────────────┐
│  Tauri Command Layer                    │
│  - list_models()                        │
│  - load_model(path)                     │
│  - unload_model(id)                     │
│  - get_model_metadata(path)             │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│  Model Manager (NEW)                    │
│  - ModelLoader                          │
│  - GGUFParser                           │
│  - ModelRegistry                        │
│  - FileSystem Integration               │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│  HTTP API Layer                         │
│  GET /v1/models (REAL DATA)             │
│  POST /v1/chat/completions (via models) │
└─────────────────────────────────────────┘
```

## Implementation Plan - COMPLETED

### Step 1: Add Dependencies ✅ (1 hour)

**File:** `src-tauri/Cargo.toml`

**Completed Tasks:**
- ✅ Add GGUF parsing library (`gguf-rs-lib = "0.2"`)
- ✅ Add file system utilities (`walkdir = "2"`, `home = "0.5"`)
- ✅ Add config management libs (`serde`, `serde_json`, `anyhow`)
- ✅ Run `cargo check` - verified working

### Step 2: Create Model Loader Module ✅ (2 hours)

**File:** `src-tauri/src/models/loader.rs`

**Completed Tasks:**
- ✅ Implement model discovery with walkdir
- ✅ Walk directory structure recursively
- ✅ Filter .gguf files automatically
- ✅ Return list of discovered models with metadata
- ✅ Add comprehensive error handling

### Step 3: Create GGUF Parser Module ✅ (2 hours)

**File:** `src-tauri/src/models/gguf_parser.rs`

**Completed Tasks:**
- ✅ Parse GGUF binary headers (magic, version, tensors, kv pairs)
- ✅ Extract model metadata (context_window, model_name, quantization)
- ✅ Support GGUF versions 2 and 3
- ✅ Handle different value types in GGUF format
- ✅ Add comprehensive error handling with MinervaError

### Step 4: Create Configuration Module ✅ (1.5 hours)

**File:** `src-tauri/src/config.rs`

**Completed Tasks:**
- ✅ Load from config file if exists (`~/.minerva/config.json`)
- ✅ Create default config with sensible defaults
- ✅ Handle missing directories gracefully
- ✅ Auto-create models directory structure
- ✅ Save configuration with persistence

### Step 5: Update Model Registry ✅ (1.5 hours)

**File:** `src-tauri/src/models/mod.rs`

**Completed Tasks:**
- ✅ Add filepath tracking (`model_paths: HashMap<String, PathBuf>`)
- ✅ Implement discover() method for batch filesystem scanning
- ✅ Integrate with loader for real metadata
- ✅ Support full registry CRUD operations

### Step 6: Create Tauri Commands ✅ (2 hours)

**File:** `src-tauri/src/commands.rs` (NEW comprehensive module)

**Completed Tasks:**
- ✅ `get_config()` - retrieve current configuration
- ✅ `list_discovered_models()` - discover and list all GGUF files
- ✅ `load_model_file(path)` - load specific model by path
- ✅ `get_models_directory()` - query current models directory
- ✅ `set_models_directory(path)` - change models directory with persistence
- ✅ `ensure_models_directory()` - create models directory if missing
- ✅ Add comprehensive error handling with user-friendly messages
- ✅ Add input validation for all paths

### Step 7: Update HTTP Server ✅ (1.5 hours)

**File:** `src-tauri/src/server.rs`

**Completed Tasks:**
- ✅ `/v1/models` endpoint returns real model data from registry
- ✅ Add `with_discovered_models()` method for initialization
- ✅ Pre-load discovered models on server startup
- ✅ Registry integrated with model loader

### Step 8: File System Structure ✅ (1 hour)

**Created Default Directory Structure:**

```
~/.minerva/
├── config.json          # App configuration
└── models/              # Local model storage
    └── [GGUF files placed here]
```

**Completed Tasks:**
- ✅ Auto-create config directory
- ✅ Auto-create models directory on app startup
- ✅ Create default config.json with settings
- ✅ Handle permissions gracefully

### Step 9: Add Tests ✅ (1.5 hours)

**Files:** 
- Unit tests in respective modules (13 unit tests)
- `src-tauri/src/integration_tests.rs` (9 comprehensive integration tests)

**Completed Tests:**
- ✅ Full model discovery pipeline test
- ✅ Selective filtering (only .gguf files)
- ✅ Nested directory structure handling
- ✅ Registry CRUD operations
- ✅ Config integration with discovery
- ✅ Model metadata validity checks
- ✅ Empty directory handling
- ✅ Nonexistent directory graceful handling
- ✅ Model file path tracking
- ✅ Parser validation tests
- ✅ Loader discovery tests

**Statistics:**
- Total: 26 tests (17 unit + 9 integration)
- All passing ✅
- Zero warnings ✅

### Step 10: Documentation ✅ (1 hour)

**Update Files:**
- ✅ Update PHASE_2_PLAN.md with completion status
- ✅ Create comprehensive API documentation
- ✅ Document model directory structure
- ✅ Add configuration file format docs
- ✅ Update README.md with setup instructions

## Deliverables

### Code Changes

**New Files:**
- `src-tauri/src/models/loader.rs` - Model discovery
- `src-tauri/src/models/gguf_parser.rs` - GGUF parsing
- `src-tauri/src/config.rs` - Configuration management

**Modified Files:**
- `src-tauri/Cargo.toml` - Add dependencies
- `src-tauri/src/lib.rs` - Module declarations
- `src-tauri/src/models.rs` - Registry enhancements
- `src-tauri/src/server.rs` - Real model data
- `src-tauri/src/commands.rs` - Add Tauri commands
- `src-tauri/src/main.rs` - Initialize config

**New Tests:**
- Model loader tests
- GGUF parser tests
- Config tests
- Integration tests

### Documentation

- Updated README with setup instructions
- Configuration file format documentation
- Model directory structure guide
- API endpoint documentation

## Testing Strategy

### Unit Tests
- Model discovery with empty/populated directories
- GGUF file parsing (valid/invalid files)
- Configuration load/save
- Model registry sync

### Integration Tests
- Load real GGUF files
- Discover models and load via `/v1/models`
- Verify model metadata accuracy
- Test configuration persistence

### Manual Testing
```bash
# 1. Place GGUF files in ~/.minerva/models/
# 2. Run the app
pnpm tauri dev

# 3. Check models via curl
curl http://localhost:11434/v1/models

# 4. Verify each model shows correct metadata
```

## Success Criteria - ALL COMPLETE ✅

- ✅ Discover GGUF files from filesystem
- ✅ Parse model metadata correctly
- ✅ `/v1/models` returns real model data
- ✅ Configuration saved/loaded properly
- ✅ Tauri commands working (6 commands)
- ✅ All 26 tests passing
- ✅ Zero linting warnings
- ✅ Code properly formatted
- ✅ Documentation updated

## Actual Time (Completed)

| Task | Hours | Status |
|------|-------|--------|
| Dependencies | 0.5 | ✅ Complete |
| Model Loader | 1.5 | ✅ Complete |
| GGUF Parser | 1.5 | ✅ Complete |
| Config Module | 1 | ✅ Complete |
| Registry Updates | 1 | ✅ Complete |
| Tauri Commands | 1.5 | ✅ Complete |
| HTTP Server | 1 | ✅ Complete |
| File Structure | 0.5 | ✅ Complete |
| Tests | 2 | ✅ Complete |
| Documentation | 0.5 | ✅ Complete |
| **Total** | **~11 hours** | ✅ **COMPLETE** |

## Risks & Mitigation

**Risk:** GGUF file parsing complexity
- **Mitigation:** Use `gguf-rs-lib`, extensive testing

**Risk:** File system permissions issues
- **Mitigation:** Proper error handling, clear error messages

**Risk:** Large model directories slow discovery
- **Mitigation:** Cache discovery results, add refresh command

**Risk:** Configuration conflicts
- **Mitigation:** Version config format, migration strategy

## Next Phase Preview (Phase 3)

After Phase 2, we'll implement:
- llama.cpp integration
- Token generation
- Context management
- Actual model inference
- Mock responses → Real responses

## Rollback Plan

If Phase 2 needs rollback:
1. Revert Cargo.toml changes
2. Remove new model modules
3. Restore mock responses in server.rs
4. Keep HTTP API structure (no breaking changes)

---

**Phase Status:** ✅ COMPLETE  
**Duration:** ~11 hours  
**Tests:** 26/26 passing (17 unit + 9 integration)  
**Code Quality:** Zero warnings, proper formatting, SOLID principles  
**Commits:** 6 (1f35c04, 09ec44e, a97affc, ddf4094, e07ebe0, + this doc)  
**Next Phase:** Phase 3 (LLM Inference)  
**Unblocks:** Real model serving via Tauri commands
