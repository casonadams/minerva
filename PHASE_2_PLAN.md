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

## Implementation Plan

### Step 1: Add Dependencies (1 hour)

**File:** `src-tauri/Cargo.toml`

```toml
# GGUF parsing
gguf-rs-lib = "0.2"

# File system utilities
walkdir = "2"
lazy_static = "1.4"

# Config management
serde_json = "1"
```

**Tasks:**
- [ ] Add GGUF parsing library
- [ ] Add file system utilities
- [ ] Add config management libs
- [ ] Run `cargo check` to verify

### Step 2: Create Model Loader Module (2 hours)

**New File:** `src-tauri/src/models/loader.rs`

```rust
pub struct ModelLoader {
    models_dir: PathBuf,
}

impl ModelLoader {
    pub fn new(models_dir: PathBuf) -> Self { ... }
    pub fn discover_models(&self) -> MinervaResult<Vec<ModelInfo>> { ... }
    pub fn load_model(&self, path: &Path) -> MinervaResult<ModelInfo> { ... }
}
```

**Tasks:**
- [ ] Implement model discovery
- [ ] Walk directory structure
- [ ] Filter .gguf files
- [ ] Return list of discovered models
- [ ] Add error handling

### Step 3: Create GGUF Parser Module (2 hours)

**New File:** `src-tauri/src/models/gguf_parser.rs`

```rust
pub struct GGUFParser;

impl GGUFParser {
    pub fn parse_metadata(path: &Path) -> MinervaResult<GGUFMetadata> { ... }
    pub fn extract_model_info(path: &Path) -> MinervaResult<ModelInfo> { ... }
}

pub struct GGUFMetadata {
    pub name: String,
    pub context_window: usize,
    pub model_size: u64,
    pub version: String,
}
```

**Tasks:**
- [ ] Parse GGUF file headers
- [ ] Extract model metadata
- [ ] Handle different GGUF versions
- [ ] Calculate context windows
- [ ] Add error handling

### Step 4: Create Configuration Module (1.5 hours)

**New File:** `src-tauri/src/config.rs`

```rust
pub struct AppConfig {
    pub models_dir: PathBuf,
    pub server_port: u16,
    pub server_host: String,
    pub gpu_enabled: bool,
}

impl AppConfig {
    pub fn load() -> MinervaResult<Self> { ... }
    pub fn load_or_default() -> Self { ... }
    pub fn save(&self) -> MinervaResult<()> { ... }
}
```

**Tasks:**
- [ ] Load from config file if exists
- [ ] Create default config
- [ ] Handle missing directories
- [ ] Create models directory
- [ ] Save configuration

### Step 5: Update Model Registry (1.5 hours)

**File:** `src-tauri/src/models.rs`

**Changes:**
```rust
pub struct ModelRegistry {
    models: HashMap<String, ModelInfo>,
    // NEW: Track file paths
    model_paths: HashMap<String, PathBuf>,
}

impl ModelRegistry {
    // NEW: Load from filesystem
    pub fn discover(&mut self, models_dir: &Path) -> MinervaResult<()> { ... }
    
    // NEW: Sync with filesystem
    pub fn sync(&mut self) -> MinervaResult<()> { ... }
}
```

**Tasks:**
- [ ] Add filepath tracking
- [ ] Implement discover method
- [ ] Implement sync method
- [ ] Add persistence layer

### Step 6: Create Tauri Commands (2 hours)

**File:** `src-tauri/src/commands.rs` (extend existing)

```rust
#[tauri::command]
pub async fn list_models_with_discovery() -> Result<Vec<ModelInfo>, String> { ... }

#[tauri::command]
pub async fn load_model_from_file(path: String) -> Result<ModelInfo, String> { ... }

#[tauri::command]
pub async fn get_models_directory() -> Result<String, String> { ... }

#[tauri::command]
pub async fn set_models_directory(path: String) -> Result<(), String> { ... }

#[tauri::command]
pub async fn refresh_models() -> Result<Vec<ModelInfo>, String> { ... }
```

**Tasks:**
- [ ] Implement all commands
- [ ] Add error handling
- [ ] Add input validation
- [ ] Test with Tauri invoke

### Step 7: Update HTTP Server (1.5 hours)

**File:** `src-tauri/src/server.rs`

**Changes:**
```rust
// Get real models from registry instead of mocks
async fn list_models(State(state): State<ServerState>) 
    -> MinervaResult<Json<ModelsListResponse>> {
    let registry = state.model_registry.lock().await;
    let models = registry.list_models();
    // ... return real data
}
```

**Tasks:**
- [ ] Update `/v1/models` to use real data
- [ ] Remove mock responses
- [ ] Add model filtering
- [ ] Add sorting by name

### Step 8: File System Structure (1 hour)

**Create Default Directory Structure:**

```
~/.minerva/
├── config.json          # App configuration
└── models/              # Local model storage
    ├── mistral-7b.gguf
    ├── llama-2-7b.gguf
    └── ...
```

**Tasks:**
- [ ] Create config directory
- [ ] Create models directory
- [ ] Create default config.json
- [ ] Handle permissions

### Step 9: Add Tests (1.5 hours)

**New File:** `src-tauri/src/models/loader_tests.rs`

**Test Cases:**
```rust
#[test]
fn test_discover_models_empty_dir() { ... }

#[test]
fn test_discover_models_with_gguf_files() { ... }

#[test]
fn test_gguf_parser_valid_file() { ... }

#[test]
fn test_gguf_parser_invalid_file() { ... }

#[test]
fn test_config_load_default() { ... }

#[test]
fn test_config_persistence() { ... }

#[test]
fn test_model_registry_sync() { ... }
```

**Tasks:**
- [ ] Write unit tests
- [ ] Test model discovery
- [ ] Test GGUF parsing
- [ ] Test configuration
- [ ] All tests pass

### Step 10: Documentation (1 hour)

**Update Files:**
- [ ] Update README.md with setup instructions
- [ ] Add GGUF file handling documentation
- [ ] Document model directory structure
- [ ] Add configuration file format docs
- [ ] Update API documentation

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

## Success Criteria

- ✅ Discover GGUF files from filesystem
- ✅ Parse model metadata correctly
- ✅ `/v1/models` returns real model data
- ✅ Configuration saved/loaded properly
- ✅ Tauri commands working
- ✅ All tests passing
- ✅ Zero linting warnings
- ✅ Code properly formatted
- ✅ Documentation updated

## Estimated Time

| Task | Hours |
|------|-------|
| Dependencies | 1 |
| Model Loader | 2 |
| GGUF Parser | 2 |
| Config Module | 1.5 |
| Registry Updates | 1.5 |
| Tauri Commands | 2 |
| HTTP Server | 1.5 |
| File Structure | 1 |
| Tests | 1.5 |
| Documentation | 1 |
| **Total** | **~15 hours** |

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

**Phase Status:** Ready to Begin  
**Blocks:** Phase 3 (LLM Inference)  
**Unblocks:** Real model serving
