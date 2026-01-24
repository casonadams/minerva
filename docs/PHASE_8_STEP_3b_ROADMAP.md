# Phase 8-Step 3b: MLX Subprocess Integration Roadmap

**Status:** Ready to Start  
**Estimated Duration:** 2-3 days  
**Prerequisites:** MLX Backend Foundation (✅ Complete)  
**Target Tests:** 806 → 830+ (add 24+ integration tests)  
**Target Lint:** 0 violations (maintained)  

---

## Overview

**Goal:** Connect the MlxBackend foundation to actual mlx-lm via subprocess, enabling real HuggingFace model inference.

**What Exists:**
- ✅ MlxBackend struct fully defined
- ✅ InferenceBackend trait implemented
- ✅ Model format detection working
- ✅ 8 unit tests proving foundation

**What We'll Add:**
- 🔨 Real subprocess integration with mlx-lm
- 🔨 HTTP client for mlx-lm server API
- 🔨 Model loading via subprocess
- 🔨 Caching for loaded models
- 🔨 Error recovery and fallbacks
- 🔨  24+ integration tests

---

## Implementation Roadmap

### Day 1: Research & Architecture

#### Morning: Environment Setup (1 hour)
```bash
# Install mlx-lm
pip install mlx-lm

# Test mlx-lm is working
python3 -c "import mlx_lm; print('✅ mlx-lm available')"

# Explore mlx-lm CLI
python3 -m mlx_lm --help

# Try serving a model
python3 -m mlx_lm.server --help
```

**Expected Output:** Help text showing available commands and options

**Key Discoveries:**
- Default port for mlx-lm server
- Available model formats
- Command-line options
- Response format

#### Afternoon: Design Subprocess Integration (2 hours)

**Current Scaffolding (in mlx_backend.rs):**
```rust
#[allow(dead_code)]
fn run_mlx_command(&self, args: &[&str]) -> MinervaResult<String> {
    // Phase 9: Will run actual mlx-lm commands
}
```

**Design Decision 1: Server vs CLI Mode**

**Option A: CLI Mode (Simpler)**
```
For each inference request:
  1. Run: python3 -m mlx_lm generate --model <model> --prompt <text> --max-tokens 100
  2. Parse JSON output
  3. Return to user
```
Pros: Simple, no background server
Cons: Slower (spawn process each time), resource intensive

**Option B: Server Mode (Better)**
```
On load_model():
  1. Spawn: python3 -m mlx_lm.server --model <model> --port 8001
  2. Wait for server ready
  3. Cache server process

On generate():
  1. POST http://localhost:8001/v1/completions
  2. Stream response back

On unload_model():
  1. Terminate server process
  2. Clean up
```
Pros: Faster (persistent server), lower latency
Cons: More complex (process management)

**Recommendation:** Start with Option B (Server Mode)
- Matches LM Studio architecture
- Better performance for multiple requests
- Enables streaming integration later

#### Evening: Design HTTP Client (1 hour)

**What We Need:**
```rust
struct MlxServerClient {
    base_url: String,      // http://localhost:8001
    timeout: Duration,     // 30s
}

impl MlxServerClient {
    fn new(port: u16) -> Self { ... }
    
    async fn generate(&self, 
        model: &str,
        prompt: &str,
        max_tokens: usize
    ) -> MinervaResult<String> { ... }
    
    async fn is_ready(&self) -> bool { ... }
}
```

**HTTP Calls Required:**
```
GET http://localhost:8001/health          # Check if running
POST http://localhost:8001/v1/completions # Generate text
{
  "model": "mistral-7b",
  "prompt": "Hello world",
  "max_tokens": 100,
  "temperature": 0.7,
  "top_p": 0.9
}
```

**Expected Response:**
```json
{
  "id": "cmpl-xxx",
  "object": "text_completion",
  "created": 1234567890,
  "model": "mistral-7b",
  "choices": [{
    "text": " this is the generated response",
    "index": 0,
    "logprobs": null,
    "finish_reason": "length"
  }],
  "usage": {
    "prompt_tokens": 2,
    "completion_tokens": 8,
    "total_tokens": 10
  }
}
```

**Crates Needed:**
- `reqwest` (HTTP client) - ✅ already in Cargo.toml
- `tokio` (async runtime) - ✅ already in Cargo.toml
- `serde_json` (JSON parsing) - ✅ already in Cargo.toml

**No new dependencies needed!**

---

### Day 2: Implementation

#### Morning: Create HTTP Client Module (2 hours)

**File:** `src-tauri/src/inference/mlx_http_client.rs` (new file, ~150 lines)

```rust
use reqwest::Client;
use std::time::Duration;
use crate::error::{MinervaError, MinervaResult};

pub struct MlxServerClient {
    http_client: Client,
    base_url: String,
    timeout: Duration,
}

impl MlxServerClient {
    pub fn new(port: u16) -> Self {
        Self {
            http_client: Client::new(),
            base_url: format!("http://localhost:{}", port),
            timeout: Duration::from_secs(30),
        }
    }

    /// Check if mlx-lm server is ready
    pub async fn is_ready(&self) -> bool {
        self.http_client
            .get(&format!("{}/health", self.base_url))
            .timeout(self.timeout)
            .send()
            .await
            .ok()
            .map(|r| r.status().is_success())
            .unwrap_or(false)
    }

    /// Generate text via mlx-lm API
    pub async fn generate(
        &self,
        model: &str,
        prompt: &str,
        max_tokens: usize,
        temperature: f32,
        top_p: f32,
    ) -> MinervaResult<String> {
        let request_body = serde_json::json!({
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        });

        let response = self.http_client
            .post(&format!("{}/v1/completions", self.base_url))
            .json(&request_body)
            .timeout(self.timeout)
            .send()
            .await
            .map_err(|e| MinervaError::InferenceError(
                format!("MLX server request failed: {}", e)
            ))?;

        let json = response.json::<serde_json::Value>().await
            .map_err(|e| MinervaError::InferenceError(
                format!("Failed to parse MLX response: {}", e)
            ))?;

        // Extract text from response
        json["choices"][0]["text"]
            .as_str()
            .ok_or_else(|| MinervaError::InferenceError(
                "Invalid MLX response format".to_string()
            ))
            .map(|s| s.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mlx_http_client_creation() {
        let client = MlxServerClient::new(8001);
        assert_eq!(client.base_url, "http://localhost:8001");
    }

    #[tokio::test]
    async fn test_mlx_http_client_is_ready_false() {
        // Server not running - should be false
        let client = MlxServerClient::new(8001);
        assert!(!client.is_ready().await);
    }
}
```

#### Afternoon: Update MlxBackend to Use HTTP Client (2 hours)

**File:** `src-tauri/src/inference/mlx_backend.rs` (modify existing)

**Changes:**
```rust
use super::mlx_http_client::MlxServerClient;
use std::process::{Child, Command};

pub struct MlxBackend {
    loaded_model: Arc<Mutex<Option<String>>>,
    mlx_status: Arc<Mutex<MlxStatus>>,
    server_process: Arc<Mutex<Option<Child>>>,  // NEW: Track server process
    http_client: Arc<Mutex<Option<MlxServerClient>>>,  // NEW: HTTP client
    n_threads: usize,
    n_ctx: usize,
}

impl MlxBackend {
    pub fn new() -> Self {
        Self {
            loaded_model: Arc::new(Mutex::new(None)),
            mlx_status: Arc::new(Mutex::new(MlxStatus::Unchecked)),
            server_process: Arc::new(Mutex::new(None)),  // NEW
            http_client: Arc::new(Mutex::new(None)),     // NEW
            n_threads: num_cpus::get(),
            n_ctx: 0,
        }
    }

    /// Start mlx-lm server subprocess
    fn start_server(&self, model_ref: &str) -> MinervaResult<Child> {
        let output = Command::new("python3")
            .arg("-m").arg("mlx_lm.server")
            .arg("--model").arg(model_ref)
            .arg("--port").arg("8001")
            .spawn()
            .map_err(|e| MinervaError::InferenceError(
                format!("Failed to start mlx-lm server: {}", e)
            ))?;

        // Wait for server to be ready
        std::thread::sleep(std::time::Duration::from_secs(2));
        
        Ok(output)
    }

    /// Stop mlx-lm server
    fn stop_server(&self) -> MinervaResult<()> {
        if let Some(mut child) = self.server_process.lock().unwrap().take() {
            let _ = child.kill();
            let _ = child.wait();
        }
        Ok(())
    }
}

impl InferenceBackend for MlxBackend {
    fn load_model(&mut self, path: &Path, n_ctx: usize) -> MinervaResult<()> {
        // Check mlx availability
        // (existing code unchanged)
        
        let model_name = Self::extract_model_name(path);
        let format = Self::detect_model_format(path);

        // START SERVER
        let server_process = self.start_server(&model_name)?;
        *self.server_process.lock().unwrap() = Some(server_process);
        
        // Create HTTP client
        let client = MlxServerClient::new(8001);
        *self.http_client.lock().unwrap() = Some(client);
        
        *self.loaded_model.lock().unwrap() = Some(model_name.clone());
        self.n_ctx = n_ctx;

        tracing::info!("MLX server started for model: {}", model_name);
        Ok(())
    }

    fn unload_model(&mut self) {
        self.stop_server().ok();
        *self.loaded_model.lock().unwrap() = None;
        *self.http_client.lock().unwrap() = None;
        self.n_ctx = 0;
        tracing::info!("MLX server stopped");
    }

    fn generate(&self, prompt: &str, params: GenerationParams) -> MinervaResult<String> {
        let model = self.loaded_model.lock().unwrap();
        let model_ref = model
            .as_ref()
            .ok_or_else(|| MinervaError::InferenceError("No model loaded".to_string()))?;

        let client = self.http_client.lock().unwrap();
        let http_client = client
            .as_ref()
            .ok_or_else(|| MinervaError::InferenceError("HTTP client not initialized".to_string()))?;

        // Make HTTP request to mlx-lm server
        let rt = tokio::runtime::Runtime::new()?;
        rt.block_on(async {
            http_client.generate(
                model_ref,
                prompt,
                params.max_tokens,
                params.temperature,
                params.top_p,
            ).await
        })
    }

    // Other methods unchanged...
}
```

---

### Day 3: Testing & Documentation

#### Morning: Integration Tests (2 hours)

**File:** `src-tauri/tests/integration/mlx_backend_integration.rs` (new file, ~250 lines)

```rust
use minerva_lib::inference::mlx_backend::MlxBackend;
use minerva_lib::inference::llama_adapter::{InferenceBackend, GenerationParams};
use std::path::Path;

/// Tests that run WITH mlx-lm installed
/// These are marked #[ignore] and must be run with:
/// cargo test mlx_backend_integration -- --ignored --test-threads=1

#[test]
#[ignore]
fn test_mlx_backend_start_server() {
    // Test: Can we start mlx-lm server?
    let mut backend = MlxBackend::new();
    
    // This will start a real server if mlx-lm is installed
    let result = backend.load_model(Path::new("mistral-7b"), 2048);
    
    match result {
        Ok(_) => {
            assert!(backend.is_loaded());
            backend.unload_model();
            // Server should be stopped
        }
        Err(e) => {
            eprintln!("mlx-lm not available or failed: {}", e);
            // This is OK in CI without mlx-lm
        }
    }
}

#[test]
#[ignore]
fn test_mlx_backend_generate_text() {
    let mut backend = MlxBackend::new();
    
    if let Ok(_) = backend.load_model(Path::new("mistral-7b"), 2048) {
        let params = GenerationParams {
            max_tokens: 50,
            temperature: 0.7,
            top_p: 0.9,
        };
        
        let result = backend.generate("Hello, world!", params);
        
        match result {
            Ok(text) => {
                assert!(!text.is_empty());
                println!("Generated: {}", text);
            }
            Err(e) => eprintln!("Generation failed: {}", e),
        }
        
        backend.unload_model();
    }
}

#[test]
fn test_mlx_backend_http_client_creation() {
    use minerva_lib::inference::mlx_http_client::MlxServerClient;
    
    let client = MlxServerClient::new(8001);
    // Should not panic
    assert_eq!(client.base_url, "http://localhost:8001");
}
```

#### Afternoon: Update Documentation (1 hour)

**File:** `docs/PHASE_8_STEP_3b_ROADMAP.md`
- Mark "Day 1-3" sections as ✅ Complete
- Add "Day 4" section for Phase 3d planning
- Update timeline

**File:** `docs/PHASE_8_PLAN.md`
- Update progress section
- Mark Step 3b as complete
- Update timeline for remaining steps

#### Final: Run All Tests & Lint (30 minutes)

```bash
# Run all tests
pnpm test
# Expected: 830+ passing (806 existing + 24 new)

# Run lint
pnpm lint
# Expected: 0 violations

# Commit
git add -A
git commit -m "feat(phase8-step3b): Implement MLX subprocess integration

- Create MlxServerClient for HTTP communication with mlx-lm
- Implement server startup/shutdown in MlxBackend.load_model()
- Add real model inference via HTTP API
- Support process management and cleanup
- Add 24+ integration tests (with #[ignore] for optional running)
- All 830+ tests passing, 0 lint violations"
```

---

## Testing Strategy

### Unit Tests (No mlx-lm Required)
- HTTP client creation
- Base URL formatting
- Request body construction
- Error handling

### Integration Tests (Require mlx-lm)
```bash
# Run optional integration tests (requires mlx-lm)
cargo test mlx_backend_integration -- --ignored --test-threads=1

# Run without (skip these tests)
cargo test  # Will skip #[ignore] tests by default
```

All integration tests marked with:
```rust
#[test]
#[ignore]  // Only runs with: cargo test -- --ignored
fn test_name() { ... }
```

---

## Error Cases to Handle

### 1. MLX Server Won't Start
```
Error: "Failed to start mlx-lm server: ..."
→ Return MinervaError::InferenceError
→ User sees: "mlx-lm not installed or unavailable"
→ Graceful fallback to llama.cpp backend
```

### 2. Server Timeout
```
Error: "MLX server request timed out (30s)"
→ Return MinervaError::InferenceError
→ Kill server process
→ User can retry or switch backends
```

### 3. Invalid Model Name
```
Error: "Model 'invalid-model' not found on HuggingFace"
→ Return MinervaError::ModelNotFound
→ Stop server
→ User sees clear error message
```

### 4. HTTP Connection Refused
```
Error: "Failed to connect to mlx-lm server at localhost:8001"
→ Return MinervaError::InferenceError
→ Check if server process crashed
→ Attempt restart or fallback
```

---

## Performance Targets

| Metric | Target | Notes |
|--------|--------|-------|
| Server startup | <3s | Python import + model load |
| First inference | <5s | HTTP request + mlx-lm generation |
| Subsequent requests | <2s | Cached model + mlx-lm generation |
| Memory overhead | <500MB | Python runtime + model weights |
| Error recovery | <100ms | Timeout and fallback |

---

## Checkpoints

### Day 1 Evening
- [ ] mlx-lm installed and tested
- [ ] Subprocess architecture designed
- [ ] HTTP client design finalized
- [ ] No code written yet, just planning

### Day 2 Evening
- [ ] MlxServerClient module complete (150 lines, ~8 tests)
- [ ] MlxBackend integration started (subprocess calls)
- [ ] All existing tests still passing
- [ ] Lint checks passing

### Day 3 Evening
- [ ] Full subprocess integration working
- [ ] 24+ integration tests written
- [ ] All 830+ tests passing
- [ ] 0 lint violations
- [ ] Ready for Phase 8-Step 3d

---

## Success Criteria

✅ Subprocess-based mlx-lm server integration works
✅ Real HuggingFace model inference possible
✅ 830+ tests passing (806 existing + 24 new)
✅ 0 lint violations
✅ Error handling robust (all edge cases covered)
✅ Performance meets targets
✅ Documentation complete
✅ Ready for next developer to continue Phase 8-Step 3d

---

## Next Phase (After This)

### Phase 8-Step 3d: Integration Tests & Refinement
- Comprehensive testing with real mlx-lm
- Performance benchmarking vs llama.cpp
- Stress testing (concurrent requests, large models)
- Documentation and examples

### Phase 8-Step 4: Backend Selection
- Auto-routing based on model format
- User-configurable backend preference
- Fallback chains (prefer MLX, fallback to llama.cpp)
- API parameter for backend selection

---

## References

### MLX-LM Documentation
- GitHub: https://github.com/ml-explore/mlx-examples/tree/main/llm
- Installation: `pip install mlx-lm`
- Server docs: `python3 -m mlx_lm.server --help`

### LM Studio (Reference Implementation)
- Uses same subprocess architecture
- https://github.com/lmstudio-ai/lmstudio
- Proven approach for managing multiple backends

### Reqwest (HTTP Client)
- Already in dependencies
- Examples: https://docs.rs/reqwest/latest/reqwest/

---

## Quick Start Command

When ready to start Phase 8-Step 3b:

```bash
# Install mlx-lm
pip install mlx-lm

# Verify installation
python3 -m mlx_lm --help

# Create new feature branch
git checkout -b phase-8/mlx-subprocess

# Start implementing MlxServerClient...
```

---

**Status:** Ready to Start  
**Complexity:** Medium (process management + async HTTP)  
**Time Estimate:** 2-3 focused days  
**Risk Level:** Low (subprocess is battle-tested approach)  

All groundwork complete. Ready for next developer! ✅
