# Test Structure Documentation

## Overview

The Minerva project follows Rust testing conventions with a clear separation between unit tests and integration tests.

**Test Summary:**
- **Unit Tests:** 80 tests in `src/` modules
- **Integration Tests:** 21 tests in `tests/`
- **Total:** 101 passing tests
- **Quality:** 0 warnings, 100% format compliance

---

## Unit Tests

Unit tests are placed in the same modules they test, using `#[cfg(test)]` blocks.

### Location
```
src/
├── inference/
│   ├── llama_engine.rs        # 8 tests
│   ├── llama_adapter.rs       # 8 tests
│   ├── token_stream.rs        # 5 tests
│   ├── gpu_context.rs         # 6 tests
│   ├── context_manager.rs     # 9 tests
│   ├── streaming.rs           # 7 tests
│   ├── metrics.rs             # 4 tests
│   └── parameters.rs          # (tests in integration only)
├── models/
│   ├── loader.rs              # 4 tests
│   └── gguf_parser.rs         # 3 tests
├── config.rs                  # 3 tests
├── commands.rs                # 1 test
└── server.rs                  # 5 tests
```

### Example Unit Test
```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_module_function() {
        let result = function_under_test();
        assert_eq!(result, expected);
    }
}
```

### Running Unit Tests
```bash
pnpm test:backend:unit
# or
cd src-tauri && cargo test --lib && cd ..
```

---

## Integration Tests

Integration tests are in the `tests/` directory and test end-to-end workflows involving multiple components.

### Location
```
tests/
└── integration_tests.rs    # 21 tests organized by domain
```

### Test Organization by Domain

#### 1. Model Discovery Tests (3 tests)
- `test_model_discovery_basic` - Single model discovery
- `test_model_discovery_multiple` - Multiple model discovery
- `test_model_discovery_filters_non_gguf` - File filtering
- `test_model_registry_discovery` - Registry integration

**What they test:**
- GGUF file detection
- Model metadata parsing
- Registry population
- File type filtering

#### 2. Inference Engine Tests (3 tests)
- `test_inference_engine_lifecycle` - Load/unload cycle
- `test_inference_generate_response` - Response generation
- `test_inference_context_info` - Context metadata

**What they test:**
- Engine lifecycle management
- Response generation quality
- Context tracking

#### 3. Token Streaming Tests (3 tests)
- `test_token_stream_collection` - Token gathering
- `test_token_stream_iteration` - Token iteration
- `test_token_stream_reset` - Stream reset functionality

**What they test:**
- Token collection mechanics
- Stream state management
- Position tracking

#### 4. Backend Abstraction Tests (3 tests)
- `test_mock_backend_lifecycle` - Backend loading
- `test_mock_backend_generation` - Response generation
- `test_mock_backend_tokenization` - Token operations

**What they test:**
- Backend trait interface
- MockBackend implementation
- Tokenization interface

#### 5. Parameter Validation Tests (4 tests)
- `test_generation_config_validation_valid` - Valid params
- `test_generation_config_validation_temperature` - Invalid temperature
- `test_generation_config_validation_top_p` - Invalid top_p
- `test_generation_config_validation_max_tokens` - Invalid max_tokens

**What they test:**
- Configuration validation
- Parameter boundary conditions
- Error handling

#### 6. GPU Context Tests (2 tests)
- `test_gpu_context_creation` - Context setup
- `test_gpu_context_allocation` - Memory management

**What they test:**
- GPU device detection
- Memory allocation/deallocation

#### 7. End-to-End Pipeline Tests (2 tests)
- `test_full_inference_pipeline` - Complete workflow
- `test_backend_with_streaming` - Backend + streaming

**What they test:**
- Full inference flow
- Component integration
- Streaming integration

### Example Integration Test
```rust
#[test]
fn test_full_inference_pipeline() {
    use minerva_lib::inference::llama_engine::LlamaEngine;
    use minerva_lib::inference::token_stream::TokenStream;

    // Setup
    let (_temp, models_dir) = setup_test_models_dir();
    let model_path = create_dummy_gguf(&models_dir, "test");

    // Test
    let mut engine = LlamaEngine::new(model_path);
    assert!(engine.load(2048).is_ok());
    
    let stream = TokenStream::new();
    let response = engine.generate("test", 100).unwrap();
    
    for word in response.split_whitespace() {
        stream.push_token(format!("{} ", word));
    }

    // Verify
    assert!(!collected.is_empty());
    engine.unload();
}
```

### Running Integration Tests
```bash
pnpm test:backend:integration
# or
cd src-tauri && cargo test --test '*' && cd ..
```

---

## Test Organization Principles

### 1. Unit Tests in Modules
- Keep tests close to code
- Easy to maintain
- Fast execution
- Test internal behavior

### 2. Integration Tests Separate
- Test cross-module workflows
- Verify APIs
- Slower but comprehensive
- Test external behavior

### 3. Clear Naming
- `test_component_scenario` - Unit test format
- Tests clearly indicate what they test
- Organized by domain in integration tests

### 4. Test Utilities
Helper functions in integration_tests.rs:
```rust
fn setup_test_models_dir() -> (TempDir, PathBuf)
fn create_dummy_gguf(dir: &Path, name: &str) -> PathBuf
```

---

## Running Tests

### All Tests
```bash
pnpm test
pnpm test:backend
# Runs: cargo test --lib && cargo test --test '*'
```

### Unit Tests Only
```bash
pnpm test:backend:unit
# Runs: cargo test --lib
```

### Integration Tests Only
```bash
pnpm test:backend:integration
# Runs: cargo test --test '*'
```

### Specific Test
```bash
cd src-tauri
cargo test test_inference_engine_lifecycle
```

### With Output
```bash
cd src-tauri
cargo test -- --nocapture --test-threads=1
```

---

## Test Coverage by Component

### Inference Module
- **llama_engine.rs:** 8 unit tests
- **llama_adapter.rs:** 8 unit tests
- **token_stream.rs:** 5 unit tests
- **gpu_context.rs:** 6 unit tests
- **context_manager.rs:** 9 unit tests
- **streaming.rs:** 7 unit tests
- **metrics.rs:** 4 unit tests
- **Integration:** 11 tests (engine, streaming, backend, pipeline)

### Models Module
- **loader.rs:** 4 unit tests
- **gguf_parser.rs:** 3 unit tests
- **Integration:** 4 tests (discovery, registry)

### Server & Config
- **server.rs:** 5 unit tests
- **config.rs:** 3 unit tests
- **commands.rs:** 1 unit test

### Total: 80 unit + 21 integration = **101 tests**

---

## Writing New Tests

### Adding a Unit Test
```rust
// In src/module/file.rs
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_feature() {
        let result = function_to_test();
        assert_eq!(result, expected_value);
    }
}
```

### Adding an Integration Test
```rust
// In tests/integration_tests.rs
#[test]
fn test_feature_across_modules() {
    // Setup
    let temp_dir = TempDir::new().unwrap();
    
    // Test
    let result = cross_module_function();
    
    // Verify
    assert!(result.is_ok());
}
```

---

## CI/CD Integration

All tests are run in CI:
```yaml
test:
  - pnpm check:all      # Lint + format check
  - pnpm test           # Run all 101 tests
```

**Quality Gate:**
- 0 clippy warnings
- 100% code format compliance
- All 101 tests passing
- No errors during build

---

## Performance

Current test execution time:
- Unit tests: ~70ms
- Integration tests: ~60ms
- Total: ~130ms

Characteristics:
- Fast execution (suitable for TDD)
- Uses temporary directories (safe)
- No external dependencies
- Can run in parallel

---

## Best Practices

### ✅ DO
- Keep unit tests in module files
- Use descriptive test names
- Test one thing per test
- Use meaningful assertions
- Organize integration tests by domain
- Clean up temporary files (handled by TempDir)

### ❌ DON'T
- Put integration tests in lib.rs
- Mix unit and integration tests
- Test implementation details
- Use magic numbers
- Ignore test failures
- Skip assertion messages

---

## Troubleshooting

### Tests Won't Run
```bash
# Ensure Cargo.toml is correct
cd src-tauri
cargo test --lib    # Run unit tests to verify
```

### Clippy Warnings in Tests
```bash
# Check for new warnings
cargo clippy --all-targets --all-features -- -D warnings
```

### Integration Tests Not Found
```bash
# Verify tests/ directory exists
ls -la tests/
# Should contain: integration_tests.rs
```

---

## Resources

- [Rust Book - Testing](https://doc.rust-lang.org/book/ch11-00-testing.html)
- [Rust API Guidelines](https://rust-lang.github.io/api-guidelines/)
- [Our AGENTS.md Standards](/AGENTS.md)

---

## Summary

Minerva uses a well-organized test structure following Rust conventions:
- **80 unit tests** in module files (fast, focused)
- **21 integration tests** in tests/ directory (comprehensive, cross-module)
- **101 total tests** organized by domain
- **~130ms** total execution time
- **0 warnings**, 100% format compliance

This structure supports rapid development, comprehensive coverage, and easy maintenance.
