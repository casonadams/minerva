# Clippy Violations Report

## Overview
Clippy configuration enforces `too-many-arguments-threshold = 3` as per engineering standards. Functions should have at most 3 parameters; complex parameter lists should be consolidated into objects/structs.

## Current Violations (6 total)

### 1. parameters.rs:90 - `make_request()` (4 args)
**File**: `src/inference/parameters.rs:90`
**Function**: `make_request()`
**Arguments**: 4 (violates threshold of 3)

```rust
fn make_request(
    temp: Option<f32>,
    top_p: Option<f32>,
    max_tokens: Option<usize>,
    freq_penalty: Option<f32>,
) -> ChatCompletionRequest { ... }
```

**Fix**: Create a struct to hold generation parameters
```rust
struct GenerationParams {
    temperature: Option<f32>,
    top_p: Option<f32>,
    max_tokens: Option<usize>,
    frequency_penalty: Option<f32>,
}

fn make_request(params: GenerationParams) -> ChatCompletionRequest { ... }
```

---

### 2. benchmarks.rs:28 - (needs inspection)
**File**: `src/inference/benchmarks.rs:28`
**Arguments**: 5 (violates threshold)

---

### 3. llama_adapter.rs:39 - (needs inspection)
**File**: `src/inference/llama_adapter.rs:39`
**Arguments**: 5 (violates threshold)

---

### 4. metrics.rs:18 - (needs inspection)
**File**: `src/inference/metrics.rs:18`
**Arguments**: 5 (violates threshold)

---

### 5. tokenizer.rs:128 - (needs inspection)
**File**: `src/inference/tokenizer.rs:128`
**Arguments**: 4 (violates threshold)

---

### 6. gguf_tensor.rs:137 - `new()` (4 args)
**File**: `src/models/gguf_tensor.rs:137`
**Function**: `GGUFTensor::new()`
**Arguments**: 4 (violates threshold)

```rust
pub fn new(
    name: String,
    data_type: GGUFDataType,
    shape: Vec<u64>,
    data: Vec<u8>,
) -> Self { ... }
```

**Fix**: Use a builder pattern or struct initialization
```rust
pub struct GGUFTensorData {
    pub name: String,
    pub data_type: GGUFDataType,
    pub shape: Vec<u64>,
    pub data: Vec<u8>,
}

impl From<GGUFTensorData> for GGUFTensor { ... }
```

---

## Engineering Standard Rationale

From `/docs/ENGINEERING_STANDARDS.md`:
- **2 parameters**: use object for more
- Dependency injection (never direct imports in business logic)
- Single responsibility (one reason to change)

Functions with too many arguments indicate:
1. **Low cohesion**: Function is trying to do too much
2. **Hard to test**: Too many mock objects needed
3. **Hard to understand**: Complex interface
4. **Hard to maintain**: Changes affect many call sites

## Next Steps

1. Create parameter/data objects for functions with 4+ args
2. Refactor affected functions to use consolidated objects
3. Run `cargo clippy --all-targets -- -D clippy::too-many-arguments` to verify
4. All tests should pass after refactoring

## Files to Modify

- [ ] `src/inference/parameters.rs` - make_request()
- [ ] `src/inference/benchmarks.rs` - line 28
- [ ] `src/inference/llama_adapter.rs` - line 39
- [ ] `src/inference/metrics.rs` - line 18
- [ ] `src/inference/tokenizer.rs` - line 128
- [ ] `src/models/gguf_tensor.rs` - GGUFTensor::new()

## Success Criteria

- ✅ All 6 violations fixed
- ✅ No new violations introduced
- ✅ All 330+ tests passing
- ✅ `cargo clippy --all-targets -- -D clippy::too-many-arguments` passes
- ✅ Code quality maintained or improved

