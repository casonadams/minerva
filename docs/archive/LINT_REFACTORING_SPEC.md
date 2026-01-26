# Lint Refactoring Specification - Phase 6 Too-Many-Arguments Violations

**Total Violations**: 23 across 5 files  
**Goal**: Fix all "too_many_arguments" violations by consolidating parameters into parameter objects  
**Current Status**: Phase 6 complete with 464 passing tests  

## Overview

The clippy linter with `too-many-arguments-threshold = 3` requires all Rust functions to have ≤ 3 parameters (including `&self`/`&mut self`). The Phase 6 code was written before this strict requirement was enforced, resulting in 23 violations.

**Strategy**: Consolidate extra function parameters into parameter structs that are already partially defined in each module.

---

## File 1: gpu_compute_engine.rs (8 violations)

### Current Violations
- Line 120: `compute_matmul(&self, a: &[f32], b: &[f32], params: MatmulParams)` - 4 params
- Line 145: `compute_attention(&self, q: &[f32], k: &[f32], v: &[f32], params: AttentionParams)` - 5 params
- Line 207: `compute_rmsnorm(&self, x: &[f32], weight: &[f32], params: RmsNormParams)` - 4 params
- Line 241: `cpu_matmul(&self, a: &[f32], b: &[f32], params: MatmulParams)` - 4 params
- Line 259: `cpu_matmul_impl(&self, a: &[f32], b: &[f32], params: &MatmulParams)` - 4 params
- Line 276: `cpu_attention(&self, q: &[f32], k: &[f32], v: &[f32], params: AttentionParams)` - 5 params
- Line 296: `cpu_attention_impl(&self, q: &[f32], k: &[f32], v: &[f32], params: &AttentionParams, seq_len: usize)` - 6 params
- Line 313: Loop index usage warning (secondary issue)

### Solution

**Step 1**: Enhance parameter structs to include data:

```rust
/// Parameters for matrix multiplication
#[derive(Debug, Clone)]
pub struct MatmulParams {
    pub a: Vec<f32>,           // Add matrix A data
    pub b: Vec<f32>,           // Add matrix B data
    pub a_rows: usize,
    pub a_cols: usize,
    pub b_cols: usize,
}

impl MatmulParams {
    pub fn new(a: Vec<f32>, b: Vec<f32>, a_rows: usize, a_cols: usize, b_cols: usize) -> Self {
        Self { a, b, a_rows, a_cols, b_cols }
    }
}

/// Parameters for attention computation
#[derive(Debug, Clone)]
pub struct AttentionParams {
    pub q: Vec<f32>,           // Add query data
    pub k: Vec<f32>,           // Add key data
    pub v: Vec<f32>,           // Add value data
    pub heads: usize,
    pub head_dim: usize,
}

impl AttentionParams {
    pub fn new(q: Vec<f32>, k: Vec<f32>, v: Vec<f32>, heads: usize, head_dim: usize) -> Self {
        Self { q, k, v, heads, head_dim }
    }
}

/// Parameters for RMSNorm computation
#[derive(Debug, Clone)]
pub struct RmsNormParams {
    pub x: Vec<f32>,           // Add input data
    pub weight: Vec<f32>,      // Add weight data
    pub eps: f32,
}

impl RmsNormParams {
    pub fn new(x: Vec<f32>, weight: Vec<f32>, eps: f32) -> Self {
        Self { x, weight, eps }
    }
}

/// Parameters for attention implementation with sequence length
#[derive(Debug, Clone)]
struct AttentionImplParams {
    pub q: Vec<f32>,
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    pub heads: usize,
    pub head_dim: usize,
    pub seq_len: usize,
}
```

**Step 2**: Update function signatures:

```rust
// BEFORE
pub fn compute_matmul(&self, a: &[f32], b: &[f32], params: MatmulParams) -> MinervaResult<ComputeResult>

// AFTER
pub fn compute_matmul(&self, params: MatmulParams) -> MinervaResult<ComputeResult>
```

**Step 3**: Update function implementations to use `params.a`, `params.b`, etc.

**Step 4**: Update all call sites in:
- impl block methods
- Test code (14 tests in this file)

---

## File 2: gpu_llama_integration.rs (4 violations)

### Current Violations
- Line 81: Multi-parameter function violation
- Other violations (exact lines in compiler output)

### Solution

Similar approach: consolidate q, k, v, input, weight, etc. into AttentionParams and other param structs. Update all call sites.

### Call Sites to Update
- gpu_llama_integration.rs:222 - `compute_matmul` call
- gpu_llama_integration.rs:242 - `compute_matmul` call
- gpu_llama_integration.rs (attention calls) - use AttentionParams

---

## File 3: kv_cache_optimizer.rs (3 violations)

### Current Violations
- Line 142: Multi-parameter function
- Line 165: Multi-parameter function
- Line 179: Multi-parameter function

### Solution

Create parameter structs for each operation and consolidate.

---

## File 4: llama_inference.rs (7 violations)

### Current Violations
- Line 78: `KVCache::store(&mut self, layer: usize, pos: usize, kv: (&[f32], &[f32]))`
- Lines 223, 252, 274, 340, 412, 542: Various multi-parameter functions

### Solution

**Create parameter structs**:

```rust
#[derive(Debug, Clone)]
pub struct KVStoreParams {
    pub layer: usize,
    pub pos: usize,
    pub k: Vec<f32>,
    pub v: Vec<f32>,
}

impl KVStoreParams {
    pub fn new(layer: usize, pos: usize, k: Vec<f32>, v: Vec<f32>) -> Self {
        Self { layer, pos, k, v }
    }
}
```

**Update function signature**:

```rust
// BEFORE
pub fn store(&mut self, layer: usize, pos: usize, kv: (&[f32], &[f32])) -> MinervaResult<()>

// AFTER
pub fn store(&mut self, params: KVStoreParams) -> MinervaResult<()>
```

---

## File 5: metal_gpu.rs (1 violation)

### Current Violation
- Line 333: Single multi-parameter function

### Solution

Create a parameter struct for the specific operation and consolidate.

---

## Implementation Checklist

### For Each File:

- [ ] Read all violations and understand function signatures
- [ ] Create parameter structs (or enhance existing ones) to consolidate data
- [ ] Update function signatures to use new parameter structs
- [ ] Update function implementations to access data via params
- [ ] Find and update ALL call sites:
  - [ ] In impl blocks (other methods calling this method)
  - [ ] In tests
  - [ ] In external modules (gpu_llama_integration, etc.)
- [ ] Run `cargo test --lib` to verify no regressions
- [ ] Verify clippy no longer flags violations for this file

### Verification Steps

After each file is complete:

```bash
# Run tests
cd src-tauri && cargo test --lib 2>&1 | grep -E "test result:|passed"

# Check clippy violations
pnpm lint:backend 2>&1 | grep "gpu_compute_engine.rs" # (for that file)

# Should see 0 violations after fix
```

### Final Verification

```bash
# Run all tests
cd src-tauri && cargo test --lib

# Check all linting
pnpm lint

# Should pass with 0 errors
```

---

## Testing Strategy

1. **Unit Tests**: Each module has 10-20 tests - ensure all still pass
2. **Integration**: Verify gpu_llama_integration still works with new signatures
3. **Full Suite**: `cargo test --lib` should pass with 464+ tests

---

## Estimated Effort

- **gpu_compute_engine.rs**: 30-45 minutes (8 violations, 14 tests)
- **gpu_llama_integration.rs**: 20-30 minutes (4 violations, 11 tests)
- **llama_inference.rs**: 30-45 minutes (7 violations, 27 tests)
- **kv_cache_optimizer.rs**: 20-30 minutes (3 violations, 19 tests)
- **metal_gpu.rs**: 10-15 minutes (1 violation, <10 tests)
- **Verification & Testing**: 15-20 minutes

**Total Estimated Time**: 2-3 hours

---

## Key Principles

1. **No `#[allow(...)]` overrides** - Properly refactor instead
2. **Owned vs References** - Param structs should own data (Vec<f32>) not borrow (&[f32])
3. **Backward Compatibility** - Public API should remain usable
4. **Test Coverage** - All tests must pass after refactoring
5. **Clean Diffs** - Each file should be a focused, clean change

---

## Call Site Mapping

### gpu_compute_engine.rs Internal Calls
- Line 129: `self.cpu_matmul(a, b, params)` → `self.cpu_matmul(MatmulParams::new(...))`
- Line 134: `self.cpu_matmul_impl(a, b, &params)` → `self.cpu_matmul_impl(&params)`
- Line 155: `self.cpu_attention(q, k, v, params)` → `self.cpu_attention(AttentionParams::new(...))`
- Line 167: `self.cpu_attention_impl(q, k, v, &params, seq_len)` → `self.cpu_attention_impl(&AttentionImplParams{...})`
- Line 222: `self.cpu_rmsnorm_impl(x, weight, params.eps)` → `self.cpu_rmsnorm_impl(&RmsNormParams::new(...))`
- Line 230: `self.cpu_rmsnorm_impl(x, weight, params.eps)` → `self.cpu_rmsnorm_impl(&RmsNormParams::new(...))`

### gpu_compute_engine.rs Test Calls
- Line 418: `engine.compute_matmul(&a, &b, params)` → `engine.compute_matmul(MatmulParams::new(a, b, ...))`
- Line 456: `engine.compute_rmsnorm(&x, &weight, params)` → `engine.compute_rmsnorm(RmsNormParams::new(x, weight, ...))`
- Line 474: `engine.compute_rmsnorm(&x, &weight, params)` → `engine.compute_rmsnorm(RmsNormParams::new(x, weight, ...))`
- Line 490: `engine.compute_attention(&q, &k, &v, params)` → `engine.compute_attention(AttentionParams::new(q, k, v, ...))`
- Line 505: `engine.compute_attention(&q, &k, &v, params)` → `engine.compute_attention(AttentionParams::new(q, k, v, ...))`
- Line 531: `engine.compute_matmul(&a, &b, params)` → `engine.compute_matmul(MatmulParams::new(a, b, ...))`
- Line 548: `engine.compute_attention(&q, &k, &v, params)` → `engine.compute_attention(AttentionParams::new(q, k, v, ...))`

### gpu_llama_integration.rs Calls
- Multiple locations calling `compute_matmul`, `compute_attention` with old signatures
- All need updating to use new Params objects

---

## Notes for Next Session

- Start with gpu_compute_engine.rs as it's most self-contained
- Test immediately after each file
- Commit each file individually with message like: `refactor(lint): consolidate parameters in gpu_compute_engine.rs`
- If hitting issues, can roll back individual files
- Final commit: `refactor(lint): resolve all 23 too-many-arguments violations`

---

## References

- Clippy Lint: https://rust-lang.github.io/rust-clippy/master/index.html#too_many_arguments
- Project Standards: docs/ENGINEERING_STANDARDS.md
- Current Status: PHASE6_STATUS.md
