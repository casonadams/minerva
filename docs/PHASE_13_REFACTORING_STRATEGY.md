# Phase 13: Inference Module Refactoring Strategy

**Phase**: 13 (Inference Code Compliance)  
**Status**: Planning  
**Target**: Reduce 43 inference files to ≤150 lines each  
**Total Lines**: 21,625 → target ~15,000-17,000 (organized into 100+ focused modules)  
**Effort**: Large (likely 8-12 sessions)  
**Priority**: High (blocking 100% compliance)

## Overview

Phase 13 involves refactoring the `inference/` package to meet Phase 11+ standards (≤150 lines per file, M≤3 complexity, ≤3 parameters). This is the largest refactoring effort yet, as the inference package contains the ML model implementations and GPU integration code.

## Current State

### Files Exceeding 150 Lines (43 total)

**Top 10 Largest Files**:
1. pure_rust_backend.rs - 1,037 lines
2. llama_inference.rs - 1,028 lines
3. metal_gpu.rs - 887 lines
4. transformer_components.rs - 695 lines
5. transformer_layers.rs - 637 lines
6. llama_adapter.rs - 614 lines
7. sampling.rs - 601 lines
8. model_loader.rs - 600 lines
9. tokenizer.rs - 597 lines
10. gpu_compute_engine.rs - 580 lines

**Total Scope**:
- 43 files over 150 lines
- 21,625 total lines in inference/
- ~10 major backend implementations (Pure Rust, Llama, MLX, Metal)
- ~8 GPU-related modules
- ~12 cache/optimization modules
- ~10 batch processing modules

## Refactoring Strategy

### Phase 13 will be split into multiple sessions:

**Session 1: Planning & Foundation (This session)**
- Analyze dependencies between modules
- Identify extraction points
- Create refactoring patterns for inference code
- Set up tracking for progress

**Sessions 2-4: Transformer Architecture** (3 sessions)
- Extract transformer components (head, attention, feedforward)
- Extract transformer layers (residual connections, normalization)
- Extract layer aggregation logic

**Sessions 5-6: Sampling & Generation** (2 sessions)
- Extract temperature scaling
- Extract top-k/top-p filtering
- Extract beam search logic
- Extract repetition penalty

**Sessions 7-8: GPU Backend Integration** (2 sessions)
- Extract Metal GPU backend modules
- Extract MLX backend modules
- Extract generic GPU abstraction

**Sessions 9-10: Model Loading & Management** (2 sessions)
- Extract GGUF loader helpers
- Extract model cache logic
- Extract model registry

**Sessions 11-12: Core Inference Engine** (2 sessions)
- Extract inference engine orchestration
- Extract pure Rust backend helpers
- Extract Llama inference specialization

**Sessions 13+: Integration & Optimization** (remaining)
- Extract batch processing logic
- Extract streaming response handling
- Extract context management
- Remaining cleanup and optimization

## Key Principles

### 1. Maintain Backward Compatibility
- Keep public APIs stable
- Use pub use re-exports for moved items
- All existing tests must pass

### 2. Extract by Concern
Don't split randomly. Split by:
- **Functional concern**: Separate responsibilities (sampling, caching, loading)
- **Architectural layer**: Separate abstractions (backends, tokenizers, components)
- **Usage pattern**: Separate how things are used (single-pass vs streaming)

### 3. Follow Proven Patterns
From previous phases:
```
Pattern 1: Extract Tests
  Large module with embedded tests
  → New file: {module}_tests.rs
  → Reduces original by 20-40%

Pattern 2: Extract Helpers
  Complex module with helper functions
  → New file: {module}_helpers.rs
  → Extract related utility functions

Pattern 3: Extract Logic
  Module with multiple concerns
  → New files: {module}_{concern1}.rs, {module}_{concern2}.rs
  → Each handles single responsibility

Pattern 4: Extract Abstraction
  Concrete implementation with shared interface
  → New file: {module}_{variant}.rs
  → Separate backend/variant implementations
```

### 4. Architecture-Aware Extraction
Inference modules have complex dependencies. Extraction must respect:
- **Tokenizer chain**: vocabulary → tokenizer → model_loader
- **Backend hierarchy**: backends → backend_selector → inference_engine
- **GPU chain**: gpu_context → gpu_compute_engine → gpu_batch_scheduler
- **Model chain**: model_registry → model_cache → model_loader

## Extraction Patterns for Inference

### Pattern A: Backend Variant Extraction

**Situation**: Single module with multiple backend implementations

**Example**: `unified_backend.rs` (526 lines)
```rust
impl InferenceBackend for PureRustImpl { /* 150 lines */ }
impl InferenceBackend for LlamaImpl { /* 150 lines */ }
impl InferenceBackend for MLXImpl { /* 150 lines */ }
```

**Extraction**:
```
unified_backend.rs (526 lines)
  ↓
unified_backend.rs (80 lines) - trait definitions, factory
unified_backend_pure_rust.rs (150 lines) - PureRust implementation
unified_backend_llama.rs (150 lines) - Llama implementation
unified_backend_mlx.rs (150 lines) - MLX implementation
```

### Pattern B: Component Tree Extraction

**Situation**: Nested components with deep hierarchies

**Example**: `transformer_components.rs` (695 lines)
```rust
pub struct Attention { /* 100 lines */ }
pub struct FeedForward { /* 100 lines */ }
pub struct LayerNorm { /* 50 lines */ }
pub struct MultiHeadAttention { /* 150 lines */ }
// ... more components
```

**Extraction**:
```
transformer_components.rs (695 lines)
  ↓
attention.rs (100 lines) - Attention impl
feedforward.rs (100 lines) - FeedForward impl
layer_norm.rs (50 lines) - LayerNorm impl
multi_head_attention.rs (150 lines) - MultiHeadAttention impl
transformer_components.rs (150 lines) - exports and factory
```

### Pattern C: Concern Extraction

**Situation**: Module handling multiple concerns

**Example**: `pure_rust_backend.rs` (1,037 lines)
```
- Token loading (150 lines)
- Model inference (300 lines)
- Cache management (200 lines)
- Sampling logic (200 lines)
- Quantization (187 lines)
```

**Extraction**:
```
pure_rust_backend.rs (1037 lines)
  ↓
pure_rust_backend.rs (150 lines) - core orchestration
pure_rust_token_loading.rs (150 lines) - token operations
pure_rust_inference.rs (300 lines) - inference loop
pure_rust_cache.rs (200 lines) - cache management
pure_rust_sampling.rs (200 lines) - sampling operations
pure_rust_quantization.rs (187 lines) - quantization logic
```

### Pattern D: GPU Abstraction Extraction

**Situation**: GPU-specific code mixed with generic logic

**Example**: `gpu_compute_engine.rs` (580 lines)
```
- Generic GPU abstraction (100 lines)
- Metal-specific code (250 lines)
- MLX-specific code (230 lines)
```

**Extraction**:
```
gpu_compute_engine.rs (580 lines)
  ↓
gpu_compute_engine.rs (100 lines) - trait and factory
gpu_compute_engine_metal.rs (250 lines) - Metal implementation
gpu_compute_engine_mlx.rs (230 lines) - MLX implementation
```

## Dependency Analysis

### Critical Paths (must refactor together)

1. **Tokenizer Path**:
   - vocabulary.rs → tokenizer.rs → model_loader.rs
   - Cannot split without coordination

2. **Backend Selection Path**:
   - backend_selector.rs → unified_backend.rs → inference_engine.rs
   - Extraction order matters

3. **GPU Path**:
   - gpu_context.rs → gpu_compute_engine.rs → gpu_batch_scheduler.rs
   - Tightly coupled

4. **Cache Path**:
   - model_cache.rs → cache_optimizer.rs → model_cache_manager.rs
   - Shared data structures

### Recommended Extraction Order

Based on dependencies:

1. **Foundation** (low-level abstractions):
   - transformer_components.rs (independently testable)
   - transformer_layers.rs (builds on components)
   - layer_norm.rs (standalone)

2. **Backend Variants** (can extract independently):
   - unified_backend.rs → variant files
   - backend_selector.rs cleanup

3. **Sampling & Generation** (independently testable):
   - sampling.rs → concern extraction
   - temperature.rs, top_k.rs, etc.

4. **Model Management** (handle dependencies carefully):
   - model_loader.rs (after vocab/tokenizer clear)
   - model_registry.rs (after loader clear)
   - model_cache.rs (after loader and registry clear)

5. **GPU Integration** (can work in parallel):
   - gpu_compute_engine.rs → variant files
   - gpu_batch_scheduler.rs (after compute engine)

6. **Core Inference** (extract last, depends on everything):
   - pure_rust_backend.rs → concern extraction
   - llama_inference.rs → concern extraction
   - inference_engine.rs (after backends ready)

## Success Criteria

### Per-File Compliance
- ✅ All 43 files ≤150 lines
- ✅ All functions ≤25 lines
- ✅ All functions ≤3 parameters
- ✅ All complexity M≤3

### Behavioral
- ✅ All 1,288 tests passing
- ✅ Zero clippy warnings
- ✅ Zero regressions
- ✅ 100% backward compatibility

### Documentation
- ✅ Refactoring patterns documented
- ✅ New modules documented
- ✅ Extraction rationale clear

## Risk Mitigation

### Risks to Watch
1. **Breaking changes**: Use pub use re-exports to maintain API
2. **Circular dependencies**: Refactor in correct order
3. **Performance regression**: Test inference speed after each major refactor
4. **Test breakage**: Run tests after each commit
5. **Over-extraction**: Create modules with actual purpose, not arbitrary splits

### Mitigation Strategies
1. **Conservative approach**: Extract when there's clear single responsibility
2. **Test-first**: Run `pnpm test` after each commit
3. **Small commits**: Break into logical, reviewable chunks
4. **Backward compatibility**: Always maintain public API
5. **Parallel testing**: Run tests during refactoring to catch issues early

## Progress Tracking

| Phase | Files | Target | Status |
|-------|-------|--------|--------|
| 11 | 262 → 0 (core) | Core refactored | ✅ COMPLETE |
| 12 | Extract tests | Facades clean | ✅ COMPLETE |
| 13 | 43 files | All ≤150 lines | ⏳ IN PROGRESS |

## Next Immediate Steps

1. **Analyze dependencies** - Map out import relationships
2. **Create extraction utilities** - Helpers for common patterns
3. **Start with transformers** - Foundation layer, independently testable
4. **Run tests continuously** - Ensure zero regressions
5. **Document patterns** - Make extraction repeatable

## Expected Timeline

| Session | Task | Files | Impact |
|---------|------|-------|--------|
| 1 (Now) | Plan & analysis | - | Foundation |
| 2-4 | Transformer refactor | 2 | ~200 lines |
| 5-6 | Sampling refactor | 1 | ~150 lines |
| 7-8 | GPU backends | 3-4 | ~400 lines |
| 9-10 | Model management | 3-4 | ~300 lines |
| 11-12 | Core inference | 2 | ~800 lines |
| 13+ | Remaining | ~25 | ~9,000 lines |
| Total | 100% compliance | 43 | 21,625 lines |

---

## Key Decision: Progressive vs Total Refactor

**Option 1: Progressive** (Recommended)
- Refactor 2-3 modules per session
- Smaller commits
- Easier to review
- Lower risk
- Slower but safer

**Option 2: Total**
- Attempt all 43 files at once
- Massive refactoring
- Higher risk
- Faster completion
- Harder to review

**Recommendation**: Progressive approach with clear milestones

---

**Status**: Ready to begin Phase 13, Session 1  
**Next Action**: Analyze transformer module structure and plan first extraction
