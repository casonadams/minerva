# Phase 4: Compute Graphs - Design Document

## Overview

Implement directed acyclic graph (DAG) based compute graphs with operation fusion to achieve 2-5x speedup over naive sequential execution.

## Architecture

### Core Concepts

1. **Operations** - Individual tensor operations (matmul, add, gelu, softmax, etc.)
2. **Compute Node** - Wrapper around operation with dependencies
3. **Compute Graph** - DAG of nodes representing full inference
4. **Fusion** - Combining multiple operations into single fused kernel
5. **Execution Engine** - Topological sort + execute with memory reuse

### Key Operations to Support

```
Transformer Blocks:
├── Linear/MatMul (dense layers)
├── Add (residual connections)
├── LayerNorm / RMSNorm
├── GELU / Activation functions
├── Softmax (attention)
├── Scaled Dot-Product Attention
└── GroupNorm (optional)
```

### Fusion Opportunities

**Common Patterns** (2-5x speedup potential):

1. **Linear + Add (Residual)**
   ```
   y = matmul(x, W) + x  [fused]
   ```
   Saves intermediate allocation for matmul output

2. **Linear + LayerNorm**
   ```
   y = layernorm(matmul(x, W))  [fused]
   ```
   Single pass through both operations

3. **Linear + Add + GELU (FFN)**
   ```
   y = gelu(matmul(matmul(x, W1), W2) + x)  [fused]
   ```
   Critical path in transformers (3 ops → 1)

4. **Attention + Add (Residual)**
   ```
   y = softmax_attn(Q, K, V) + x  [fused]
   ```
   Avoid intermediate attention output tensor

5. **Softmax + Scale**
   ```
   y = scale(softmax(x))  [fused]
   ```
   Common in attention normalization

## Implementation Plan

### File Structure

```
src-tauri/src/inference/mlx_native/
├── compute_graph.rs (120L)
│   ├── ComputeGraph struct
│   ├── Operation enum
│   ├── ComputeNode struct
│   └── add_operation() / build()
│
├── compute_graph_ops.rs (100L)
│   ├── MatMulOp
│   ├── AddOp
│   ├── GeluOp
│   ├── LayerNormOp
│   ├── SoftmaxOp
│   └── trait ExecutableOp
│
├── compute_graph_fusion.rs (100L)
│   ├── FusionPattern enum
│   ├── detect_patterns()
│   ├── fuse_linear_add()
│   ├── fuse_linear_gelu()
│   └── fuse_attention_residual()
│
├── compute_graph_executor.rs (80L)
│   ├── GraphExecutor struct
│   ├── topological_sort()
│   ├── execute_node()
│   └── execute_graph()
│
├── compute_graph_test.rs (60L)
│   ├── test_graph_creation
│   ├── test_fusion_patterns
│   ├── test_execution
│   └── test_memory_reuse
│
└── mod.rs
    └── pub use compute_graph::ComputeGraph
```

### Type Definitions

```rust
pub enum Operation {
    MatMul { out_shape: (usize, usize) },
    Add,
    Gelu,
    LayerNorm { eps: f32 },
    Softmax { dim: usize },
    ScaledDotProductAttention { scale: f32 },
}

pub struct ComputeNode {
    id: NodeId,
    operation: Operation,
    inputs: Vec<NodeId>,
    output_shape: ArrayShape,
}

pub struct ComputeGraph {
    nodes: HashMap<NodeId, ComputeNode>,
    operations: HashMap<NodeId, Box<dyn ExecutableOp>>,
    dependencies: HashMap<NodeId, Vec<NodeId>>,
}

pub enum FusionPattern {
    LinearAdd,          // matmul + add
    LinearGelu,         // matmul + gelu
    LinearAddGelu,      // matmul + add + gelu (FFN)
    AttentionResidual,  // attention + add
    SoftmaxScale,       // softmax + scale
}
```

### Execution Flow

```
1. Build Graph
   ├── Add operations (nodes + edges)
   ├── Specify dependencies
   └── Mark outputs

2. Detect Patterns
   ├── Pattern matching on node sequences
   ├── Validate fusion safety
   └── Mark fusible nodes

3. Optimize Graph
   ├── Merge nodes into fused operations
   ├── Reorder operations (preserve deps)
   └── Allocate workspace buffers

4. Execute
   ├── Topological sort
   ├── For each node:
   │   ├── Dequant if needed (Phase 3)
   │   ├── Execute fused op
   │   ├── Store intermediate (if needed)
   │   └── Free old tensors (memory reuse)
   └── Return output
```

## Performance Analysis

### Memory Efficiency

**Naive (Sequential)**:
```
Layer i:
  z = matmul(x, W)        [alloc: hidden_size]
  y = z + x               [alloc: hidden_size]  (z still in memory)
  h = gelu(y)             [alloc: hidden_size]  (y still in memory)
Total: 3× hidden_size allocations
```

**Fused**:
```
Layer i:
  h = fused_linear_add_gelu(x, W)  [alloc: hidden_size]
Total: 1× hidden_size allocations
```

**Speedup from reduced allocation**: 2-3x (memory bandwidth bound)

### Compute Efficiency

**Attention with residual**:
```
Naive:
  attn_out = attention(Q, K, V)     [alloc, compute]
  y = attn_out + x                  [compute only]
  Total passes: 2 (separate kernels)

Fused:
  y = attention_residual(Q, K, V, x) [alloc, compute]
  Total passes: 1 (single fused kernel)
```

**Speedup**: 1.5-2x for attention-heavy operations

## Test Strategy

### Unit Tests

1. **Graph Construction** (10 tests)
   - Add single operation
   - Add multiple operations
   - Dependency tracking
   - Invalid dependency detection

2. **Pattern Detection** (15 tests)
   - Detect linear-add pattern
   - Detect linear-gelu pattern
   - Detect attention-residual pattern
   - False positive prevention

3. **Execution** (10 tests)
   - Execute single node
   - Execute sequential nodes
   - Execute with fusion
   - Output correctness verification

4. **Memory Reuse** (5 tests)
   - Intermediate tensor freeing
   - Workspace allocation
   - Peak memory tracking

### Integration Tests

1. **Full Transformer Block** (5 tests)
   - Self-attention block
   - FFN block
   - Combined block
   - Compare naive vs fused output (should be identical)

2. **Multi-layer** (3 tests)
   - 2-layer transformer
   - 4-layer transformer
   - Verify chain correctness

## Implementation Order

1. **Phase 4.1** - Core Graph Structure
   - Implement ComputeGraph, ComputeNode
   - Implement Operation enum
   - Create graph building API

2. **Phase 4.2** - Basic Operations
   - Implement ExecutableOp trait
   - Implement MatMul, Add, Gelu ops
   - Basic execution engine

3. **Phase 4.3** - Pattern Detection & Fusion
   - Implement pattern detection
   - Implement fusion combiners
   - Merge operations

4. **Phase 4.4** - Memory Optimization
   - Implement workspace management
   - Implement memory reuse
   - Implement topological sorting

5. **Phase 4.5** - Testing & Benchmarking
   - Comprehensive test suite
   - Benchmark vs naive
   - Document performance gains

## Success Criteria

- ✅ All 30+ tests passing
- ✅ 2-5x speedup vs naive sequential
- ✅ Zero clippy warnings
- ✅ All files ≤150 lines
- ✅ Memory efficiency verified
- ✅ Pattern detection accurate (no false positives)
- ✅ Output correctness (fused = naive)

## Time Estimate

- Total: 4-6 hours
- Phase 4.1: 1 hour
- Phase 4.2: 1.5 hours
- Phase 4.3: 1.5 hours
- Phase 4.4: 1 hour
- Phase 4.5: 1 hour

## Risk Mitigation

**Risk**: Fusion creates correctness issues
- **Mitigation**: Always compare fused output to naive, extensive testing

**Risk**: Code becomes >150 lines
- **Mitigation**: Careful modular design from start, split as needed

**Risk**: Fusion patterns don't generalize
- **Mitigation**: Start with common patterns only (linear-add-gelu, attention)

## Next Phase Integration

Phase 5 (Metal GPU) will use this compute graph to:
1. Compile fused operations to Metal shaders
2. Execute on GPU with minimal CPU overhead
3. Achieve 5-10x speedup from GPU parallelization
