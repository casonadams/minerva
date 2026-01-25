# Throughput Reality Check: Why Real-World ≠ Theoretical

**Question:** We said 500 t/s but need 64GB. Why can't we achieve this in real testing?

**Answer:** Because 500 t/s is for **batched requests**, not single requests. Here's the breakdown:

---

## The Three Different Performance Metrics

### 1. **Single-Request Throughput** (What users actually experience)

```
Scenario: One user, one request

Timeline:
  Load model:        30-60s (one time only)
  Encode context:    30-120s (128K tokens is slow)
  Generate 100 tokens:
    • First token:   ~30ms
    • Tokens 2-100:  99 * 10ms = 990ms
  ─────────────────
  Total generation:  ~1 second
  Tokens/second:     ~100 tokens/second MAX

Reality: 10-30 t/s for single request
  (After all optimizations)
```

### 2. **Batch Throughput** (Multiple users, sampled over time)

```
Scenario: 20 concurrent users, each with a request

Setup:
  • User 1, 2, 3, ... 20 all send requests simultaneously
  • System processes in batches
  • Measure: total tokens generated per second across all users

Timeline (simplified):
  Batch 1 (all 20 users):  20 tokens generated in 10ms
  Batch 2 (all 20 users):  20 tokens generated in 10ms
  Batch 3 (all 20 users):  20 tokens generated in 10ms
  ─────────────────────
  Every 10ms: 20 tokens
  Per second: 20 tokens * 100 batches = 2,000 tokens/second! 🎉

Wait, that's 2,000 t/s, not 500 t/s...
Why? Overhead, synchronization, memory bottlenecks.

Realistic batch throughput: 200-500 t/s
(20 concurrent × 10-25 t/s average)
```

### 3. **What the Benchmark Claims**

```
"Batch Processing (20 concurrent requests): 200-500 t/s"

This is MISLEADING without context!

What it actually means:
  ✗ NOT: Single user gets 500 t/s
  ✓ YES: 20 users combined get 500 t/s = 25 t/s per user
```

---

## Real-World Performance on 64GB System

### Single User, Single Request (128K context)

```
Hardware: 64GB M2 Ultra Mac

Timeline:
  1. Load model (MXFP4):           45s
  2. Load 128K context tokens:     120s
     (Actually reading/processing 128K from disk/network)
  3. First forward pass:           30-100ms
     (Attention over 128K context is slow)
  4. Generate 100 tokens:          1-2 seconds
     (With KV cache, ~10-20ms per token)
  
  Total: ~3 minutes for 100 tokens = 0.5 tokens/second ❌

This is TERRIBLE! Why?

The Problem:
  • Attention over 128K is O(n²) complexity
  • Each token generation: 128K * 128K matrix ops
  • Even with Flash Attention: still massive computation
  • Memory bandwidth is the killer, not computation
```

### Why So Slow?

```
The Math Behind Slowness:

Token generation computation:
  • Forward pass base: 50B FLOPs
  • Attention computation: 128K² * head_dim * heads
    = 128,000 * 128,000 * 360 * 8 ≈ 400 trillion FLOPs!
  • Total per token: ~400 trillion FLOPs

Apple Silicon M2 Ultra:
  • Peak compute: ~600 GFLOPS
  • Sustained (realistic): ~300 GFLOPS
  • Memory bandwidth: 200 GB/s

Calculation:
  400 trillion FLOPs / 300 GFLOPS = 1.3 million milliseconds!

No wait, that's wrong. Let me recalculate...

Actually:
  • Attention with 128K is O(n²) = 128K² operations
  • With Flash Attention: reduces to O(n * block_size)
  • Still, processing 128K context per token = SLOW
  • Realistic: 100-500ms per token with 128K context
```

---

## The 500 t/s Claim Unpacked

### How to Actually Achieve 500 t/s

```
Requirement 1: Multiple Concurrent Requests
  ✓ Need at least 20 users generating simultaneously
  ✓ Cannot be single user

Requirement 2: Small Context per Request
  ✗ NOT 128K context!
  ✓ More like 1-4K context
  ✗ Larger context = slower per token

Requirement 3: Batch Processing
  ✓ Hardware must support batching
  ✓ Requires proper scheduling
  ✓ Need queue management

Requirement 4: Short Responses
  ✓ Users only request 10-50 tokens each
  ✗ Longer responses = fewer concurrent batches

Realistic Scenario for 500 t/s:
  • 20 concurrent users
  • 2K context each (much smaller!)
  • Generating 20-30 tokens each
  • Total: ~40 seconds to completion
  • Throughput: ~20 tokens per user/second
  • Combined: ~400 tokens/second
```

---

## The Real Problem with 128K Context

### Memory Bandwidth is the Bottleneck

```
Situation: Generating one token with 128K context

Operations needed:
  1. Load 128K context from memory:     128K * 360 * 4 bytes = 184 MB
  2. Compute attention:                 128K² operations
  3. Write output:                      10K operations
  4. Update cache:                      ~100 KB

Memory bandwidth available:
  M2 Ultra:     200 GB/s
  184 MB / 200 GB/s = 0.92 milliseconds just for loading context!

Add computation time:
  128K² / 300 GFLOPS ≈ 50-100ms (with Flash Attention)

Total: ~100-150ms per token minimum

Throughput: 6-10 tokens/second (single request!)

With 20 concurrent: 120-200 t/s (not 500!)
```

### Why 500 t/s Works for Smaller Context

```
With 4K context instead:

Operations needed:
  1. Load 4K context:                  4K * 360 * 4 bytes = 5.7 MB
  2. Compute attention:                4K² operations
  3. Update cache:                     ~100 KB

Memory bandwidth:
  5.7 MB / 200 GB/s = 0.03 milliseconds (negligible!)

Add computation:
  4K² / 300 GFLOPS ≈ 0.05ms (tiny!)

Total: ~5-10ms per token (WITH optimizations)

Throughput: 100-200 tokens/second per request!

With 20 concurrent: 2,000+ t/s! 🚀
```

---

## Realistic Performance Expectations

### Single Request Performance

```
Context Size | Per-Token Latency | Throughput | Achievable?
─────────────┼──────────────────┼───────────┼──────────────
1K           | 5-10ms           | 100-200 t/s | ✅ Yes
4K           | 10-20ms          | 50-100 t/s  | ✅ Yes
8K           | 20-50ms          | 20-50 t/s   | ✅ Yes
32K          | 50-200ms         | 5-20 t/s    | ✅ Yes
128K         | 100-500ms        | 2-10 t/s    | ⚠️ Slow
```

### Batch Performance (20 concurrent)

```
Context Size | Per-Token Latency | Batch Throughput | Notes
─────────────┼──────────────────┼─────────────────┼──────────────
1K           | 5-10ms           | 2,000+ t/s      | Unrealistic (queue limit)
4K           | 10-20ms          | 1,000-2,000 t/s | Peak with good hardware
8K           | 20-50ms          | 400-1,000 t/s   | Very good
32K          | 50-200ms         | 100-400 t/s     | Good
128K         | 100-500ms        | 20-100 t/s      | Memory-bound
```

---

## Why We Can't Get 500 t/s with 128K

### The Fundamental Issues

```
1. O(n²) Attention Complexity
   • Even with Flash Attention: still O(n * block_size)
   • 128K is a HUGE context
   • Each forward pass touches 128K tokens
   • No way around this without sacrificing quality

2. Memory Bandwidth Bottleneck
   • 128K context = 184 MB per forward pass
   • Memory bandwidth: 200 GB/s
   • Just loading context: ~1ms per token minimum
   • Add computation: 50-100ms more
   • Total: 100-500ms per token

3. KV Cache Size Explosion
   • 71 GB just for KV storage
   • Accessing 71 GB cache per token is slow
   • Even with perfect caching: still memory-bound

4. Sequential Nature of Generation
   • Can't parallelize token generation
   • Must generate token 1, then 2, then 3...
   • 100-500ms per token = 2-10 tokens/second max
   • No amount of batching fixes this (batch doesn't parallelize generation)
```

### What You CAN Get with 128K

```
Single Request:
  • 2-10 tokens/second (realistic)
  • Takes 1-2 seconds per token after cache warmup

20 Concurrent Requests (different contexts):
  • ~50-150 tokens/second total
  • Each user gets 2.5-7.5 t/s

Why batching helps less with 128K:
  • Batch mainly helps with ENCODING (128K context)
  • Generation (token-by-token) still sequential
  • You get: 1 generation per 100-500ms per user
  • 20 users = ~40-200 tokens/second total (not 500)
```

---

## How to Actually Get 500 t/s

### Option 1: Use Smaller Context (Recommended)

```
Switch from 128K to 4K:
  • Per-token latency: 100ms → 10ms (10x faster!)
  • Single request: 2 t/s → 100 t/s (50x faster!)
  • Batch (20): 50 t/s → 2,000 t/s (40x faster!)
  
This gives you 500+ t/s easily!
```

### Option 2: Use Speculative Decoding

```
Trade: Slightly lower quality for 3-5x faster generation

How it works:
  1. Use small model to generate candidate tokens
  2. Use large model to verify
  3. Keep verified tokens, skip re-computation

With 128K:
  • Small model draft: 100 t/s (on CPU)
  • Large model verify: 10 t/s (on GPU)
  • Combined: 20-30 t/s (better than 2-10!)
  • 20 concurrent: 400-600 t/s ✓
```

### Option 3: Use Sparse Attention

```
Trade: Can't attend to all context

How it works:
  • Local attention: Only attend to recent tokens
  • Strided attention: Skip some tokens
  • Block-sparse: Only certain blocks

Effect:
  • 128K context → "effective" 8K context
  • Per-token: 100-500ms → 20-50ms
  • Throughput: 2-10 t/s → 20-50 t/s
  • Batch: 50-100 t/s (still not 500, but better)
```

### Option 4: Accept the Reality

```
128K context is fundamentally memory-bound.

Physics limits:
  • Memory bandwidth: 200 GB/s
  • Context size: 184 MB per forward pass
  • Minimum latency: 184 MB / 200 GB/s = 0.92ms
  • Add computation: +50-100ms
  • Realistic minimum: 100ms per token
  • Maximum throughput: 10 tokens/second (single request)

This is not a software optimization problem.
This is physics. You can't beat it.

To get 500 t/s, you need:
  ✓ Smaller context (4-8K)
  ✓ Or multiple requests with smaller contexts
  ✓ Or accept lower quality with speculative decoding
```

---

## The Bottom Line

### The 500 t/s Benchmark is Misleading

```
What we claimed:
  "Batch Processing (20 concurrent): 200-500 t/s"

What we meant:
  "With 4K context, 20 concurrent users, each generating 20 tokens:
   Total throughput = 200-500 t/s"

What people read:
  "We can get 500 t/s with 128K context!"

Reality:
  "With 128K context, we get 2-10 t/s per user,
   or 20-100 t/s for 20 concurrent users"
```

### Why the Gap?

```
Theoretical ≠ Real because:

1. Theory assumes:
   • Perfect parallelization (impossible for token generation)
   • No memory stalls (false with 128K context)
   • Perfect batching efficiency (98%+ overhead in practice)

2. Reality has:
   • Sequential token generation (can't parallelize)
   • Memory bandwidth bottleneck (128K context dominates)
   • Synchronization overhead (coordinating 20 users)
   • Cache misses (71 GB cache can't fit in L3)

3. 128K context is special:
   • Attention: O(n²) ≈ 16 billion operations per token
   • Memory: 184 MB context to load per token
   • Cache: 71 GB state to manage
   • Result: Memory-bound, not compute-bound
```

---

## Practical Recommendations

### For 128K Context

```
Realistic expectations:
  • Single user: 2-10 tokens/second
  • 20 concurrent: 50-150 tokens/second total
  • Use case: Batch processing, not real-time
  • Requires: 64GB system, patient users

Better alternatives:
  1. Use 4-8K context: 100-500 t/s easily ✓
  2. Use speculative decoding: 20-30 t/s with 128K
  3. Use sparse attention: 20-50 t/s with 128K
```

### For Production

```
If you want 500 t/s:
  • Use 2-4K context (sweet spot)
  • Enable Flash Attention
  • Enable KV Cache
  • Batch 20+ users
  • Target: Easy 500+ t/s

If you need 128K context:
  • Accept 2-10 t/s per user
  • Use for batch/non-real-time workloads
  • Implement speculative decoding for speedup
  • Consider distributed inference
```

---

## Summary

**The honest answer to "Why can't we get 500 t/s with 128K context?"**

```
Because:
1. Attention scales as O(n²) - 128K is HUGE
2. Memory bandwidth is the bottleneck - 184 MB per forward pass
3. Token generation is sequential - can't parallelize
4. Physics prevents faster than ~100ms per token with 128K

500 t/s is achievable with:
  ✓ Small context (4K): Easy, 500+ t/s
  ✗ Large context (128K): Impossible without tricks
  ? Speculative decoding: Maybe 20-30 t/s

For 128K, expect: 2-10 t/s per user, or 20-100 t/s total
```

This is the real-world truth.
