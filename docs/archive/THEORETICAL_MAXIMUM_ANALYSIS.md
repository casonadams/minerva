# Theoretical Maximum Throughput Analysis: What's Achievable?

**Date:** January 25, 2026  
**Question:** What's the absolute maximum tokens/sec we can achieve?  
**Answer:** 500-2000+ t/s depending on hardware and optimization level

---

## The Hardware Reality (Your Machine)

First, let's understand what we're working with. I need to know:
- GPU type? (NVIDIA, AMD, Apple Metal?)
- VRAM? (8GB, 16GB, 40GB+?)
- CPU cores? (8, 16, 32+?)

For this analysis, I'll assume **modern GPU** (RTX 4090 equivalent or Apple M-series):

```
Hardware Specs (Assumed):
  GPU Memory: 24GB (RTX 4090) or 40GB+ (H100)
  GPU Compute: 130 TFLOPS (4090) or 1400 TFLOPS (H100)
  Memory Bandwidth: 1.0 TB/s (4090) or 3.3 TB/s (H100)
  CPU: 16+ cores at 4+ GHz
```

---

## Theoretical Limits (Physics-Based)

### The Roofline Model

Every system has a **compute roofline** - a theoretical performance ceiling:

```
Performance = min(Peak Compute, Peak Memory Bandwidth * Arithmetic Intensity)

Where:
  Peak Compute = TFLOPS (GPU capability)
  Peak Memory Bandwidth = GB/s available
  Arithmetic Intensity = FLOPs per byte of memory moved
```

### For LLM Inference (TinyLlama-1.1B)

**Forward Pass Computation:**
```
FLOPs per token = 2 * seq_len * hidden_size * (4/3 * hidden_size)
                = 2 * seq_len * 2048 * (4/3 * 2048)
                = 11.3 * seq_len * 10^9 FLOPs

For seq_len=1 (generation): 11.3 billion FLOPs
For seq_len=100 (context): 1.13 trillion FLOPs
```

**Memory Requirements:**
```
Weights: 1.1B params * 4 bytes (FP32) = 4.4 GB
Activations: batch_size * seq_len * 2048 * 4 bytes

For batch=1, seq_len=1:
  Total memory movement ≈ 4.4 GB (weights loaded once)
  Arithmetic intensity = 11.3B FLOPs / 4.4B bytes = 2.57 FLOPs/byte
```

**Theoretical Maximum:**
```
Peak Compute (GPU): 130 TFLOPS (RTX 4090)
Peak Bandwidth: 1 TB/s = 1000 GB/s

Roofline = min(130 TFLOPS, 1000 GB/s * 2.57 FLOPs/byte)
         = min(130 TFLOPS, 2.57 TFLOPS)
         = 2.57 TFLOPS (memory-bound!)

Max throughput = 2.57 TFLOPS / 11.3B FLOPs per token
               = 227 tokens/sec
```

### For Mistral-7B

```
FLOPs per token = 2 * 7B * hidden_size * (4/3 * hidden_size)
                = 2 * 7B * 4096 * (4/3 * 4096)
                = 297 billion FLOPs

Memory = 7B params * 4 bytes = 28 GB
Arithmetic intensity = 297B / 28B = 10.6 FLOPs/byte

Roofline = min(130 TFLOPS, 2.57 TFLOPS)  [still memory-bound]
Max throughput = 2.57 TFLOPS / 297B FLOPs
               = 8.6 tokens/sec
```

**Key Insight:** Larger models are MORE memory-bound!

---

## How to Break Through the Memory Ceiling

### Method 1: Quantization (4-bit)

**What:** Reduce weights from FP32 (4 bytes) to INT4 (0.5 bytes)

```
4-bit quantization:
  Memory: 28 GB → 3.5 GB (8x reduction!)
  Arithmetic intensity increases: 10.6 → 85 FLOPs/byte
  
Roofline = min(130 TFLOPS, 1000 * 85)
         = min(130 TFLOPS, 85 TFLOPS)
         = 85 TFLOPS (still compute-bound)

Max throughput = 85 TFLOPS / 297B FLOPs
               = 286 tokens/sec (33x improvement!)
```

**Mistral-7B with INT4:**
```
Expected throughput: 286 * (1.1B / 7B) = 45 tokens/sec
With KV cache: 90 tokens/sec
With batching: 500+ tokens/sec
```

### Method 2: Speculative Decoding

**What:** Use a small model to predict next token, then verify with large model

```
Idea:
  1. Draft model (Phi-2, 3B) generates 4 tokens (fast)
  2. Verify with Mistral-7B in batch (fast)
  3. If correct, keep tokens; else revert and continue
  
Result: Process 4-8 tokens simultaneously!

Speedup: 3-5x with minimal accuracy loss
```

### Method 3: Tensor Parallelism + Pipeline Parallelism

**What:** Distribute model across multiple GPUs/processes

```
Single GPU (24GB VRAM):
  Can fit: Mistral-7B (28GB) - just barely with quantization
  
Dual GPU (2x24GB = 48GB VRAM):
  Can fit: Full precision
  Can parallelize across 2 GPUs
  Speedup: ~1.8-1.9x (some communication overhead)
  
4x GPU (4x24GB = 96GB VRAM):
  Can fit: Multiple copies + batching
  Speedup: ~3.5-3.8x
```

---

## Realistic Maximum by Hardware

### Consumer GPU (RTX 4090, 24GB)

**Single Model (quantized):**
```
TinyLlama-1.1B INT8:    200-300 t/s    ✓ Realistic
Mistral-7B INT4:        40-60 t/s      ✓ Realistic
Mistral-7B FP16:        15-25 t/s      ✓ Realistic (OOM edge)
```

**With Speculative Decoding:**
```
Mistral-7B + Phi-2:     120-200 t/s    ✓ Achievable
```

**With Batching (Batch=8):**
```
Mistral-7B + Batching:  300-500 t/s    ✓ Achievable
```

### High-End GPU (H100, 80GB)

```
Mistral-7B FP32:        100-150 t/s    ✓ Realistic
Llama-70B FP32:         20-30 t/s      ✓ Realistic
Llama-70B FP8:          150-200 t/s    ✓ Realistic
```

### Apple Metal (M-series, unified memory)

```
TinyLlama FP32:         50-100 t/s     ✓ Realistic
Mistral-7B FP32:        30-50 t/s      ✓ Realistic (memory-unified helps)
```

---

## The Stack for Maximum Performance

### Tier 1: Baseline (50-100 t/s)
1. GPU backend (burn-rs)
2. KV cache
3. Basic Flash Attention

**Time:** 2-3 days  
**Effort:** Easy

### Tier 2: Aggressive (100-200 t/s)
4. Operator fusion
5. Memory-efficient attention
6. Request batching (batch=4-8)

**Time:** 2-3 days  
**Effort:** Medium

### Tier 3: Extreme (200-500+ t/s)
7. INT8/INT4 quantization
8. Speculative decoding
9. Tensor parallelism
10. Multi-GPU batching

**Time:** 3-5 days  
**Effort:** Hard

---

## My Recommendation: Three-Level Strategy

### PHASE 4A: Get 150+ t/s (2-3 days)
**Focus:** Single request, TinyLlama-1.1B

```
Target: 150-200 t/s
Techniques:
  ✓ GPU backend (burn-rs)
  ✓ KV cache
  ✓ Flash Attention
  ✓ Operator fusion
  ✓ Simple batching
```

### PHASE 4B: Get 300+ t/s (2-3 days)
**Focus:** Batch processing, Mistral-7B

```
Target: 300-500 t/s
Techniques:
  ✓ Everything from 4A
  ✓ INT8 quantization
  ✓ Batch processing (batch=8)
  ✓ Request pipelining
```

### PHASE 4C: Get 1000+ t/s (3-5 days)
**Focus:** Multi-request production scale

```
Target: 1000+ t/s sustained
Techniques:
  ✓ Everything from 4B
  ✓ Speculative decoding
  ✓ Tensor parallelism (if multi-GPU)
  ✓ Advanced batching strategies
  ✓ Request queue optimization
```

---

## Maximum by Use Case

### Use Case 1: Single Interactive Request
```
User sends prompt, waits for response

Optimal: Minimize latency (TTFT + streaming)
Not throughput

Target: <100ms TTFT, <10ms per token
Achievable: With optimizations above
```

### Use Case 2: Batch Processing
```
100 requests come in at once

Optimal: Maximize throughput
Batch all together

Target: 500-1000 t/s
Achievable: With batching + quantization
```

### Use Case 3: Production API Server
```
Requests arrive continuously at varying rates

Optimal: Sustained throughput + low latency
Smart batching + fallback

Target: 300-500 t/s sustained
Achievable: With Tier 2 optimizations
```

---

## The Math for Your Target: 150+ t/s

**Goal:** 150+ tokens/second for TinyLlama-1.1B

**Requirements:**
```
Tokens: 150
Time: 1 second
FLOPs per token: 11.3 billion (for seq_len=1)

Total FLOPs needed: 150 * 11.3B = 1.695 trillion FLOPs
Time available: 1 second

Required throughput: 1.695 TFLOPS
GPU capability: 130+ TFLOPS

Headroom: 130 / 1.695 = 76.6x

This is VERY achievable!
```

**Actual constraint: Memory bandwidth**

```
Memory to move per token:
  - Weights: 1.1B * 4 bytes = 4.4 GB (amortized)
  - Activations: 2048 * 2048 * 4 bytes = 16 MB
  - Total per token: ~100 MB (with batching amortization)

Bandwidth needed: 150 tokens/sec * 100 MB = 15 GB/s
GPU bandwidth: 1000 GB/s

Headroom: 1000 / 15 = 67x

Still VERY achievable!
```

---

## Absolute Maximum We Can Achieve

### If We Optimize Everything (Tier 3)

**Hardware:** RTX 4090 or equivalent

```
Single Request (Batch=1):
  TinyLlama-1.1B:   200-300 t/s
  Mistral-7B:       60-100 t/s

Batch=8:
  TinyLlama-1.1B:   1200-2400 t/s (!)
  Mistral-7B:       400-800 t/s (!)

Production (mixed load):
  Sustained:        500-1000 t/s
```

### If GPU is H100 or similar
```
Single Request:
  Mistral-7B FP32:  150-250 t/s

Batch=8:
  Mistral-7B:       1000-2000 t/s (!)
```

---

## My Answer: Aim for This

### Phase 4 Goal (Realistic, Achievable)
```
✓ TinyLlama-1.1B: 150-200 t/s single request
✓ Mistral-7B: 80-120 t/s single request
✓ With batch=8: 500-1000 t/s
```

### The Path
```
Days 1-2:   GPU backend + KV cache + Flash Attention
            → 100-150 t/s

Days 2-3:   Operator fusion + quantization
            → 150-200 t/s

Days 3-4:   Batching + pipelining + validation
            → 300-500 t/s sustained
```

### Why This Level

**Not aiming for 2000+ t/s because:**
1. Requires speculative decoding (complex)
2. Requires tensor parallelism (multi-GPU)
3. Requires advanced quantization (research-level)
4. Takes 1-2 weeks instead of 4 days

**But 150-200 t/s because:**
1. Achievable in 4 days
2. Uses well-known techniques
3. Works on single GPU
4. 5-10x better than current
5. Production-ready quality

---

## The Aggressive Option: Go for 500+ t/s

If you want to be **really ambitious** and have the time, here's how:

### The 500+ t/s Plan (Extended, 6-7 days)

**Day 1-2:** Base GPU backend + KV cache (100-150 t/s)

**Day 3:** Flash Attention + fusion (150-200 t/s)

**Day 4:** INT8 quantization (200-300 t/s)

**Day 5:** Request batching (300-500 t/s with batch=4)

**Day 6:** Speculative decoding setup (400-600 t/s)

**Day 7:** Multi-GPU tensor parallelism (600-1000+ t/s)

**Result:** 500-1000 t/s sustained throughput

**Effort:** 50-60 hours (intense week)

---

## What I Recommend

### Option A: Conservative (150+ t/s, 4 days)
**Start now, finish in 4 days**
- GPU backend
- KV cache
- Flash Attention
- Operator fusion
- Basic batching

Result: 150-200 t/s, solid production code

### Option B: Ambitious (300-500 t/s, 6-7 days)
**Extend timeline by 2-3 days**
- Everything in Option A
- INT8 quantization
- Advanced batching
- Speculative decoding prep

Result: 300-500 t/s, research-quality code

### Option C: Maximum (1000+ t/s, multi-GPU)
**For multi-GPU setup**
- Everything in Option B
- Tensor parallelism
- Multi-GPU batching
- Advanced kernel fusion

Result: 1000+ t/s, enterprise-grade code

---

## My Strong Recommendation

**Go with Option B: 300-500 t/s in 6-7 days**

**Why:**
1. Still reasonable timeline
2. Gets you into "research-grade" territory
3. Competitive with industry tools (ollama, vLLM lite)
4. INT8 quantization is worth it (4x memory benefit)
5. Speculative decoding is achievable with careful planning
6. If you get stuck, can ship with Option A (150 t/s)

**Not Option C because:**
1. Multi-GPU complexity adds 2-3 more days
2. Most people have single GPU anyway
3. Can always add later

---

## Final Answer to Your Question

### "What's the maximum achievable?"

**Theoretical Max:** 2000+ t/s (with everything optimized, multi-GPU)

**Realistic Max (single GPU):** 
- 200-300 t/s single request
- 500-1000 t/s batch processing

**What we should aim for:** **300-500 t/s in 6-7 days**

This is:
- ✅ Achievable with known techniques
- ✅ Beats industry standards
- ✅ Production-quality code
- ✅ Reasonable timeline
- ✅ Can ship incrementally (150 t/s → 300 t/s → 500 t/s)

---

## Timeline Summary

```
Days 1-2:  GPU backend         → 100-150 t/s
Days 2-3:  Optimizations       → 150-200 t/s
Days 3-4:  Quantization        → 200-300 t/s
Days 4-5:  Advanced batching   → 300-400 t/s
Days 5-6:  Speculative decode  → 400-500 t/s
Days 6-7:  Validation & tune   → Sustained 300-500 t/s
```

**Total effort:** 50-55 hours  
**Expected result:** 300-500 t/s sustained  
**Quality:** Production-ready

---

## My Vote: Go for 300-500 t/s

Let's build something that:
1. Is **actually fast** (300-500 t/s is VERY fast)
2. Is **completable** (6-7 days is reasonable)
3. Is **production-ready** (not beta/experimental)
4. Is **competitive** (matches or beats ollama/vLLM)
5. Has **clear milestones** (150→200→300→400→500)

**Shall we go for it?** 🚀

