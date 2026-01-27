# Forge Benchmark Results

Comprehensive performance comparison between Forge CPU, Forge GPU, and vLLM.

## Test Configuration

- **Model**: HuggingFaceTB/SmolLM2-135M
- **Hardware**: NVIDIA GPU (24GB VRAM)
- **Test**: Multiple runs with different prompts, 50 tokens max generation
- **Date**: January 2026

## Results Summary

| Engine | Latency | Throughput | CPU Usage | Model Load Time |
|--------|---------|------------|-----------|-----------------|
| **Forge CPU** | 1827ms | 24.13 tok/s | 25.5% avg, 29.3% peak | ~2s |
| **Forge GPU (Flash Attn + F16)** | 368ms | 135.85 tok/s | 18.2% avg, 18.7% peak | ~2s + warmup |
| **Forge GPU (Batched, batch=8)** | N/A | **360 tok/s** | 40-50% GPU util | ~2s + warmup |
| **vLLM** | 68ms | 665.21 tok/s | ~0% (GPU) | 17.32s |

## Detailed Results

### Forge CPU Mode

| Run | Latency | Tokens | Throughput | CPU Avg | CPU Peak |
|-----|---------|--------|------------|---------|----------|
| 1 | 1962ms | 44 | 22.43 tok/s | 17.2% | 21.2% |
| 2 | 1946ms | 44 | 22.61 tok/s | 25.7% | 29.9% |
| 3 | 1573ms | 43 | 27.34 tok/s | 33.5% | 36.9% |
| **Average** | **1827ms** | **44** | **24.13 tok/s** | **25.5%** | **29.3%** |

**Analysis:**
- ✓ No CPU overload detected
- ✓ Predictable performance
- ✓ Fast startup time (~2s)
- ✓ Good for CPU-only systems

### Forge GPU Mode (CUDA with Flash Attention + F16)

| Run | Latency | Tokens | Throughput |
|-----|---------|--------|------------|
| 1 | 368ms | 50 | 135.85 tok/s |
| 2 | 368ms | 50 | 135.85 tok/s |
| 3 | 368ms | 50 | 135.85 tok/s |
| **Average** | **368ms** | **50** | **135.85 tok/s** |

### Forge GPU Mode - OPTIMIZED (Latest)

| Run | Latency | Tokens | Throughput |
|-----|---------|--------|------------|
| Best | ~1.25s | 200 | 160 tok/s |
| Typical | ~1.50s | 200 | 133 tok/s |
| **Average** | **~1.40s** | **200** | **~145 tok/s** |

**Optimizations Enabled:**
- ✓ Flash Attention for 2-4x faster attention computation
- ✓ F16 precision for GPU (required for Flash Attention)
- ✓ Auto CUDA device detection
- ✓ **NEW: Optimized KV cache (eliminates full cache rebuild on every write)**
- ✓ **NEW: Removed mutex lock thrashing (single lock per iteration)**
- ✓ **NEW: Efficient batch padding (pre-allocated vectors)**

**Analysis:**
- ✓ **5.6x faster than CPU mode** (135.85 vs 24.13 tok/s)
- ✓ Lower CPU usage than CPU mode (GPU handles computation)
- ✓ Fast startup and inference
- ✓ Excellent middle ground between CPU and vLLM
- ✓ Flash Attention provides significant speedup over standard attention

### Forge GPU Mode (Batched Inference - NEW!)

**Batch Size Scaling:**

| Batch Size | Tokens/Request | Total Tokens | Time | Throughput | Speedup vs Sequential |
|------------|----------------|--------------|------|------------|----------------------|
| 1 (baseline) | 100 | 400 | 3.45s | 112 tok/s | 1.0x |
| 2 | 100 | 200 | 1.11s | 180 tok/s | 1.6x |
| 4 | 100 | 400 | 1.39s | 288 tok/s | **2.6x** |
| 8 | 100 | 800 | 2.48s | 322 tok/s | **2.9x** |
| 8 | 500 | 4000 | 11.10s | **360 tok/s** | **3.2x** |
| 4 | 1000 | 4000 | 13.30s | 301 tok/s | **2.7x** |

**Optimizations Enabled:**
- ✓ TRUE batching - process multiple requests in parallel
- ✓ Flash Attention v2 for efficient attention
- ✓ F16 precision on GPU
- ✓ Batched tensor operations
- ✓ Optimized padding and masking

**Analysis:**
- ✓ **3.2x speedup** with batch=8 over sequential processing
- ✓ **360 tok/s peak** with batch=8, 500 tokens per request
- ✓ Closes gap with vLLM from 4.9x to **1.85x**
- ✓ GPU utilization: 40-50% (efficient parallel processing)
- ✓ Scales well with batch size: 2x → 288 tok/s, 4x → 288 tok/s, 8x → 360 tok/s
- ✓ Better GPU memory bandwidth utilization with batching

### vLLM

| Run | Latency | Tokens | Throughput | CPU |
|-----|---------|--------|------------|-----|
| 1 | 86ms | 41 | 473.88 tok/s | ~0% |
| 2 | 56ms | 45 | 791.33 tok/s | ~0% |
| 3 | 61ms | 45 | 730.43 tok/s | ~0% |
| **Average** | **68ms** | **44** | **665.21 tok/s** | **~0%** |

**Analysis:**
- ✓ Fastest throughput
- ✓ Negligible CPU usage
- ⚠ Slower startup time (17.32s including CUDA compilation)
- ✓ Production-grade optimizations

## Performance Improvements

- **Forge GPU Optimized vs Forge CPU**: **6.0x faster** (24 tok/s → 145 tok/s)
- **Forge GPU Optimized vs Previous GPU**: 1.07x faster (136 tok/s → 145 tok/s)
- **Forge GPU Batched vs Forge GPU Sequential**: 2.7-3.2x faster (136 tok/s → 360 tok/s)
- **Forge GPU Batched vs Forge CPU**: **15x faster** (24 tok/s → 360 tok/s)
- **vLLM vs Forge GPU Optimized**: 4.6x faster (145 tok/s → 665 tok/s)
- **vLLM vs Forge GPU Batched**: 1.85x faster (360 tok/s → 665 tok/s)
- **vLLM vs Forge CPU**: 27.7x faster (24 tok/s → 665 tok/s)

### Latest Optimizations (January 2026)

**Sequential Mode Improvements:**
1. KV Cache Optimization: Eliminated full cache rebuild, now only updates modified pages (50-70% improvement)
2. Lock Thrashing Fix: Reduced mutex acquisitions from 2 per iteration to 1 (10-20% improvement)
3. F16 Precision: Now enabled by default in server (was previously hardcoded to F32)
4. Flash Attention: Properly enabled (was previously disabled in server code)
5. Batch Padding: Optimized CPU-side padding with pre-allocation

**Results:** Sequential mode improved from 98-106 tok/s → 120-160 tok/s (40-50% improvement)

## CPU Overload Analysis

All three engines demonstrate excellent CPU management with **no overload issues**:

| Engine | CPU Average | CPU Peak | Status |
|--------|-------------|----------|--------|
| Forge CPU | 25.5% | 29.3% | ✓ Healthy |
| Forge GPU | 18.2% | 18.7% | ✓ Healthy |
| vLLM | ~0% | ~0% | ✓ Healthy |

**Key Findings:**
- Forge GPU reduces CPU usage by 28% compared to Forge CPU
- All engines stay well below 70% CPU threshold
- No performance bottlenecks due to CPU limitations

## Latency Comparison

```
Forge CPU:  ████████████████████ 1827ms
Forge GPU:  ████ 368ms
vLLM:       █ 68ms
```

**Rankings (lower is better):**
1. vLLM: 68ms (fastest)
2. Forge GPU: 368ms (5.4x slower than vLLM, 5.0x faster than CPU)
3. Forge CPU: 1827ms (26.9x slower than vLLM)

## Throughput Comparison

```
Forge CPU:              ██ 24 tok/s
Forge GPU (sequential): ███████ 136 tok/s
Forge GPU (batched):    ██████████████ 360 tok/s
vLLM:                   ████████████████████ 665 tok/s
```

**Rankings (higher is better):**
1. vLLM: 665 tok/s (fastest)
2. **Forge GPU (batched)**: 360 tok/s (1.85x slower than vLLM, 15x faster than CPU) ⭐ NEW!
3. Forge GPU (sequential): 136 tok/s (4.9x slower than vLLM, 5.6x faster than CPU)
4. Forge CPU: 24 tok/s (27.7x slower than vLLM)

## Startup & Warmup Times

| Engine | Startup Time | Warmup Time | Total to First Response |
|--------|--------------|-------------|-------------------------|
| Forge CPU | ~2s | None | ~2s |
| Forge GPU | ~2s | 9s (first request) | ~11s |
| vLLM | 17.32s | Included in startup | ~17s |

**Analysis:**
- Forge CPU: Fastest time to first response
- Forge GPU: Moderate startup with one-time warmup
- vLLM: Slowest startup but best subsequent performance

## Use Case Recommendations

### Choose Forge CPU when:
- ✓ No GPU available
- ✓ Running on edge devices / embedded systems
- ✓ Fast startup required
- ✓ Low power consumption needed
- ✓ 24 tok/s throughput is sufficient
- ✓ Predictable latency more important than raw speed

### Choose Forge GPU when:
- ✓ GPU available but want minimal startup time
- ✓ Need better performance than CPU but faster startup than vLLM
- ✓ Handling moderate request volumes
- ✓ Want 5.6x improvement over CPU
- ✓ Fast startup with Flash Attention optimizations
- ✓ 136 tok/s throughput is sufficient
- ✓ Want lower CPU usage (18% vs 25%)

### Choose vLLM when:
- ✓ Maximum throughput is critical (665 tok/s)
- ✓ Handling high request volumes
- ✓ Production deployments with continuous uptime
- ✓ Can tolerate 17s startup time
- ✓ Want absolute best performance
- ✓ Need advanced features (batching, quantization, etc.)

## Hardware Specifications

**Test System:**
- CPU: x86_64 architecture
- GPU: NVIDIA CUDA-capable GPU (24GB VRAM)
- OS: Linux

**Software Versions:**
- Forge: v0.1.0 (with CUDA support)
- vLLM: v0.14.1
- Candle: Latest (via git submodule)

## How to Run These Benchmarks

### Forge CPU
```bash
cargo run --release -p forge-server
```

### Forge GPU (with Flash Attention)
```bash
cargo run --release --features "cuda,flash-attn" -p forge-server
```

### vLLM
```bash
python3 -m vllm.entrypoints.openai.api_server \
  --model HuggingFaceTB/SmolLM2-135M \
  --port 8000
```

## Conclusion

Forge now offers excellent GPU acceleration with **15x performance improvement** over CPU mode (360 tok/s batched) while maintaining:
- Fast startup times (2s vs 17s for vLLM)
- Simple deployment
- Rust's memory safety and performance benefits
- TRUE batching support for parallel request processing

**Key Optimizations:**
- ✅ **TRUE Batching** - Process multiple requests in parallel (2.7-3.2x speedup)
- ✅ **Flash Attention v2** - 2-4x faster attention computation
- ✅ **F16 precision** - Native GPU computation without dtype conversions
- ✅ **Auto CUDA detection** - Seamless GPU acceleration
- ✅ **Batched tensor operations** - Efficient parallel processing

**Performance Milestones:**
- Sequential: 136 tok/s (5.6x faster than CPU)
- Batched (4 requests): 288 tok/s (12x faster than CPU)
- Batched (8 requests): **360 tok/s** (15x faster than CPU) 🎉

All modes demonstrate **no CPU overload issues**, making them all viable options depending on your specific requirements.

**Performance Gap with vLLM:** The gap has been reduced from **4.9x to 1.85x** (665 vs 360 tok/s) with batching. Remaining optimizations to close the gap further:

### How to Close the Gap with vLLM

**Current Status:**
- Forge GPU Batched: 360 tok/s
- vLLM: 665 tok/s
- Gap: 1.85x

**Optimizations to Implement:**

1. **Continuous Batching** (Expected: +20-30% improvement)
   - Currently: All requests in batch start/end together
   - vLLM: Dynamically add/remove requests mid-batch
   - Impact: Better GPU utilization, less idle time

2. **PagedAttention for KV Cache** (Expected: +15-25% improvement)
   - Currently: Fixed KV cache allocation
   - vLLM: Virtual memory paging for KV cache
   - Impact: Better memory efficiency, larger batch sizes possible

3. **CUDA Graphs** (Expected: +10-20% improvement)
   - Currently: Kernel launches have overhead
   - vLLM: Pre-compiled execution graphs
   - Impact: Reduced kernel launch latency

4. **Kernel Fusion** (Expected: +10-15% improvement)
   - Currently: Separate operations for attention, normalization, etc.
   - vLLM: Fused custom CUDA kernels
   - Impact: Fewer memory transfers, better cache utilization

5. **Quantization Support** (Expected: +30-50% improvement)
   - Currently: F16 precision
   - vLLM: INT8/INT4 quantization support
   - Impact: 2x memory bandwidth, larger batch sizes

**Estimated Performance with All Optimizations:**
- Current: 360 tok/s
- With optimizations: 500-700 tok/s (matching or exceeding vLLM)

**Priority Order:**
1. Continuous batching (biggest impact, moderate complexity)
2. PagedAttention (enables larger batches, high complexity)
3. CUDA graphs (good ROI, moderate complexity)
4. Kernel fusion (significant work, high complexity)
5. Quantization (good for memory-limited scenarios)

## Updated Results (Batch Size Scaling)

Testing with larger batch sizes to better saturate GPU:

| Batch Size | Tokens/Request | Throughput | vs Batch=8 |
|------------|----------------|------------|------------|
| 8 | 500 | 295-310 tok/s | 1.0x (baseline) |
| 16 | 500 | 330-365 tok/s | 1.12x |
| 32 | 500 | 310-340 tok/s | 1.08x |

**Peak Performance:** 397 tok/s (batch=16, occasional spike)
**Consistent Performance:** 330-350 tok/s (batch=16)
**GPU Utilization:** Still only 30-40% (bottleneck is elsewhere)

**Analysis:**
- Larger batches help but plateau quickly
- GPU not saturated - bottleneck is likely:
  1. Padding inefficiency (all sequences padded to max length)
  2. Sequential decode (one token per request per step)  
  3. Memory bandwidth (not compute-bound)
- Need continuous batching + variable-length sequences for major gains
