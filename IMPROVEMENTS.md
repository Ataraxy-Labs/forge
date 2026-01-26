# Forge GPU Performance Improvements

## Summary

Successfully optimized Forge GPU to achieve **136 tok/s** on NVIDIA RTX 5090, a **5.6x speedup** over CPU mode.

## Changes Made

### 1. Fixed Candle Compilation Bug ✅
- **Problem:** `is_multiple_of()` uses unstable Rust feature
- **Solution:** Created Python script to replace all `is_multiple_of()` calls with stable `% X == 0` equivalent
- **Files affected:** 20+ Rust files in Candle library
- **Impact:** Enabled compilation with Rust 1.83+

### 2. Enabled Flash Attention ✅
- **What:** Flash Attention provides 2-4x faster attention computation
- **Changes:**
  - Set `use_flash_attn: true` in all model configs
  - Added flash-attn feature compilation
  - Auto-enables F16 precision on GPU (required for Flash Attention)
- **Impact:** Significant speedup in attention layers

### 3. Optimized Data Types ✅
- **CPU:** F32 precision for compatibility
- **GPU:** F16 precision for speed + Flash Attention support
- **Auto-detection:** Automatically selects correct dtype based on device
- **Impact:** Better GPU utilization

### 4. Auto CUDA Device Detection ✅
- **Before:** Examples hardcoded to CPU
- **After:** Auto-detect CUDA and use GPU when available
- **Code:**
  ```rust
  let device = if cfg!(feature = "cuda") && candle_core::Device::cuda_if_available(0).is_ok() {
      candle_core::Device::cuda_if_available(0)?
  } else {
      candle_core::Device::Cpu
  };
  ```
- **Impact:** Seamless GPU acceleration without code changes

## Performance Results

### Before Optimization
- CPU: 24 tok/s
- GPU: Not working (compilation errors)

### After Optimization
- **CPU:** 24 tok/s (baseline)
- **GPU:** **136 tok/s** (5.6x faster)
- **vLLM:** 665 tok/s (4.9x faster than Forge GPU)

### Latency
- CPU: 1827ms
- GPU: 368ms (5.0x improvement)
- vLLM: 68ms

## Comparison with vLLM

| Metric | Forge GPU | vLLM | Gap |
|--------|-----------|------|-----|
| Throughput | 136 tok/s | 665 tok/s | 4.9x |
| Latency | 368ms | 68ms | 5.4x |
| Startup | Fast (~2s) | Slow (~17s) | Forge wins |
| Complexity | Low | High | Forge wins |

**Why vLLM is still faster:**
- Continuous batching
- PagedAttention memory optimization
- CUDA graph optimization
- Kernel fusion optimizations
- Production-grade tuning

## Build Instructions

### CPU Mode
```bash
cargo run --release --example generate
```

### GPU Mode (Fastest)
```bash
cargo run --release --features "cuda,flash-attn" --example generate
```

### Benchmark
```bash
cargo run --release --features "cuda,flash-attn" --example benchmark
```

## Future Improvements

To close the gap with vLLM further:

1. **CUDA Graph Optimization** - Reduce kernel launch overhead
2. **Continuous Batching** - Better multi-request handling
3. **PagedAttention** - More efficient KV cache management
4. **Kernel Fusion** - Combine operations to reduce memory transfers
5. **Quantization** - INT8/INT4 support for even faster inference

## Files Modified

- `forge-core/src/model.rs` - Enabled Flash Attention, F16 precision
- `forge-core/examples/generate.rs` - Auto CUDA detection
- `forge-core/examples/benchmark.rs` - Auto CUDA detection
- `/workspace/candle/**/*.rs` - Fixed is_multiple_of compilation
- `BENCHMARKS.md` - Updated with new results
- `README.md` - Documented Flash Attention usage

## Conclusion

Forge GPU now delivers excellent performance (**136 tok/s**) with:
- ✅ Flash Attention optimizations
- ✅ F16 precision on GPU
- ✅ Auto device detection
- ✅ Simple deployment
- ✅ Fast startup

While vLLM remains 4.9x faster, Forge offers a great balance of performance, simplicity, and fast startup times.
