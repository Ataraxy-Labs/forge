# Forge

High-performance LLM inference engine in Rust, built on Candle. Inspired by Cloudflare's Infire.

## Features

- **Multi-architecture support**: LLaMA, Qwen2, Qwen3, Mistral, SmolLM
- Real model inference using HuggingFace transformers
- Automatic model download from HuggingFace Hub
- **Continuous batching** for high-throughput multi-request processing
- KV caching for efficient token generation
- Temperature, top-p, top-k sampling
- Streaming text generation
- OpenAI-compatible HTTP API
- Multi-backend: CPU, Metal (macOS), CUDA

## Quick Start

### Prerequisites

- **Rust 1.83+** (required for `icu_normalizer_data` dependency)
- HuggingFace account (for some models)

#### Installing Rust 1.83+

If you have an older Rust version, upgrade using rustup:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain 1.83.0
source "$HOME/.cargo/env"
```

### Build

```bash
# Clone candle (required dependency)
git clone https://github.com/huggingface/candle.git

# Fix candle for Rust 1.83 compatibility (replace unstable is_multiple_of)
find candle -name "*.rs" -exec sed -i 's/\.is_multiple_of(\([^)]*\))/% \1 == 0/g' {} \;

# Clone forge
git clone https://github.com/Ataraxy-Labs/forge.git
cd forge

# Build (CPU)
cargo build --release

# Build with Metal (macOS)
cargo build --release --features metal

# Build with CUDA
cargo build --release --features cuda
```

**Note:** The candle dependency uses the unstable `is_multiple_of()` feature which requires nightly Rust. The sed command above replaces it with the stable `% X == 0` equivalent for compatibility with Rust 1.83.

### Run CLI

```bash
# Generate text with SmolLM-135M (default, smallest/fastest)
cargo run --release --example generate -- \
  --prompt "The best programming language is" \
  --max-tokens 50

# Use Qwen2 model
cargo run --release --example generate -- \
  --model "Qwen/Qwen2-0.5B" \
  --prompt "Hello, I am" \
  --max-tokens 30

# Use Qwen3 model
cargo run --release --example generate -- \
  --model "Qwen/Qwen3-0.6B" \
  --prompt "The meaning of life is" \
  --max-tokens 30

# Use TinyLlama
cargo run --release --example generate -- \
  --model "TinyLlama/TinyLlama-1.1B-Chat-v1.0" \
  --prompt "Hello, my name is" \
  --max-tokens 30

# Run batched inference (multiple concurrent requests)
cargo run --release --example batch_generate

# Run performance benchmark
cargo run --release --example benchmark
```

### Run Server

```bash
# Start with default model (SmolLM-135M) on CPU
cargo run --release -p forge-server

# Start with CUDA GPU acceleration (auto-detects GPU)
cargo run --release --features cuda -p forge-server

# Start with a specific model
FORGE_MODEL="TinyLlama/TinyLlama-1.1B-Chat-v1.0" cargo run --release --features cuda -p forge-server

# Force specific device (cpu, cuda, or metal)
FORGE_DEVICE=cuda cargo run --release --features cuda -p forge-server
FORGE_DEVICE=cpu cargo run --release -p forge-server

# Change port
FORGE_PORT=3000 cargo run --release --features cuda -p forge-server
```

### API Usage

```bash
# Health check
curl http://localhost:8080/health

# Generate text
curl -X POST http://localhost:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Hello, world!",
    "max_tokens": 50,
    "temperature": 0.7,
    "top_p": 0.9
  }'

# List models
curl http://localhost:8080/v1/models
```

## Supported Models

### LLaMA Family
- HuggingFaceTB/SmolLM2-135M (default)
- HuggingFaceTB/SmolLM2-360M
- HuggingFaceTB/SmolLM2-1.7B
- TinyLlama/TinyLlama-1.1B-Chat-v1.0
- meta-llama/Llama-3.2-1B (requires HF token)
- meta-llama/Llama-3.2-3B (requires HF token)
- Mistral models

### Qwen Family
- Qwen/Qwen2-0.5B
- Qwen/Qwen2-1.5B
- Qwen/Qwen2-7B
- Qwen/Qwen3-0.6B
- Qwen/Qwen3-1.7B
- Qwen/Qwen3-4B

Most LLaMA, Qwen2, and Qwen3 architecture models on HuggingFace should work.

## Project Structure

```
forge/
├── forge-core/           # Inference engine
│   ├── src/
│   │   ├── model.rs          # Model loading (LLaMA, Qwen2, Qwen3)
│   │   ├── engine.rs         # Single-request inference
│   │   ├── batch_engine.rs   # Batched inference with scheduling
│   │   ├── kv_cache.rs       # Paged KV cache
│   │   └── scheduler.rs      # Request scheduling
│   └── examples/
│       ├── generate.rs       # Single-request CLI
│       ├── batch_generate.rs # Batched inference demo
│       └── benchmark.rs      # Performance benchmarks
├── forge-server/         # HTTP API server
└── forge-kernels/        # GPU kernel optimizations
```

## Configuration

Environment variables:
- `FORGE_MODEL` - HuggingFace model ID (default: SmolLM2-135M)
- `FORGE_PORT` - Server port (default: 8080)
- `FORGE_DEVICE` - Device to use: `cpu`, `cuda`, or `metal` (default: auto-detect)
- `HF_HOME` - HuggingFace cache directory
- `HF_TOKEN` - HuggingFace API token (for gated models)

## Performance

### Single Request (CPU Mode)
| Model | Throughput | Notes |
|-------|-----------|-------|
| SmolLM-135M | ~47 tok/s | Apple M-series |
| Qwen2-0.5B | ~30 tok/s | Apple M-series |
| TinyLlama-1.1B | ~8 tok/s | x86_64 |

### Batched Inference (CPU Mode)
| Batch Size | Throughput | Per-Request Latency |
|------------|------------|---------------------|
| 8 requests | ~53 tok/s | ~950ms |

### GPU Mode (CUDA)
| Model | Throughput | vs CPU |
|-------|-----------|--------|
| SmolLM-135M | ~141 tok/s | 5.9x faster |
| TinyLlama-1.1B | ~40 tok/s | estimated |

**GPU Mode (Metal on Apple Silicon):** Expect 5-10x improvement over CPU.

**Note:** First GPU request includes CUDA graph compilation (~9s warmup), subsequent requests are fast.

Run `cargo run --release --example benchmark` to measure performance on your hardware.

## License

MIT OR Apache-2.0
