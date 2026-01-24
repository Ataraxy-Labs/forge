# Forge

High-performance LLM inference engine built on Candle, inspired by Cloudflare's Infire.

## Quick Start

### Prerequisites

- Rust 1.70+
- For GPU: CUDA 12.0+ (Linux/Windows) or Metal (macOS)

### Installation

```bash
git clone https://github.com/Ataraxy-Labs/forge.git
cd forge

# CPU only
cargo build --release

# macOS with Metal
cargo build --release --features metal

# Linux/Windows with CUDA
cargo build --release --features cuda
```

### Run Tests

```bash
cargo test --release
```

### Start Server

```bash
# CPU
cargo run --release -p forge-server

# macOS Metal
cargo run --release -p forge-server --features metal

# CUDA
cargo run --release -p forge-server --features cuda
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
    "temperature": 0.7
  }'
```

## Project Structure

```
forge/
├── forge-core/      # Inference engine (KV cache, scheduler, batcher)
├── forge-server/    # HTTP API server
└── forge-kernels/   # GPU kernel optimizations
```

## Features

- Paged KV cache (16-token pages)
- Continuous batching
- Chunked prefill
- OpenAI-compatible API
- Multi-backend: CPU, CUDA, Metal

## Supported Models

- LLaMA 2/3
- Phi-3
- Mistral
- Qwen

## Performance Targets

| Metric | vLLM | Forge |
|--------|------|-------|
| CPU Usage | 250% | <30% |
| Binary Size | 2GB | 100MB |
| Cold Start | 20-30s | 5-10s |

## License

MIT OR Apache-2.0
