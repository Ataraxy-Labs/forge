# Forge

High-performance LLM inference engine in Rust, built on Candle. Inspired by Cloudflare's Infire.

## Features

- Real model inference using HuggingFace transformers
- Automatic model download from HuggingFace Hub
- KV caching for efficient token generation
- Temperature, top-p, top-k sampling
- Streaming text generation
- OpenAI-compatible HTTP API
- Multi-backend: CPU, Metal (macOS), CUDA

## Quick Start

### Prerequisites

- Rust 1.70+
- HuggingFace account (for some models)

### Build

```bash
# Clone candle (required dependency)
git clone https://github.com/huggingface/candle.git

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

### Run CLI

```bash
# Generate text with SmolLM-135M (default, smallest/fastest)
cargo run --release --example generate -- \
  --prompt "The best programming language is" \
  --max-tokens 50

# Use a different model
cargo run --release --example generate -- \
  --model "TinyLlama/TinyLlama-1.1B-Chat-v1.0" \
  --prompt "Hello, my name is" \
  --max-tokens 30
```

### Run Server

```bash
# Start with default model (SmolLM-135M)
cargo run --release -p forge-server

# Start with a specific model
FORGE_MODEL="TinyLlama/TinyLlama-1.1B-Chat-v1.0" cargo run --release -p forge-server

# Change port
FORGE_PORT=3000 cargo run --release -p forge-server
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

Tested and working:
- HuggingFaceTB/SmolLM2-135M (default)
- HuggingFaceTB/SmolLM2-360M
- HuggingFaceTB/SmolLM2-1.7B
- TinyLlama/TinyLlama-1.1B-Chat-v1.0
- meta-llama/Llama-3.2-1B (requires HF token)
- meta-llama/Llama-3.2-3B (requires HF token)

Any LLaMA-architecture model on HuggingFace should work.

## Project Structure

```
forge/
├── forge-core/       # Inference engine
│   ├── src/
│   │   ├── model.rs      # Model loading from HuggingFace
│   │   ├── engine.rs     # Inference orchestration
│   │   ├── kv_cache.rs   # Paged KV cache
│   │   └── scheduler.rs  # Request scheduling
│   └── examples/
│       └── generate.rs   # CLI example
├── forge-server/     # HTTP API server
└── forge-kernels/    # GPU kernel optimizations
```

## Configuration

Environment variables:
- `FORGE_MODEL` - HuggingFace model ID (default: SmolLM2-135M)
- `FORGE_PORT` - Server port (default: 8080)
- `HF_HOME` - HuggingFace cache directory
- `HF_TOKEN` - HuggingFace API token (for gated models)

## Performance

On Apple M1 Pro (CPU):
- SmolLM-135M: ~1.0 tok/s
- TinyLlama-1.1B: ~0.3 tok/s

With Metal acceleration expect 5-10x improvement.

## License

MIT OR Apache-2.0
