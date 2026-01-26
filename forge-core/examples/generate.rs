//! Example: Direct text generation using Forge
//!
//! Supports multiple model architectures: LLaMA, Qwen2, Qwen3, Mistral, SmolLM
//!
//! Run with:
//!   cargo run --example generate -- --prompt "Hello, my name is"
//!   cargo run --example generate -- --model Qwen/Qwen2-0.5B --prompt "Hello"
//!   cargo run --example generate -- --model Qwen/Qwen3-0.6B --prompt "Hello"

use anyhow::Result;
use forge_core::{InferenceEngine, SamplingParams, EngineConfig, ModelConfig, ModelArch};
use std::io::Write;

fn main() -> Result<()> {
    // Parse args
    let args: Vec<String> = std::env::args().collect();
    let prompt = args.iter()
        .position(|a| a == "--prompt")
        .and_then(|i| args.get(i + 1))
        .map(|s| s.as_str())
        .unwrap_or("The best programming language is");

    let max_tokens = args.iter()
        .position(|a| a == "--max-tokens")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(50);

    let model_id = args.iter()
        .position(|a| a == "--model")
        .and_then(|i| args.get(i + 1))
        .map(|s| s.as_str())
        .unwrap_or("HuggingFaceTB/SmolLM2-135M");

    // Detect architecture
    let arch = ModelArch::from_model_id(model_id);

    println!("Forge - LLM Inference Engine");
    println!("Model: {}", model_id);
    println!("Architecture: {:?}", arch);
    println!("Prompt: {}", prompt);
    println!("Max tokens: {}", max_tokens);
    println!();

    // Create engine - use F16 for GPU (required for Flash Attention), F32 for CPU
    println!("Loading model...");
    let use_cuda = cfg!(feature = "cuda") && candle_core::Device::cuda_if_available(0).is_ok();
    let dtype = if use_cuda {
        candle_core::DType::F16  // F16 required for Flash Attention on GPU
    } else {
        candle_core::DType::F32  // F32 for CPU
    };

    let model_config = ModelConfig {
        model_id: model_id.to_string(),
        revision: "main".to_string(),
        dtype,
        use_flash_attn: true,
    };
    let engine_config = EngineConfig::with_model(model_config);

    // Auto-detect device: CUDA if available, otherwise CPU
    let device = if cfg!(feature = "cuda") && candle_core::Device::cuda_if_available(0).is_ok() {
        candle_core::Device::cuda_if_available(0)?
    } else {
        candle_core::Device::Cpu
    };

    let engine = InferenceEngine::new(engine_config, device)?;

    let stats = engine.stats();
    println!("Model loaded!");
    println!("  Vocab size: {}", stats.vocab_size);
    println!("  Hidden size: {}", stats.hidden_size);
    println!("  Layers: {}", stats.num_layers);
    println!();

    // Generate with streaming
    println!("Generating...");
    println!("---");
    print!("{}", prompt);
    std::io::stdout().flush()?;

    let params = SamplingParams {
        max_tokens,
        temperature: 0.8,
        top_p: Some(0.9),
        top_k: None,
        stop_tokens: vec![],
        seed: Some(42),
    };

    let result = engine.generate_streaming(prompt, &params, |text| {
        print!("{}", text);
        std::io::stdout().flush().ok();
    })?;

    println!();
    println!("---");
    println!();
    println!("Stats:");
    println!("  Prompt tokens: {}", result.prompt_tokens);
    println!("  Generated tokens: {}", result.generated_tokens);
    println!("  Speed: {:.2} tok/s", result.tokens_per_second);
    println!("  Finish reason: {:?}", result.finish_reason);

    Ok(())
}
