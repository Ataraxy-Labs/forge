//! Simple batch test: Compare single vs batched generation
//!
//! Run with: cargo run --release --features "cuda,flash-attn" --example simple_batch_test

use anyhow::Result;
use forge_core::{InferenceEngine, EngineConfig, ModelConfig, SamplingParams};
use std::time::Instant;

fn main() -> Result<()> {
    println!("Forge - Batch Size Comparison");
    println!("==============================\n");

    // Auto-detect device
    let device = if cfg!(feature = "cuda") && candle_core::Device::cuda_if_available(0).is_ok() {
        candle_core::Device::cuda_if_available(0)?
    } else {
        candle_core::Device::Cpu
    };
    let is_gpu = matches!(device, candle_core::Device::Cuda(_));

    let model_config = ModelConfig {
        model_id: "HuggingFaceTB/SmolLM2-135M".to_string(),
        revision: "main".to_string(),
        dtype: if is_gpu { candle_core::DType::F16 } else { candle_core::DType::F32 },
        use_flash_attn: is_gpu && cfg!(feature = "flash-attn"),
    };

    let engine_config = EngineConfig::with_model(model_config);

    println!("Device: {:?}", device);
    println!("Flash Attention: {}", engine_config.model_config.use_flash_attn);
    println!("Loading model...\n");

    let engine = InferenceEngine::new(engine_config, device)?;

    let prompts = vec![
        "The future of AI",
        "Quantum computing is",
        "Machine learning helps",
        "Python programming",
    ];

    let params = SamplingParams {
        max_tokens: 100,
        temperature: 0.8,
        top_p: Some(0.9),
        top_k: None,
        stop_tokens: vec![],
        seed: Some(42),
    };

    // Test 1: Sequential generation (effective batch=1)
    println!("=== Test 1: Sequential (4 prompts one-by-one) ===");
    let start = Instant::now();
    let mut total_tokens = 0;

    for prompt in &prompts {
        let result = engine.generate(prompt, &params)?;
        total_tokens += result.generated_tokens;
    }

    let elapsed = start.elapsed().as_secs_f64();
    let throughput = total_tokens as f64 / elapsed;

    println!("  Total tokens: {}", total_tokens);
    println!("  Time: {:.2}s", elapsed);
    println!("  Throughput: {:.2} tok/s", throughput);
    println!();

    // Test 2: Longer sequences
    println!("=== Test 2: Single long generation (500 tokens) ===");
    let params_long = SamplingParams {
        max_tokens: 500,
        temperature: 0.8,
        top_p: Some(0.9),
        top_k: None,
        stop_tokens: vec![],
        seed: Some(42),
    };

    let result = engine.generate("The future of artificial intelligence", &params_long)?;
    let throughput_long = result.tokens_per_second;

    println!("  Generated tokens: {}", result.generated_tokens);
    println!("  Throughput: {:.2} tok/s", throughput_long);
    println!();

    // Test 3: Very long sequence
    println!("=== Test 3: Very long generation (2000 tokens) ===");
    let params_vlong = SamplingParams {
        max_tokens: 2000,
        temperature: 0.8,
        top_p: Some(0.9),
        top_k: None,
        stop_tokens: vec![],
        seed: Some(42),
    };

    let result = engine.generate("Write a detailed explanation", &params_vlong)?;
    let throughput_vlong = result.tokens_per_second;

    println!("  Generated tokens: {}", result.generated_tokens);
    println!("  Throughput: {:.2} tok/s", throughput_vlong);
    println!();

    println!("=== Summary ===");
    println!("Sequential (4x100): {:.2} tok/s", throughput);
    println!("Single 500 tokens: {:.2} tok/s", throughput_long);
    println!("Single 2000 tokens: {:.2} tok/s", throughput_vlong);

    Ok(())
}
