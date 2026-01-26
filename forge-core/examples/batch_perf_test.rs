//! Test batched vs sequential performance
//!
//! Run with: cargo run --release --features "cuda,flash-attn" --example batch_perf_test

use anyhow::Result;
use forge_core::{InferenceEngine, BatchInferenceEngine, EngineConfig, BatchEngineConfig, ModelConfig, SamplingParams};
use std::time::Instant;

fn main() -> Result<()> {
    println!("Forge - Batching Performance Test");
    println!("===================================\n");

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

    println!("Device: {:?}", device);
    println!("Flash Attention: {}\n", model_config.use_flash_attn);

    // Test 1: Sequential (baseline)
    println!("=== Test 1: Sequential Processing ===");
    let engine_config = EngineConfig::with_model(model_config.clone());
    let engine = InferenceEngine::new(engine_config, device.clone())?;

    let prompts = vec![
        "The future of AI",
        "Quantum computing",
        "Machine learning",
        "Deep neural networks",
    ];

    let params = SamplingParams {
        max_tokens: 100,
        temperature: 0.8,
        top_p: Some(0.9),
        top_k: None,
        stop_tokens: vec![],
        seed: Some(42),
    };

    let start = Instant::now();
    let mut total_tokens = 0;

    for prompt in &prompts {
        let result = engine.generate(prompt, &params)?;
        total_tokens += result.generated_tokens;
    }

    let elapsed = start.elapsed().as_secs_f64();
    let seq_throughput = total_tokens as f64 / elapsed;

    println!("  Prompts: {}", prompts.len());
    println!("  Total tokens: {}", total_tokens);
    println!("  Time: {:.2}s", elapsed);
    println!("  Throughput: {:.2} tok/s", seq_throughput);
    println!();

    // Test 2: Batched (with TRUE batching)
    println!("=== Test 2: Batched Processing (TRUE BATCHING) ===");
    let batch_config = BatchEngineConfig {
        model_config: model_config.clone(),
        num_kv_pages: 256,
        max_seq_len: 4096,
        repeat_penalty: 1.1,
        repeat_last_n: 64,
    };

    let batch_engine = BatchInferenceEngine::new(batch_config, device.clone())?;

    // Submit all requests
    for prompt in &prompts {
        batch_engine.submit_request(prompt, params.clone())?;
    }

    // Process with timing
    let start = Instant::now();
    let mut batch_total_tokens = 0;

    loop {
        let results = batch_engine.step()?;

        if results.is_empty() {
            let stats = batch_engine.scheduler_stats();
            if stats.pending_requests == 0 && stats.running_requests == 0 {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(1));
            continue;
        }

        batch_total_tokens += results.len();
    }

    let elapsed = start.elapsed().as_secs_f64();
    let batch_throughput = batch_total_tokens as f64 / elapsed;

    println!("  Prompts: {}", prompts.len());
    println!("  Total tokens: {}", batch_total_tokens);
    println!("  Time: {:.2}s", elapsed);
    println!("  Throughput: {:.2} tok/s", batch_throughput);
    println!();

    // Summary
    println!("=== RESULTS ===");
    println!("Sequential: {:.2} tok/s", seq_throughput);
    println!("Batched:    {:.2} tok/s", batch_throughput);
    println!("Speedup:    {:.2}x", batch_throughput / seq_throughput);

    Ok(())
}
