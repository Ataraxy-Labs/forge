//! Benchmark: Compare inference performance across configurations
//!
//! Run with: cargo run --release --example benchmark

use anyhow::Result;
use forge_core::{InferenceEngine, EngineConfig, ModelConfig, SamplingParams};
use forge_core::{BatchInferenceEngine, BatchEngineConfig};
use std::time::Instant;

fn benchmark_single_request(prompt: &str, max_tokens: usize, num_runs: usize) -> Result<(f64, f64)> {
    let config = EngineConfig {
        model_config: ModelConfig::smollm_135m(),
        max_seq_len: 2048,
        repeat_penalty: 1.1,
        repeat_last_n: 64,
    };

    let engine = InferenceEngine::new(config, candle_core::Device::Cpu)?;

    let params = SamplingParams {
        max_tokens,
        temperature: 0.8,
        top_p: Some(0.9),
        top_k: None,
        stop_tokens: vec![],
        seed: Some(42),
    };

    // Warmup
    let _ = engine.generate(prompt, &params)?;

    // Benchmark
    let mut total_tokens = 0;
    let mut total_time = 0.0;

    for _ in 0..num_runs {
        let start = Instant::now();
        let result = engine.generate(prompt, &params)?;
        let elapsed = start.elapsed().as_secs_f64();

        total_tokens += result.tokens.len();
        total_time += elapsed;
    }

    let avg_throughput = total_tokens as f64 / total_time;
    let avg_latency = total_time / num_runs as f64;

    Ok((avg_throughput, avg_latency))
}

fn benchmark_batched(prompts: &[&str], max_tokens: usize) -> Result<(f64, f64)> {
    let config = BatchEngineConfig {
        model_config: ModelConfig::smollm_135m(),
        num_kv_pages: 128,
        repeat_penalty: 1.1,
        repeat_last_n: 64,
        ..Default::default()
    };

    let engine = BatchInferenceEngine::new(config, candle_core::Device::Cpu)?;

    let params = SamplingParams {
        max_tokens,
        temperature: 0.8,
        top_p: Some(0.9),
        top_k: None,
        stop_tokens: vec![],
        seed: Some(42),
    };

    // Submit all requests
    for prompt in prompts {
        engine.submit_request(prompt, params.clone())?;
    }

    // Process all requests
    let start = Instant::now();
    let mut total_tokens = 0;

    loop {
        let results = engine.step()?;
        if results.is_empty() {
            let stats = engine.scheduler_stats();
            if stats.pending_requests == 0 && stats.running_requests == 0 {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(1));
            continue;
        }
        total_tokens += results.len();
    }

    let elapsed = start.elapsed().as_secs_f64();
    let throughput = total_tokens as f64 / elapsed;

    Ok((throughput, elapsed))
}

fn main() -> Result<()> {
    println!("Forge Performance Benchmark");
    println!("============================\n");

    let prompt = "The future of artificial intelligence is";
    let max_tokens = 50;
    let num_runs = 3;

    // Single request benchmark
    println!("1. Single Request Performance (SmolLM-135M, CPU)");
    println!("   Prompt: \"{}...\"", &prompt[..30.min(prompt.len())]);
    println!("   Max tokens: {}", max_tokens);
    println!("   Runs: {}", num_runs);

    let (throughput, latency) = benchmark_single_request(prompt, max_tokens, num_runs)?;
    println!("   ----------------------------------------");
    println!("   Throughput: {:.2} tok/s", throughput);
    println!("   Avg latency: {:.2}ms per request", latency * 1000.0);
    println!();

    // Batched benchmark
    let batch_prompts = vec![
        "The meaning of life is",
        "Python is a programming language that",
        "Machine learning algorithms",
        "The best way to learn",
        "Artificial intelligence will",
        "The future of technology is",
        "Deep learning models are",
        "Natural language processing enables",
    ];

    println!("2. Batched Inference Performance (SmolLM-135M, CPU)");
    println!("   Requests: {}", batch_prompts.len());
    println!("   Max tokens per request: {}", max_tokens);

    let (batch_throughput, batch_time) = benchmark_batched(&batch_prompts, max_tokens)?;
    println!("   ----------------------------------------");
    println!("   Total throughput: {:.2} tok/s", batch_throughput);
    println!("   Total time: {:.2}s for {} requests", batch_time, batch_prompts.len());
    println!("   Effective per-request time: {:.2}ms", (batch_time / batch_prompts.len() as f64) * 1000.0);
    println!();

    // Summary
    println!("============================");
    println!("Summary:");
    println!("============================");
    println!("  Single request: {:.2} tok/s", throughput);
    println!("  Batched ({}x): {:.2} tok/s", batch_prompts.len(), batch_throughput);
    println!();

    // Comparison with vLLM (rough estimates from benchmarks)
    println!("Comparison (CPU mode):");
    println!("  Forge single:  {:.0} tok/s", throughput);
    println!("  Forge batched: {:.0} tok/s", batch_throughput);
    println!("  vLLM (est.):   ~50-100 tok/s (similar model, CPU)");
    println!();
    println!("Note: GPU mode with CUDA can achieve 5-10x higher throughput.");

    Ok(())
}
