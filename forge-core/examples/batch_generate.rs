//! Example: Batched text generation using Forge
//!
//! Demonstrates continuous batching with multiple concurrent requests.
//!
//! Run with: cargo run --release --example batch_generate

use anyhow::Result;
use forge_core::{BatchInferenceEngine, BatchEngineConfig, ModelConfig, SamplingParams};
use std::time::Instant;

fn main() -> Result<()> {
    println!("Forge - Batched LLM Inference Engine");
    println!("=====================================\n");

    // Auto-detect device: CUDA if available, otherwise CPU
    let device = if cfg!(feature = "cuda") {
        candle_core::Device::cuda_if_available(0).unwrap_or(candle_core::Device::Cpu)
    } else {
        candle_core::Device::Cpu
    };
    let is_gpu = matches!(device, candle_core::Device::Cuda(_));

    // Create engine config with proper flash attention setting
    let model_config = ModelConfig {
        model_id: "HuggingFaceTB/SmolLM2-135M".to_string(),
        revision: "main".to_string(),
        dtype: if is_gpu { candle_core::DType::F16 } else { candle_core::DType::F32 },
        use_flash_attn: is_gpu && cfg!(feature = "flash-attn"),
    };

    let engine_config = BatchEngineConfig {
        model_config,
        num_kv_pages: 128,
        repeat_penalty: 1.1,
        repeat_last_n: 64,
        ..Default::default()
    };

    println!("Device: {}", if is_gpu { "CUDA GPU" } else { "CPU" });
    println!("Flash Attention: {}\n", engine_config.model_config.use_flash_attn);
    println!("Loading model...");
    let engine = BatchInferenceEngine::new(engine_config, device)?;
    println!("Model loaded!\n");

    // Define prompts for batched generation
    let prompts = vec![
        "The meaning of life is",
        "Python is a programming language that",
        "Machine learning algorithms",
        "The best way to learn",
    ];

    let params = SamplingParams {
        max_tokens: 30,
        temperature: 0.8,
        top_p: Some(0.9),
        top_k: None,
        stop_tokens: vec![],
        seed: Some(42),
    };

    // Submit all requests
    println!("Submitting {} requests...", prompts.len());
    let mut request_ids = Vec::new();
    for (_i, prompt) in prompts.iter().enumerate() {
        let id = engine.submit_request(prompt, params.clone())?;
        request_ids.push((id, prompt.to_string()));
        println!("  Request {}: \"{}...\"", id, &prompt[..prompt.len().min(30)]);
    }
    println!();

    // Process all requests
    println!("Processing batched inference...");
    let start = Instant::now();

    let mut completed = vec![false; request_ids.len()];
    let mut generated_texts: Vec<Option<Vec<u32>>> = vec![None; request_ids.len()];

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

        for (req_id, token, is_finished) in results {
            // Find index in our list
            if let Some(idx) = request_ids.iter().position(|(id, _)| *id == req_id) {
                // Append token
                if generated_texts[idx].is_none() {
                    generated_texts[idx] = Some(Vec::new());
                }
                if let Some(ref mut tokens) = generated_texts[idx] {
                    tokens.push(token);
                }

                if is_finished {
                    completed[idx] = true;
                }
            }
        }
    }

    let elapsed = start.elapsed();
    let stats = engine.stats();

    // Print results
    println!("\n=====================================");
    println!("Results:");
    println!("=====================================\n");

    for ((id, prompt), tokens) in request_ids.iter().zip(generated_texts.iter()) {
        println!("Request {} (prompt: \"{}...\"):", id, &prompt[..prompt.len().min(20)]);
        if let Some(tokens) = tokens {
            let text = engine.decode(tokens)?;
            println!("  Generated: {}", text);
            println!("  Tokens: {}", tokens.len());
        }
        println!();
    }

    println!("=====================================");
    println!("Statistics:");
    println!("=====================================");
    println!("  Total requests: {}", stats.total_requests);
    println!("  Completed requests: {}", stats.completed_requests);
    println!("  Total tokens generated: {}", stats.total_tokens_generated);
    println!("  Total time: {:.2}s", elapsed.as_secs_f64());
    println!("  Throughput: {:.2} tok/s", stats.total_tokens_generated as f64 / elapsed.as_secs_f64());

    let cache_stats = engine.cache_stats();
    println!("\nKV Cache:");
    println!("  Total pages: {}", cache_stats.total_pages);
    println!("  Used pages: {}", cache_stats.used_pages);
    println!("  Free pages: {}", cache_stats.free_pages);

    Ok(())
}
