//! Batched benchmark to test GPU utilization with Flash Attention
//!
//! Run with: cargo run --release --features "cuda,flash-attn" --example batch_benchmark

use anyhow::Result;
use forge_core::{BatchInferenceEngine, BatchEngineConfig, ModelConfig, SamplingParams};
use std::time::Instant;

fn main() -> Result<()> {
    println!("Forge - Batched GPU Benchmark");
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

    let engine_config = BatchEngineConfig {
        model_config,
        num_kv_pages: 512,  // Increased for longer sequences
        max_seq_len: 8192,  // Support longer contexts
        repeat_penalty: 1.1,
        repeat_last_n: 64,
    };

    println!("Device: {:?}", device);
    println!("Flash Attention: {}", engine_config.model_config.use_flash_attn);
    println!("Dtype: {:?}", engine_config.model_config.dtype);
    println!("\nLoading model...");

    let engine = BatchInferenceEngine::new(engine_config, device)?;
    println!("Model loaded!\n");

    // Test different batch sizes and sequence lengths
    let test_configs = vec![
        ("Batch=4, Tokens=512", 4, 512),
        ("Batch=4, Tokens=1024", 4, 1024),
        ("Batch=8, Tokens=512", 8, 512),
        ("Batch=8, Tokens=1024", 8, 1024),
        ("Batch=4, Tokens=2048", 4, 2048),
    ];

    for (name, batch_size, max_tokens) in test_configs {
        println!("=== {} ===", name);

        // Create diverse prompts
        let base_prompts = vec![
            "The future of artificial intelligence",
            "Explain quantum computing in detail",
            "Write a story about space exploration",
            "Describe the history of computer science",
            "List the benefits of machine learning",
            "Analyze the impact of social media",
            "Discuss renewable energy sources",
            "Explain how neural networks work",
        ];

        let prompts: Vec<&str> = base_prompts.iter()
            .take(batch_size)
            .map(|s| *s)
            .collect();

        let params = SamplingParams {
            max_tokens,
            temperature: 0.8,
            top_p: Some(0.9),
            top_k: None,
            stop_tokens: vec![],
            seed: Some(42),
        };

        // Submit all requests
        let mut request_ids = Vec::new();
        for prompt in prompts.iter() {
            let id = engine.submit_request(prompt, params.clone())?;
            request_ids.push(id);
        }

        // Process with timing
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

        println!("  Total tokens: {}", total_tokens);
        println!("  Time: {:.2}s", elapsed);
        println!("  Throughput: {:.2} tok/s", throughput);
        println!();
    }

    Ok(())
}
