//! Test batching scalability with different batch sizes
//!
//! Run with: cargo run --release --features "cuda,flash-attn" --example batch_scale_test

use anyhow::Result;
use forge_core::{BatchInferenceEngine, BatchEngineConfig, ModelConfig, SamplingParams};
use std::time::Instant;

fn main() -> Result<()> {
    println!("Forge - Batch Scalability Test");
    println!("================================\n");

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

    let batch_config = BatchEngineConfig {
        model_config,
        num_kv_pages: 512,
        max_seq_len: 8192,
        repeat_penalty: 1.1,
        repeat_last_n: 64,
    };

    println!("Device: {:?}", device);
    println!("Flash Attention: {}\n", batch_config.model_config.use_flash_attn);

    let engine = BatchInferenceEngine::new(batch_config, device)?;

    let base_prompts = vec![
        "The future of artificial intelligence",
        "Explain quantum computing",
        "Machine learning techniques",
        "Deep neural network architectures",
        "Natural language processing",
        "Computer vision applications",
        "Reinforcement learning algorithms",
        "Generative AI models",
    ];

    // Test different batch sizes with different token lengths
    let test_configs = vec![
        ("Batch=2, Tokens=100", 2, 100),
        ("Batch=4, Tokens=100", 4, 100),
        ("Batch=8, Tokens=100", 8, 100),
        ("Batch=4, Tokens=500", 4, 500),
        ("Batch=8, Tokens=500", 8, 500),
        ("Batch=4, Tokens=1000", 4, 1000),
    ];

    for (name, batch_size, max_tokens) in test_configs {
        println!("=== {} ===", name);

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

        // Submit requests
        for prompt in &prompts {
            engine.submit_request(prompt, params.clone())?;
        }

        // Process
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
