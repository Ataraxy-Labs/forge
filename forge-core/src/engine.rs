//! Inference engine - orchestrates forward passes

use anyhow::{Result, anyhow};
use candle_core::{Device, DType, Tensor};
use std::sync::Arc;
use parking_lot::Mutex;

use crate::batcher::Sampler;
use crate::request::{SamplingParams, InferenceRequest, InferenceResponse, FinishReason};
use crate::kv_cache::PagedKVCache;
use crate::scheduler::Scheduler;

/// Configuration for the inference engine
#[derive(Debug, Clone)]
pub struct EngineConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub max_seq_len: usize,
    pub dtype: DType,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            vocab_size: 32000,
            hidden_size: 4096,
            num_layers: 32,
            num_heads: 32,
            head_dim: 128,
            max_seq_len: 4096,
            dtype: DType::F32,
        }
    }
}

/// Inference engine that orchestrates model execution
pub struct InferenceEngine {
    config: EngineConfig,
    device: Device,
    sampler: Mutex<Sampler>,
    kv_cache: Arc<Mutex<PagedKVCache>>,
    scheduler: Arc<Scheduler>,
}

impl InferenceEngine {
    /// Create a new inference engine
    pub fn new(config: EngineConfig, device: Device) -> Result<Self> {
        let kv_cache = Arc::new(Mutex::new(PagedKVCache::new(
            config.num_layers,
            config.num_heads,
            config.head_dim,
            1024, // total pages
            config.dtype,
            device.clone(),
        )?));

        let scheduler = Arc::new(Scheduler::new(kv_cache.clone()));

        Ok(Self {
            config,
            device,
            sampler: Mutex::new(Sampler::new(42)),
            kv_cache,
            scheduler,
        })
    }

    /// Create with default config for CPU
    pub fn new_cpu() -> Result<Self> {
        Self::new(EngineConfig::default(), Device::Cpu)
    }

    /// Create with Metal backend (macOS)
    #[cfg(feature = "metal")]
    pub fn new_metal() -> Result<Self> {
        let device = Device::new_metal(0)?;
        Self::new(EngineConfig::default(), device)
    }

    /// Create with CUDA backend
    #[cfg(feature = "cuda")]
    pub fn new_cuda(device_id: usize) -> Result<Self> {
        let device = Device::new_cuda(device_id)?;
        Self::new(EngineConfig::default(), device)
    }

    /// Get the device being used
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Generate text from a prompt (simplified for demo)
    pub fn generate(&self, tokens: &[u32], params: &SamplingParams) -> Result<Vec<u32>> {
        let mut output_tokens = Vec::new();
        let mut input_tokens = tokens.to_vec();

        for _ in 0..params.max_tokens {
            // Create mock logits (in real impl, this would be model forward pass)
            let logits = self.mock_forward(&input_tokens)?;

            // Sample next token
            let next_token = self.sampler.lock().sample(&logits, params)?;

            // Check for stop token
            if params.stop_tokens.contains(&next_token) {
                break;
            }

            output_tokens.push(next_token);
            input_tokens.push(next_token);
        }

        Ok(output_tokens)
    }

    /// Mock forward pass - returns random logits
    /// Replace with actual model inference
    fn mock_forward(&self, _tokens: &[u32]) -> Result<Tensor> {
        // Generate random logits for demonstration
        let logits = Tensor::randn(
            0.0f32,
            1.0,
            (1, self.config.vocab_size),
            &self.device,
        )?;
        Ok(logits)
    }

    /// Get engine statistics
    pub fn stats(&self) -> EngineStats {
        let cache_stats = self.kv_cache.lock().stats();
        let scheduler_stats = self.scheduler.stats();

        EngineStats {
            device: format!("{:?}", self.device),
            vocab_size: self.config.vocab_size,
            num_layers: self.config.num_layers,
            cache_pages_total: cache_stats.total_pages,
            cache_pages_used: cache_stats.used_pages,
            pending_requests: scheduler_stats.pending_requests,
            running_requests: scheduler_stats.running_requests,
        }
    }
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct EngineStats {
    pub device: String,
    pub vocab_size: usize,
    pub num_layers: usize,
    pub cache_pages_total: usize,
    pub cache_pages_used: usize,
    pub pending_requests: usize,
    pub running_requests: usize,
}
