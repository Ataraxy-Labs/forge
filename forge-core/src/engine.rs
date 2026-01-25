//! Inference engine - orchestrates forward passes with real model execution
//!
//! This module implements the core inference loop using multiple model architectures
//! including LLaMA, Qwen2, and Qwen3 via the CausalLM trait.

use anyhow::{anyhow, Result};
use candle_core::{Device, Tensor};
use std::sync::{Arc, Mutex};
use tracing::info;

use crate::model::{LoadedModel, ModelConfig, TokenOutputStream, create_logits_processor, load_model};
use crate::request::SamplingParams;

/// Configuration for the inference engine
#[derive(Debug, Clone)]
pub struct EngineConfig {
    /// Model configuration
    pub model_config: ModelConfig,
    /// Maximum sequence length
    pub max_seq_len: usize,
    /// Repeat penalty (1.0 = no penalty)
    pub repeat_penalty: f32,
    /// Context size for repeat penalty
    pub repeat_last_n: usize,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            model_config: ModelConfig::default(),
            max_seq_len: 4096,
            repeat_penalty: 1.1,
            repeat_last_n: 64,
        }
    }
}

impl EngineConfig {
    pub fn with_model(model_config: ModelConfig) -> Self {
        Self {
            model_config,
            ..Default::default()
        }
    }
}

/// Engine statistics
#[derive(Debug, Clone, serde::Serialize)]
pub struct EngineStats {
    pub device: String,
    pub model_id: String,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub dtype: String,
    pub max_seq_len: usize,
}

/// Result of a single generation
#[derive(Debug, Clone)]
pub struct GenerationResult {
    /// Generated tokens
    pub tokens: Vec<u32>,
    /// Generated text
    pub text: String,
    /// Number of prompt tokens
    pub prompt_tokens: usize,
    /// Number of generated tokens
    pub generated_tokens: usize,
    /// Tokens per second
    pub tokens_per_second: f64,
    /// Finish reason
    pub finish_reason: FinishReason,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub enum FinishReason {
    Length,
    Stop,
    Error,
}

/// Inference engine that performs actual model execution
pub struct InferenceEngine {
    config: EngineConfig,
    device: Device,
    model: Arc<Mutex<LoadedModel>>,
}

impl InferenceEngine {
    /// Create a new inference engine with the specified model
    pub fn new(config: EngineConfig, device: Device) -> Result<Self> {
        info!("Initializing inference engine...");
        let model = load_model(&config.model_config, &device)?;
        info!("Loaded model architecture: {:?}", model.arch);

        Ok(Self {
            config,
            device,
            model: Arc::new(Mutex::new(model)),
        })
    }

    /// Create with default config for CPU using SmolLM-135M (small, fast model)
    pub fn new_cpu() -> Result<Self> {
        let config = EngineConfig::with_model(ModelConfig::smollm_135m());
        Self::new(config, Device::Cpu)
    }

    /// Create with TinyLlama 1.1B on CPU
    pub fn new_cpu_tinyllama() -> Result<Self> {
        let config = EngineConfig::with_model(ModelConfig::tinyllama());
        Self::new(config, Device::Cpu)
    }

    /// Create with SmolLM-360M on CPU
    pub fn new_cpu_smollm_360m() -> Result<Self> {
        let config = EngineConfig::with_model(ModelConfig::smollm_360m());
        Self::new(config, Device::Cpu)
    }

    /// Create with Metal backend (macOS)
    #[cfg(feature = "metal")]
    pub fn new_metal() -> Result<Self> {
        let device = Device::new_metal(0)?;
        let config = EngineConfig::with_model(ModelConfig::smollm_135m());
        Self::new(config, device)
    }

    /// Create with Metal backend using TinyLlama
    #[cfg(feature = "metal")]
    pub fn new_metal_tinyllama() -> Result<Self> {
        let device = Device::new_metal(0)?;
        let config = EngineConfig::with_model(ModelConfig::tinyllama());
        Self::new(config, device)
    }

    /// Create with CUDA backend
    #[cfg(feature = "cuda")]
    pub fn new_cuda(device_id: usize) -> Result<Self> {
        let device = Device::new_cuda(device_id)?;
        let config = EngineConfig::with_model(ModelConfig::smollm_135m());
        Self::new(config, device)
    }

    /// Get the device being used
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Get access to the loaded model (locked)
    pub fn with_model<F, R>(&self, f: F) -> Result<R>
    where
        F: FnOnce(&LoadedModel) -> R,
    {
        let model = self.model.lock().map_err(|e| anyhow!("Failed to lock model: {}", e))?;
        Ok(f(&model))
    }

    /// Encode a prompt to tokens
    pub fn encode(&self, prompt: &str) -> Result<Vec<u32>> {
        let model = self.model.lock().map_err(|e| anyhow!("Failed to lock model: {}", e))?;
        model.tokenizer
            .encode(prompt, true)
            .map_err(|e| anyhow!("Tokenization error: {}", e))
            .map(|enc| enc.get_ids().to_vec())
    }

    /// Decode tokens to text
    pub fn decode(&self, tokens: &[u32]) -> Result<String> {
        let model = self.model.lock().map_err(|e| anyhow!("Failed to lock model: {}", e))?;
        model.tokenizer
            .decode(tokens, true)
            .map_err(|e| anyhow!("Decode error: {}", e))
    }

    /// Generate text from a prompt
    pub fn generate(&self, prompt: &str, params: &SamplingParams) -> Result<GenerationResult> {
        let tokens = self.encode(prompt)?;
        self.generate_from_tokens(&tokens, params)
    }

    /// Generate text from tokens
    pub fn generate_from_tokens(&self, prompt_tokens: &[u32], params: &SamplingParams) -> Result<GenerationResult> {
        let prompt_len = prompt_tokens.len();
        let mut tokens = prompt_tokens.to_vec();
        let mut generated_tokens = Vec::new();

        // Create logits processor
        let mut logits_processor = create_logits_processor(
            params.seed.unwrap_or(42),
            params.temperature,
            params.top_p,
            params.top_k,
        );

        let start_time = std::time::Instant::now();
        let mut index_pos = 0;

        // Clear KV cache before starting generation
        {
            let mut model = self.model.lock().map_err(|e| anyhow!("Failed to lock model: {}", e))?;
            model.model.clear_kv_cache();
        }

        for i in 0..params.max_tokens {
            // Determine context size (full for first token, 1 for subsequent)
            let (context_size, context_index) = if i > 0 {
                (1, index_pos)
            } else {
                (tokens.len(), 0)
            };

            // Get the context tokens
            let ctxt = &tokens[tokens.len().saturating_sub(context_size)..];
            let input = Tensor::new(ctxt, &self.device)?.unsqueeze(0)?;

            // Forward pass through the model
            let logits = {
                let mut model = self.model.lock().map_err(|e| anyhow!("Failed to lock model: {}", e))?;
                model.model.forward(&input, context_index)?
            };
            // Squeeze out batch dimension, handle both [batch, vocab] and [batch, 1, vocab] shapes
            let logits = logits.squeeze(0).map_err(|e| anyhow!("Squeeze batch dim error: {}", e))?;
            let logits = if logits.dims().len() > 1 {
                logits.squeeze(0).map_err(|e| anyhow!("Squeeze seq dim error: {}", e))?
            } else {
                logits
            };

            // Apply repeat penalty
            let logits = if self.config.repeat_penalty != 1.0 {
                let start_at = tokens.len().saturating_sub(self.config.repeat_last_n);
                candle_transformers::utils::apply_repeat_penalty(
                    &logits,
                    self.config.repeat_penalty,
                    &tokens[start_at..],
                ).map_err(|e| anyhow!("Repeat penalty error: {}", e))?
            } else {
                logits
            };

            index_pos += ctxt.len();

            // Sample next token
            let next_token = logits_processor.sample(&logits)
                .map_err(|e| anyhow!("Sampling error: {}", e))?;

            // Check for stop tokens
            if params.stop_tokens.contains(&next_token) {
                return Ok(GenerationResult {
                    tokens: generated_tokens.clone(),
                    text: self.decode(&generated_tokens)?,
                    prompt_tokens: prompt_len,
                    generated_tokens: generated_tokens.len(),
                    tokens_per_second: generated_tokens.len() as f64 / start_time.elapsed().as_secs_f64(),
                    finish_reason: FinishReason::Stop,
                });
            }

            // Check for EOS token
            let is_eos = {
                let model = self.model.lock().map_err(|e| anyhow!("Failed to lock model: {}", e))?;
                model.is_eos_token(next_token)
            };

            if is_eos {
                return Ok(GenerationResult {
                    tokens: generated_tokens.clone(),
                    text: self.decode(&generated_tokens)?,
                    prompt_tokens: prompt_len,
                    generated_tokens: generated_tokens.len(),
                    tokens_per_second: generated_tokens.len() as f64 / start_time.elapsed().as_secs_f64(),
                    finish_reason: FinishReason::Stop,
                });
            }

            generated_tokens.push(next_token);
            tokens.push(next_token);
        }

        let elapsed = start_time.elapsed();
        let tokens_per_second = generated_tokens.len() as f64 / elapsed.as_secs_f64();

        Ok(GenerationResult {
            tokens: generated_tokens.clone(),
            text: self.decode(&generated_tokens)?,
            prompt_tokens: prompt_len,
            generated_tokens: generated_tokens.len(),
            tokens_per_second,
            finish_reason: FinishReason::Length,
        })
    }

    /// Generate with streaming output
    pub fn generate_streaming<F>(&self, prompt: &str, params: &SamplingParams, mut callback: F) -> Result<GenerationResult>
    where
        F: FnMut(&str),
    {
        let prompt_tokens = self.encode(prompt)?;
        let prompt_len = prompt_tokens.len();
        let mut tokens = prompt_tokens.clone();
        let mut generated_tokens = Vec::new();

        // Create logits processor
        let mut logits_processor = create_logits_processor(
            params.seed.unwrap_or(42),
            params.temperature,
            params.top_p,
            params.top_k,
        );

        // Create token output stream for incremental decoding
        let token_stream = {
            let model = self.model.lock().map_err(|e| anyhow!("Failed to lock model: {}", e))?;
            TokenOutputStream::new(model.tokenizer.clone())
        };
        let mut token_stream = token_stream;

        let start_time = std::time::Instant::now();
        let mut index_pos = 0;

        // Clear KV cache before starting generation
        {
            let mut model = self.model.lock().map_err(|e| anyhow!("Failed to lock model: {}", e))?;
            model.model.clear_kv_cache();
        }

        for i in 0..params.max_tokens {
            let (context_size, context_index) = if i > 0 {
                (1, index_pos)
            } else {
                (tokens.len(), 0)
            };

            let ctxt = &tokens[tokens.len().saturating_sub(context_size)..];
            let input = Tensor::new(ctxt, &self.device)?.unsqueeze(0)?;

            // Forward pass through the model
            let logits = {
                let mut model = self.model.lock().map_err(|e| anyhow!("Failed to lock model: {}", e))?;
                model.model.forward(&input, context_index)?
            };
            // Squeeze out batch dimension, handle both [batch, vocab] and [batch, 1, vocab] shapes
            let logits = logits.squeeze(0).map_err(|e| anyhow!("Squeeze batch dim error: {}", e))?;
            let logits = if logits.dims().len() > 1 {
                logits.squeeze(0).map_err(|e| anyhow!("Squeeze seq dim error: {}", e))?
            } else {
                logits
            };

            let logits = if self.config.repeat_penalty != 1.0 {
                let start_at = tokens.len().saturating_sub(self.config.repeat_last_n);
                candle_transformers::utils::apply_repeat_penalty(
                    &logits,
                    self.config.repeat_penalty,
                    &tokens[start_at..],
                ).map_err(|e| anyhow!("Repeat penalty error: {}", e))?
            } else {
                logits
            };

            index_pos += ctxt.len();

            let next_token = logits_processor.sample(&logits)
                .map_err(|e| anyhow!("Sampling error: {}", e))?;

            // Check for stop tokens
            if params.stop_tokens.contains(&next_token) {
                // Flush remaining tokens
                if let Ok(Some(text)) = token_stream.decode_rest() {
                    callback(&text);
                }
                return Ok(GenerationResult {
                    tokens: generated_tokens.clone(),
                    text: token_stream.decode_all()?,
                    prompt_tokens: prompt_len,
                    generated_tokens: generated_tokens.len(),
                    tokens_per_second: generated_tokens.len() as f64 / start_time.elapsed().as_secs_f64(),
                    finish_reason: FinishReason::Stop,
                });
            }

            // Check for EOS token
            let is_eos = {
                let model = self.model.lock().map_err(|e| anyhow!("Failed to lock model: {}", e))?;
                model.is_eos_token(next_token)
            };

            if is_eos {
                if let Ok(Some(text)) = token_stream.decode_rest() {
                    callback(&text);
                }
                return Ok(GenerationResult {
                    tokens: generated_tokens.clone(),
                    text: token_stream.decode_all()?,
                    prompt_tokens: prompt_len,
                    generated_tokens: generated_tokens.len(),
                    tokens_per_second: generated_tokens.len() as f64 / start_time.elapsed().as_secs_f64(),
                    finish_reason: FinishReason::Stop,
                });
            }

            generated_tokens.push(next_token);
            tokens.push(next_token);

            // Stream output
            if let Ok(Some(text)) = token_stream.next_token(next_token) {
                callback(&text);
            }
        }

        // Flush remaining tokens
        if let Ok(Some(text)) = token_stream.decode_rest() {
            callback(&text);
        }

        let elapsed = start_time.elapsed();
        let tokens_per_second = generated_tokens.len() as f64 / elapsed.as_secs_f64();

        Ok(GenerationResult {
            tokens: generated_tokens,
            text: token_stream.decode_all()?,
            prompt_tokens: prompt_len,
            generated_tokens: token_stream.get_tokens().len(),
            tokens_per_second,
            finish_reason: FinishReason::Length,
        })
    }

    /// Get engine statistics
    pub fn stats(&self) -> EngineStats {
        let model = self.model.lock().unwrap();
        EngineStats {
            device: format!("{:?}", self.device),
            model_id: self.config.model_config.model_id.clone(),
            vocab_size: model.vocab_size(),
            hidden_size: model.hidden_size(),
            num_layers: model.num_layers(),
            dtype: format!("{:?}", model.dtype),
            max_seq_len: self.config.max_seq_len,
        }
    }
}
