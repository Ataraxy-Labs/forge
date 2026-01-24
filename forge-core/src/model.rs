//! Model loading and management
//!
//! Handles downloading models from HuggingFace Hub and loading them into memory.

use anyhow::{anyhow, bail, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::llama::{self as llama_model, Cache, Config, Llama, LlamaConfig, LlamaEosToks};
use candle_transformers::generation::{LogitsProcessor, Sampling};
use hf_hub::{api::sync::Api, Repo, RepoType};
use std::path::PathBuf;
use tokenizers::Tokenizer;
use tracing::info;

/// Supported model architectures
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelArch {
    Llama,
    Llama2,
    Llama3,
    Llama31,
    Llama32,
    Mistral,
    TinyLlama,
    SmolLM,
}

impl ModelArch {
    pub fn from_model_id(model_id: &str) -> Self {
        let lower = model_id.to_lowercase();
        if lower.contains("llama-3.2") || lower.contains("llama-32") {
            ModelArch::Llama32
        } else if lower.contains("llama-3.1") || lower.contains("llama-31") {
            ModelArch::Llama31
        } else if lower.contains("llama-3") || lower.contains("llama3") {
            ModelArch::Llama3
        } else if lower.contains("llama-2") || lower.contains("llama2") {
            ModelArch::Llama2
        } else if lower.contains("mistral") {
            ModelArch::Mistral
        } else if lower.contains("tinyllama") {
            ModelArch::TinyLlama
        } else if lower.contains("smol") {
            ModelArch::SmolLM
        } else {
            ModelArch::Llama
        }
    }

    pub fn is_sharded(&self, model_id: &str) -> bool {
        let lower = model_id.to_lowercase();
        // Smaller models are single file
        if lower.contains("1b") || lower.contains("3b") ||
           lower.contains("tiny") || lower.contains("smol") ||
           lower.contains("135m") || lower.contains("360m") {
            false
        } else {
            true
        }
    }
}

/// Configuration for model loading
#[derive(Debug, Clone)]
pub struct ModelConfig {
    /// HuggingFace model ID (e.g., "meta-llama/Llama-3.2-1B")
    pub model_id: String,
    /// Model revision/branch
    pub revision: String,
    /// Data type for model weights
    pub dtype: DType,
    /// Whether to use flash attention
    pub use_flash_attn: bool,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            model_id: "HuggingFaceTB/SmolLM2-135M".to_string(),
            revision: "main".to_string(),
            dtype: DType::F32,
            use_flash_attn: false,
        }
    }
}

impl ModelConfig {
    pub fn llama32_1b() -> Self {
        Self {
            model_id: "meta-llama/Llama-3.2-1B".to_string(),
            revision: "main".to_string(),
            dtype: DType::F16,
            use_flash_attn: false,
        }
    }

    pub fn llama32_3b() -> Self {
        Self {
            model_id: "meta-llama/Llama-3.2-3B".to_string(),
            revision: "main".to_string(),
            dtype: DType::F16,
            use_flash_attn: false,
        }
    }

    pub fn tinyllama() -> Self {
        Self {
            model_id: "TinyLlama/TinyLlama-1.1B-Chat-v1.0".to_string(),
            revision: "main".to_string(),
            dtype: DType::F32,
            use_flash_attn: false,
        }
    }

    pub fn smollm_135m() -> Self {
        Self {
            model_id: "HuggingFaceTB/SmolLM2-135M".to_string(),
            revision: "main".to_string(),
            dtype: DType::F32,
            use_flash_attn: false,
        }
    }

    pub fn smollm_360m() -> Self {
        Self {
            model_id: "HuggingFaceTB/SmolLM2-360M".to_string(),
            revision: "main".to_string(),
            dtype: DType::F32,
            use_flash_attn: false,
        }
    }

    pub fn smollm_1_7b() -> Self {
        Self {
            model_id: "HuggingFaceTB/SmolLM2-1.7B".to_string(),
            revision: "main".to_string(),
            dtype: DType::F16,
            use_flash_attn: false,
        }
    }
}

/// Loaded model ready for inference
pub struct LoadedModel {
    pub model: Llama,
    pub tokenizer: Tokenizer,
    pub config: Config,
    pub device: Device,
    pub dtype: DType,
    pub eos_token_id: Option<LlamaEosToks>,
}

impl LoadedModel {
    /// Create a new KV cache for this model
    pub fn new_cache(&self) -> Result<Cache> {
        Cache::new(true, self.dtype, &self.config, &self.device).map_err(|e| anyhow!(e))
    }

    /// Get vocab size
    pub fn vocab_size(&self) -> usize {
        self.config.vocab_size
    }

    /// Get hidden size
    pub fn hidden_size(&self) -> usize {
        self.config.hidden_size
    }

    /// Get number of layers
    pub fn num_layers(&self) -> usize {
        self.config.num_hidden_layers
    }
}

/// Load safetensors files (handles both single and sharded models)
fn load_safetensors(api: &hf_hub::api::sync::ApiRepo, is_sharded: bool) -> Result<Vec<PathBuf>> {
    if is_sharded {
        // Try to load sharded model
        let json_file = api.get("model.safetensors.index.json")?;
        let json: serde_json::Value = serde_json::from_reader(&std::fs::File::open(&json_file)?)?;
        let weight_map = match json.get("weight_map") {
            Some(serde_json::Value::Object(map)) => map,
            _ => bail!("no weight map in {:?}", json_file),
        };

        let mut safetensors_files = std::collections::HashSet::new();
        for value in weight_map.values() {
            if let Some(file) = value.as_str() {
                safetensors_files.insert(file.to_string());
            }
        }

        let safetensors_files = safetensors_files
            .iter()
            .map(|v| api.get(v))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(safetensors_files)
    } else {
        // Single model.safetensors file
        Ok(vec![api.get("model.safetensors")?])
    }
}

/// Load a model from HuggingFace Hub
pub fn load_model(config: &ModelConfig, device: &Device) -> Result<LoadedModel> {
    info!("Loading model: {}", config.model_id);

    let api = Api::new()?;
    let repo = api.repo(Repo::with_revision(
        config.model_id.clone(),
        RepoType::Model,
        config.revision.clone(),
    ));

    // Load tokenizer
    info!("Loading tokenizer...");
    let tokenizer_file = repo.get("tokenizer.json")?;
    let tokenizer = Tokenizer::from_file(&tokenizer_file).map_err(|e| anyhow!("Failed to load tokenizer: {}", e))?;

    // Load config
    info!("Loading model config...");
    let config_file = repo.get("config.json")?;
    let llama_config: LlamaConfig = serde_json::from_slice(&std::fs::read(&config_file)?)?;
    let model_config = llama_config.into_config(config.use_flash_attn);

    // Get EOS token ID
    let eos_token_id = model_config.eos_token_id.clone().or_else(|| {
        tokenizer.token_to_id("</s>").map(LlamaEosToks::Single)
    });

    // Load model weights
    info!("Loading model weights...");
    let arch = ModelArch::from_model_id(&config.model_id);
    let is_sharded = arch.is_sharded(&config.model_id);

    let filenames = load_safetensors(&repo, is_sharded)?;
    info!("Loading {} safetensor file(s)", filenames.len());

    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&filenames, config.dtype, device)? };
    let model = Llama::load(vb, &model_config).map_err(|e| anyhow!("Failed to load model: {}", e))?;

    info!("Model loaded successfully!");
    info!("  - Vocab size: {}", model_config.vocab_size);
    info!("  - Hidden size: {}", model_config.hidden_size);
    info!("  - Layers: {}", model_config.num_hidden_layers);
    info!("  - Heads: {}", model_config.num_attention_heads);
    info!("  - KV Heads: {}", model_config.num_key_value_heads);

    Ok(LoadedModel {
        model,
        tokenizer,
        config: model_config,
        device: device.clone(),
        dtype: config.dtype,
        eos_token_id,
    })
}

/// Token output stream for incremental decoding
pub struct TokenOutputStream {
    tokenizer: Tokenizer,
    tokens: Vec<u32>,
    prev_index: usize,
    current_index: usize,
}

impl TokenOutputStream {
    pub fn new(tokenizer: Tokenizer) -> Self {
        Self {
            tokenizer,
            tokens: Vec::new(),
            prev_index: 0,
            current_index: 0,
        }
    }

    pub fn next_token(&mut self, token: u32) -> Result<Option<String>> {
        let prev_text = if self.tokens.is_empty() {
            String::new()
        } else {
            let tokens = &self.tokens[self.prev_index..self.current_index];
            self.tokenizer.decode(tokens, true)
                .map_err(|e| anyhow!("Decode error: {}", e))?
        };
        self.tokens.push(token);
        let text = self.tokenizer.decode(&self.tokens[self.prev_index..], true)
            .map_err(|e| anyhow!("Decode error: {}", e))?;
        if text.len() > prev_text.len() && text.chars().last().map_or(false, |c| !c.is_whitespace()) {
            let text = text.split_at(prev_text.len());
            self.current_index = self.tokens.len();
            Ok(Some(text.1.to_string()))
        } else {
            Ok(None)
        }
    }

    pub fn decode_rest(&self) -> Result<Option<String>> {
        let prev_text = if self.tokens.is_empty() {
            String::new()
        } else {
            let tokens = &self.tokens[self.prev_index..self.current_index];
            self.tokenizer.decode(tokens, true)
                .map_err(|e| anyhow!("Decode error: {}", e))?
        };
        let text = self.tokenizer.decode(&self.tokens[self.prev_index..], true)
            .map_err(|e| anyhow!("Decode error: {}", e))?;
        if text.len() > prev_text.len() {
            Ok(Some(text.split_at(prev_text.len()).1.to_string()))
        } else {
            Ok(None)
        }
    }

    pub fn decode_all(&self) -> Result<String> {
        self.tokenizer.decode(&self.tokens, true)
            .map_err(|e| anyhow!("Decode error: {}", e))
    }

    pub fn get_token(&self, idx: usize) -> Option<u32> {
        self.tokens.get(idx).copied()
    }

    pub fn get_tokens(&self) -> &[u32] {
        &self.tokens
    }

    pub fn clear(&mut self) {
        self.tokens.clear();
        self.prev_index = 0;
        self.current_index = 0;
    }
}

/// Create a logits processor with sampling parameters
pub fn create_logits_processor(
    seed: u64,
    temperature: f64,
    top_p: Option<f64>,
    top_k: Option<usize>,
) -> LogitsProcessor {
    let sampling = if temperature <= 0. {
        Sampling::ArgMax
    } else {
        match (top_k, top_p) {
            (None, None) => Sampling::All { temperature },
            (Some(k), None) => Sampling::TopK { k, temperature },
            (None, Some(p)) => Sampling::TopP { p, temperature },
            (Some(k), Some(p)) => Sampling::TopKThenTopP { k, p, temperature },
        }
    };
    LogitsProcessor::from_sampling(seed, sampling)
}
