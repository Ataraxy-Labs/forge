//! Model loading and management
//!
//! Handles downloading models from HuggingFace Hub and loading them into memory.
//! Supports multiple model architectures including LLaMA, Qwen2, and Qwen3.

use anyhow::{anyhow, bail, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::llama::{Cache, Config as LlamaConfig, Llama, LlamaConfig as LlamaHfConfig, LlamaEosToks};
use candle_transformers::models::qwen2::{Config as Qwen2Config, ModelForCausalLM as Qwen2Model};
use candle_transformers::models::qwen3::{Config as Qwen3Config, ModelForCausalLM as Qwen3Model};
use candle_transformers::generation::{LogitsProcessor, Sampling};
use hf_hub::{api::sync::Api, Repo, RepoType};
use std::path::PathBuf;
use std::sync::Mutex;
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
    Qwen2,
    Qwen3,
}

impl ModelArch {
    pub fn from_model_id(model_id: &str) -> Self {
        let lower = model_id.to_lowercase();
        // Check Qwen first (more specific patterns)
        if lower.contains("qwen3") || lower.contains("qwen-3") || lower.contains("qwen/qwen3") {
            ModelArch::Qwen3
        } else if lower.contains("qwen2") || lower.contains("qwen-2") || lower.contains("qwen/qwen2") || lower.contains("qwen-") {
            ModelArch::Qwen2
        } else if lower.contains("llama-3.2") || lower.contains("llama-32") {
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
           lower.contains("135m") || lower.contains("360m") ||
           lower.contains("0.5b") || lower.contains("0.6b") ||
           lower.contains("1.5b") || lower.contains("1.7b") {
            false
        } else {
            true
        }
    }

    /// Check if this architecture is a Qwen model
    pub fn is_qwen(&self) -> bool {
        matches!(self, ModelArch::Qwen2 | ModelArch::Qwen3)
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
            use_flash_attn: true,
        }
    }
}

impl ModelConfig {
    pub fn llama32_1b() -> Self {
        Self {
            model_id: "meta-llama/Llama-3.2-1B".to_string(),
            revision: "main".to_string(),
            dtype: DType::F16,
            use_flash_attn: true,
        }
    }

    pub fn llama32_3b() -> Self {
        Self {
            model_id: "meta-llama/Llama-3.2-3B".to_string(),
            revision: "main".to_string(),
            dtype: DType::F16,
            use_flash_attn: true,
        }
    }

    pub fn tinyllama() -> Self {
        Self {
            model_id: "TinyLlama/TinyLlama-1.1B-Chat-v1.0".to_string(),
            revision: "main".to_string(),
            dtype: DType::F32,
            use_flash_attn: true,
        }
    }

    pub fn smollm_135m() -> Self {
        Self {
            model_id: "HuggingFaceTB/SmolLM2-135M".to_string(),
            revision: "main".to_string(),
            dtype: DType::F16,
            use_flash_attn: true,
        }
    }

    pub fn smollm_360m() -> Self {
        Self {
            model_id: "HuggingFaceTB/SmolLM2-360M".to_string(),
            revision: "main".to_string(),
            dtype: DType::F32,
            use_flash_attn: true,
        }
    }

    pub fn smollm_1_7b() -> Self {
        Self {
            model_id: "HuggingFaceTB/SmolLM2-1.7B".to_string(),
            revision: "main".to_string(),
            dtype: DType::F16,
            use_flash_attn: true,
        }
    }

    // Qwen2 model presets
    pub fn qwen2_0_5b() -> Self {
        Self {
            model_id: "Qwen/Qwen2-0.5B".to_string(),
            revision: "main".to_string(),
            dtype: DType::F16,
            use_flash_attn: true,
        }
    }

    pub fn qwen2_1_5b() -> Self {
        Self {
            model_id: "Qwen/Qwen2-1.5B".to_string(),
            revision: "main".to_string(),
            dtype: DType::F16,
            use_flash_attn: true,
        }
    }

    pub fn qwen2_7b() -> Self {
        Self {
            model_id: "Qwen/Qwen2-7B".to_string(),
            revision: "main".to_string(),
            dtype: DType::F16,
            use_flash_attn: true,
        }
    }

    // Qwen3 model presets
    pub fn qwen3_0_6b() -> Self {
        Self {
            model_id: "Qwen/Qwen3-0.6B".to_string(),
            revision: "main".to_string(),
            dtype: DType::F16,
            use_flash_attn: true,
        }
    }

    pub fn qwen3_1_7b() -> Self {
        Self {
            model_id: "Qwen/Qwen3-1.7B".to_string(),
            revision: "main".to_string(),
            dtype: DType::F16,
            use_flash_attn: true,
        }
    }

    pub fn qwen3_4b() -> Self {
        Self {
            model_id: "Qwen/Qwen3-4B".to_string(),
            revision: "main".to_string(),
            dtype: DType::F16,
            use_flash_attn: true,
        }
    }
}

/// Trait for causal language models - abstracts over different model architectures
pub trait CausalLM: Send {
    /// Forward pass - takes input tokens and sequence offset, returns logits
    fn forward(&mut self, input_ids: &Tensor, seq_offset: usize) -> Result<Tensor>;

    /// Clear the KV cache
    fn clear_kv_cache(&mut self);

    /// Get vocabulary size
    fn vocab_size(&self) -> usize;

    /// Get hidden size
    fn hidden_size(&self) -> usize;

    /// Get number of layers
    fn num_layers(&self) -> usize;
}

/// Wrapper for Llama models that implements CausalLM
pub struct LlamaWrapper {
    model: Llama,
    cache: Cache,  // No Mutex - already protected by outer lock in engine
    config: LlamaConfig,
    dtype: DType,
    device: Device,
}

impl LlamaWrapper {
    pub fn new(model: Llama, config: LlamaConfig, dtype: DType, device: &Device) -> Result<Self> {
        let cache = Cache::new(true, dtype, &config, device).map_err(|e| anyhow!(e))?;
        Ok(Self {
            model,
            cache,  // Direct ownership, no mutex
            config,
            dtype,
            device: device.clone(),
        })
    }
}

impl CausalLM for LlamaWrapper {
    fn forward(&mut self, input_ids: &Tensor, seq_offset: usize) -> Result<Tensor> {
        // No mutex lock needed - caller already has exclusive access via &mut self
        self.model.forward(input_ids, seq_offset, &mut self.cache)
            .map_err(|e| anyhow!("Llama forward error: {}", e))
    }

    fn clear_kv_cache(&mut self) {
        // Create a new cache to reset KV state - no mutex needed
        if let Ok(new_cache) = Cache::new(true, self.dtype, &self.config, &self.device) {
            self.cache = new_cache;
        }
    }

    fn vocab_size(&self) -> usize {
        self.config.vocab_size
    }

    fn hidden_size(&self) -> usize {
        self.config.hidden_size
    }

    fn num_layers(&self) -> usize {
        self.config.num_hidden_layers
    }
}

/// Wrapper for Qwen2 models that implements CausalLM
pub struct Qwen2Wrapper {
    model: Qwen2Model,
    config: Qwen2Config,
}

impl Qwen2Wrapper {
    pub fn new(model: Qwen2Model, config: Qwen2Config) -> Self {
        Self { model, config }
    }
}

impl CausalLM for Qwen2Wrapper {
    fn forward(&mut self, input_ids: &Tensor, seq_offset: usize) -> Result<Tensor> {
        self.model.forward(input_ids, seq_offset)
            .map_err(|e| anyhow!("Qwen2 forward error: {}", e))
    }

    fn clear_kv_cache(&mut self) {
        self.model.clear_kv_cache();
    }

    fn vocab_size(&self) -> usize {
        self.config.vocab_size
    }

    fn hidden_size(&self) -> usize {
        self.config.hidden_size
    }

    fn num_layers(&self) -> usize {
        self.config.num_hidden_layers
    }
}

/// Wrapper for Qwen3 models that implements CausalLM
pub struct Qwen3Wrapper {
    model: Qwen3Model,
    config: Qwen3Config,
}

impl Qwen3Wrapper {
    pub fn new(model: Qwen3Model, config: Qwen3Config) -> Self {
        Self { model, config }
    }
}

impl CausalLM for Qwen3Wrapper {
    fn forward(&mut self, input_ids: &Tensor, seq_offset: usize) -> Result<Tensor> {
        self.model.forward(input_ids, seq_offset)
            .map_err(|e| anyhow!("Qwen3 forward error: {}", e))
    }

    fn clear_kv_cache(&mut self) {
        self.model.clear_kv_cache();
    }

    fn vocab_size(&self) -> usize {
        self.config.vocab_size
    }

    fn hidden_size(&self) -> usize {
        self.config.hidden_size
    }

    fn num_layers(&self) -> usize {
        self.config.num_hidden_layers
    }
}

/// EOS token representation that works across model types
#[derive(Debug, Clone)]
pub enum EosToken {
    Single(u32),
    Multiple(Vec<u32>),
}

impl EosToken {
    pub fn contains(&self, token: u32) -> bool {
        match self {
            EosToken::Single(eos) => token == *eos,
            EosToken::Multiple(eos_ids) => eos_ids.contains(&token),
        }
    }
}

impl From<LlamaEosToks> for EosToken {
    fn from(eos: LlamaEosToks) -> Self {
        match eos {
            LlamaEosToks::Single(id) => EosToken::Single(id),
            LlamaEosToks::Multiple(ids) => EosToken::Multiple(ids),
        }
    }
}

/// Loaded model ready for inference
pub struct LoadedModel {
    pub model: Box<dyn CausalLM>,
    pub tokenizer: Tokenizer,
    pub arch: ModelArch,
    pub device: Device,
    pub dtype: DType,
    pub eos_token_id: Option<EosToken>,
    // Cached model info
    vocab_size: usize,
    hidden_size: usize,
    num_layers: usize,
}

impl LoadedModel {
    /// Get vocab size
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Get hidden size
    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    /// Get number of layers
    pub fn num_layers(&self) -> usize {
        self.num_layers
    }

    /// Check if a token is an EOS token
    pub fn is_eos_token(&self, token: u32) -> bool {
        self.eos_token_id.as_ref().map_or(false, |eos| eos.contains(token))
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

    // Detect architecture
    let arch = ModelArch::from_model_id(&config.model_id);
    info!("Detected architecture: {:?}", arch);

    // Load model weights
    info!("Loading model weights...");
    let is_sharded = arch.is_sharded(&config.model_id);
    let filenames = load_safetensors(&repo, is_sharded)?;
    info!("Loading {} safetensor file(s)", filenames.len());

    // Load config and model based on architecture
    let config_file = repo.get("config.json")?;
    let config_bytes = std::fs::read(&config_file)?;

    match arch {
        ModelArch::Qwen2 => load_qwen2_model(config, device, tokenizer, &config_bytes, &filenames, arch),
        ModelArch::Qwen3 => load_qwen3_model(config, device, tokenizer, &config_bytes, &filenames, arch),
        _ => load_llama_model(config, device, tokenizer, &config_bytes, &filenames, arch),
    }
}

/// Load a Llama-family model
fn load_llama_model(
    config: &ModelConfig,
    device: &Device,
    tokenizer: Tokenizer,
    config_bytes: &[u8],
    filenames: &[PathBuf],
    arch: ModelArch,
) -> Result<LoadedModel> {
    let llama_config: LlamaHfConfig = serde_json::from_slice(config_bytes)?;
    let model_config = llama_config.into_config(config.use_flash_attn);

    // Get EOS token ID
    let eos_token_id = model_config.eos_token_id.clone()
        .map(EosToken::from)
        .or_else(|| tokenizer.token_to_id("</s>").map(EosToken::Single));

    let vb = unsafe { VarBuilder::from_mmaped_safetensors(filenames, config.dtype, device)? };
    let model = Llama::load(vb, &model_config).map_err(|e| anyhow!("Failed to load Llama model: {}", e))?;

    let vocab_size = model_config.vocab_size;
    let hidden_size = model_config.hidden_size;
    let num_layers = model_config.num_hidden_layers;

    info!("Llama model loaded successfully!");
    info!("  - Vocab size: {}", vocab_size);
    info!("  - Hidden size: {}", hidden_size);
    info!("  - Layers: {}", num_layers);
    info!("  - Heads: {}", model_config.num_attention_heads);
    info!("  - KV Heads: {}", model_config.num_key_value_heads);

    let wrapper = LlamaWrapper::new(model, model_config, config.dtype, device)?;

    Ok(LoadedModel {
        model: Box::new(wrapper),
        tokenizer,
        arch,
        device: device.clone(),
        dtype: config.dtype,
        eos_token_id,
        vocab_size,
        hidden_size,
        num_layers,
    })
}

/// Load a Qwen2 model
fn load_qwen2_model(
    config: &ModelConfig,
    device: &Device,
    tokenizer: Tokenizer,
    config_bytes: &[u8],
    filenames: &[PathBuf],
    arch: ModelArch,
) -> Result<LoadedModel> {
    let qwen_config: Qwen2Config = serde_json::from_slice(config_bytes)?;

    // Get EOS token from tokenizer
    let eos_token_id = tokenizer.token_to_id("<|endoftext|>")
        .or_else(|| tokenizer.token_to_id("</s>"))
        .or_else(|| tokenizer.token_to_id("<|im_end|>"))
        .map(EosToken::Single);

    let vb = unsafe { VarBuilder::from_mmaped_safetensors(filenames, config.dtype, device)? };
    let model = Qwen2Model::new(&qwen_config, vb).map_err(|e| anyhow!("Failed to load Qwen2 model: {}", e))?;

    let vocab_size = qwen_config.vocab_size;
    let hidden_size = qwen_config.hidden_size;
    let num_layers = qwen_config.num_hidden_layers;

    info!("Qwen2 model loaded successfully!");
    info!("  - Vocab size: {}", vocab_size);
    info!("  - Hidden size: {}", hidden_size);
    info!("  - Layers: {}", num_layers);
    info!("  - Heads: {}", qwen_config.num_attention_heads);
    info!("  - KV Heads: {}", qwen_config.num_key_value_heads);

    let wrapper = Qwen2Wrapper::new(model, qwen_config);

    Ok(LoadedModel {
        model: Box::new(wrapper),
        tokenizer,
        arch,
        device: device.clone(),
        dtype: config.dtype,
        eos_token_id,
        vocab_size,
        hidden_size,
        num_layers,
    })
}

/// Load a Qwen3 model
fn load_qwen3_model(
    config: &ModelConfig,
    device: &Device,
    tokenizer: Tokenizer,
    config_bytes: &[u8],
    filenames: &[PathBuf],
    arch: ModelArch,
) -> Result<LoadedModel> {
    let qwen_config: Qwen3Config = serde_json::from_slice(config_bytes)?;

    // Get EOS token from tokenizer
    let eos_token_id = tokenizer.token_to_id("<|endoftext|>")
        .or_else(|| tokenizer.token_to_id("</s>"))
        .or_else(|| tokenizer.token_to_id("<|im_end|>"))
        .map(EosToken::Single);

    let vb = unsafe { VarBuilder::from_mmaped_safetensors(filenames, config.dtype, device)? };
    let model = Qwen3Model::new(&qwen_config, vb).map_err(|e| anyhow!("Failed to load Qwen3 model: {}", e))?;

    let vocab_size = qwen_config.vocab_size;
    let hidden_size = qwen_config.hidden_size;
    let num_layers = qwen_config.num_hidden_layers;

    info!("Qwen3 model loaded successfully!");
    info!("  - Vocab size: {}", vocab_size);
    info!("  - Hidden size: {}", hidden_size);
    info!("  - Layers: {}", num_layers);
    info!("  - Heads: {}", qwen_config.num_attention_heads);
    info!("  - KV Heads: {}", qwen_config.num_key_value_heads);

    let wrapper = Qwen3Wrapper::new(model, qwen_config);

    Ok(LoadedModel {
        model: Box::new(wrapper),
        tokenizer,
        arch,
        device: device.clone(),
        dtype: config.dtype,
        eos_token_id,
        vocab_size,
        hidden_size,
        num_layers,
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
