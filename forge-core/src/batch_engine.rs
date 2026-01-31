//! Batched inference engine with continuous batching support
//!
//! This module implements a batched inference engine that can process multiple
//! requests concurrently using the scheduler for request management.

use anyhow::{anyhow, Result};
use candle_core::{Device, Tensor};
use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::Arc;
use tracing::{debug, info};

use crate::kv_cache::{PagedKVCache, PAGE_SIZE};
use crate::model::{CausalLM, ModelConfig, load_model, create_logits_processor, EosToken};
use crate::request::{InferenceRequest, RequestState, SamplingParams};
use crate::scheduler::Scheduler;

/// Configuration for the batched inference engine
#[derive(Debug, Clone)]
pub struct BatchEngineConfig {
    /// Model configuration
    pub model_config: ModelConfig,
    /// Maximum sequence length
    pub max_seq_len: usize,
    /// Number of KV cache pages
    pub num_kv_pages: usize,
    /// Repeat penalty (1.0 = no penalty)
    pub repeat_penalty: f32,
    /// Context size for repeat penalty
    pub repeat_last_n: usize,
}

impl Default for BatchEngineConfig {
    fn default() -> Self {
        Self {
            model_config: ModelConfig::default(),
            max_seq_len: 4096,
            num_kv_pages: 256,  // 256 pages * 16 tokens = 4096 max tokens
            repeat_penalty: 1.1,
            repeat_last_n: 64,
        }
    }
}

/// Statistics for the batched engine
#[derive(Debug, Clone, Default)]
pub struct BatchEngineStats {
    pub total_requests: u64,
    pub completed_requests: u64,
    pub total_tokens_generated: u64,
    pub current_batch_size: usize,
    pub pending_requests: usize,
}

/// State for a request being processed
struct ActiveRequest {
    state: RequestState,
    logits_processor: candle_transformers::generation::LogitsProcessor,
    /// KV cache pages allocated to this request
    allocated_pages: Vec<usize>,
    /// Current position in the sequence (for tracking page usage)
    seq_position: usize,
}

/// Batched inference engine with continuous batching
pub struct BatchInferenceEngine {
    config: BatchEngineConfig,
    device: Device,
    model: Arc<Mutex<Box<dyn CausalLM>>>,
    tokenizer: tokenizers::Tokenizer,
    eos_token_id: Option<EosToken>,
    kv_cache: Arc<Mutex<PagedKVCache>>,
    scheduler: Arc<Scheduler>,
    active_requests: Arc<Mutex<HashMap<u64, ActiveRequest>>>,
    stats: Arc<Mutex<BatchEngineStats>>,
    next_request_id: Arc<Mutex<u64>>,
    /// Tracks which request currently owns the model's internal KV cache
    /// When switching to a different request, we need to handle cache invalidation
    current_cache_owner: Arc<Mutex<Option<u64>>>,
}

impl BatchInferenceEngine {
    /// Create a new batched inference engine
    pub fn new(config: BatchEngineConfig, device: Device) -> Result<Self> {
        info!("Initializing batched inference engine...");

        // Load the model
        let loaded_model = load_model(&config.model_config, &device)?;
        let tokenizer = loaded_model.tokenizer.clone();
        let eos_token_id = loaded_model.eos_token_id.clone();
        let num_layers = loaded_model.num_layers();
        let num_heads = 32; // This should come from model config
        let head_dim = loaded_model.hidden_size() / num_heads;

        // Create KV cache
        let kv_cache = PagedKVCache::new(
            num_layers,
            num_heads,
            head_dim,
            config.num_kv_pages,
            loaded_model.dtype,
            device.clone(),
        )?;
        let kv_cache = Arc::new(Mutex::new(kv_cache));

        // Create scheduler
        let scheduler = Arc::new(Scheduler::new(kv_cache.clone()));

        info!("Batched engine initialized with {} KV pages", config.num_kv_pages);

        Ok(Self {
            config,
            device,
            model: Arc::new(Mutex::new(loaded_model.model)),
            tokenizer,
            eos_token_id,
            kv_cache,
            scheduler,
            active_requests: Arc::new(Mutex::new(HashMap::new())),
            stats: Arc::new(Mutex::new(BatchEngineStats::default())),
            next_request_id: Arc::new(Mutex::new(1)),
            current_cache_owner: Arc::new(Mutex::new(None)),
        })
    }

    /// Submit a new inference request
    pub fn submit_request(&self, prompt: &str, params: SamplingParams) -> Result<u64> {
        // Tokenize the prompt
        let encoding = self.tokenizer.encode(prompt, true)
            .map_err(|e| anyhow!("Tokenization error: {}", e))?;
        let tokens = encoding.get_ids().to_vec();

        // Generate request ID
        let request_id = {
            let mut id = self.next_request_id.lock();
            let rid = *id;
            *id += 1;
            rid
        };

        // Calculate prompt length before moving tokens
        let prompt_len = tokens.len();

        // Create the request
        let (tx, _rx) = tokio::sync::oneshot::channel();
        let request = InferenceRequest {
            id: request_id,
            tokens,
            params: params.clone(),
            response_tx: Arc::new(tokio::sync::Mutex::new(Some(tx))),
        };

        // Create logits processor
        let logits_processor = create_logits_processor(
            params.seed.unwrap_or(42),
            params.temperature,
            params.top_p,
            params.top_k,
        );

        // Add to scheduler
        self.scheduler.add_request(request.clone());

        // Allocate KV cache pages for this request
        // Estimate pages needed: prompt_len / PAGE_SIZE + max_tokens / PAGE_SIZE
        let max_tokens = params.max_tokens;
        let total_tokens = prompt_len + max_tokens;
        let pages_needed = (total_tokens + PAGE_SIZE - 1) / PAGE_SIZE;

        let mut allocated_pages = Vec::with_capacity(pages_needed);
        {
            let kv_cache = self.kv_cache.lock();
            for _ in 0..pages_needed {
                if let Some(page_id) = kv_cache.allocate_page() {
                    allocated_pages.push(page_id);
                } else {
                    // Out of pages - free what we allocated and return error
                    for &page_id in &allocated_pages {
                        kv_cache.free_page(page_id);
                    }
                    return Err(anyhow!("Out of KV cache pages"));
                }
            }
        }

        // Track in active requests
        {
            let mut active = self.active_requests.lock();
            active.insert(request_id, ActiveRequest {
                state: RequestState::new(request),
                logits_processor,
                allocated_pages,
                seq_position: 0,
            });
        }

        // Update stats
        {
            let mut stats = self.stats.lock();
            stats.total_requests += 1;
        }

        debug!("Submitted request {} with {} prompt tokens", request_id, encoding.get_ids().len());
        Ok(request_id)
    }

    /// Process one step of batched inference
    /// Returns list of (request_id, generated_token, is_finished) tuples
    ///
    /// OPTIMIZATION: Separates prefill and decode phases for efficiency
    /// - Prefill: Process one request at a time (variable length prompts)
    /// - Decode: Batch all decode requests together (all have 1 token, no padding needed)
    pub fn step(&self) -> Result<Vec<(u64, u32, bool)>> {
        // Get next batch from scheduler
        let batch = match self.scheduler.get_next_batch() {
            Some(b) => b,
            None => return Ok(vec![]),
        };

        let batch_size = batch.request_ids.len();
        debug!("Processing batch with {} requests", batch_size);

        // PHASE 1: Separate prefill and decode requests
        let mut prefill_requests = Vec::new();
        let mut decode_requests = Vec::new();

        {
            let active = self.active_requests.lock();
            for &request_id in &batch.request_ids {
                if let Some(active_req) = active.get(&request_id) {
                    if active_req.state.is_prefill {
                        prefill_requests.push(request_id);
                    } else {
                        decode_requests.push(request_id);
                    }
                }
            }
        }

        let mut results = Vec::new();
        let mut generated_tokens = Vec::new();

        // PHASE 2: Process prefill requests one at a time (they have variable lengths)
        // This uses the model's KV cache efficiently for each request
        for request_id in prefill_requests {
            if let Some((token, is_finished)) = self.process_prefill_request(request_id)? {
                results.push((request_id, token, is_finished));
                generated_tokens.push((request_id, token));
            }
        }

        // PHASE 3: Process decode requests efficiently
        // OPTIMIZATION: Process one request at a time to leverage KV cache
        // Prioritize current cache owner to maximize cache hits
        if !decode_requests.is_empty() {
            // Sort to prioritize cache owner (avoid allocation if possible)
            let cache_owner = *self.current_cache_owner.lock();

            // Find the request to process - prefer cache owner
            let request_id = if let Some(owner) = cache_owner {
                if decode_requests.contains(&owner) {
                    owner
                } else {
                    decode_requests[0]
                }
            } else {
                decode_requests[0]
            };

            // Process single token
            if let Some((token, is_finished)) = self.process_decode_optimized(request_id)? {
                results.push((request_id, token, is_finished));
                generated_tokens.push((request_id, token));
            }
        }

        // Clean up finished requests
        let mut finished_requests = Vec::new();
        {
            let active = self.active_requests.lock();
            for &(request_id, _, is_finished) in &results {
                if is_finished {
                    if let Some(active_req) = active.get(&request_id) {
                        finished_requests.push((request_id, active_req.allocated_pages.clone()));
                    }
                }
            }
        }

        if !finished_requests.is_empty() {
            // Free KV cache pages
            let kv_cache = self.kv_cache.lock();
            for (_, pages) in &finished_requests {
                kv_cache.free_pages(pages);
            }
            drop(kv_cache);

            // Remove from active requests and update stats
            let mut active = self.active_requests.lock();
            let mut stats = self.stats.lock();
            for (request_id, _) in finished_requests {
                active.remove(&request_id);
                stats.completed_requests += 1;
            }
        }

        // Update scheduler with generated tokens
        self.scheduler.update_requests(generated_tokens);

        // Update stats
        {
            let mut stats = self.stats.lock();
            stats.total_tokens_generated += results.len() as u64;
            stats.current_batch_size = batch_size;
        }

        Ok(results)
    }

    /// Process a prefill request (prompt processing)
    /// Uses the model's internal KV cache for efficient attention
    fn process_prefill_request(&self, request_id: u64) -> Result<Option<(u32, bool)>> {
        let mut active = self.active_requests.lock();
        let active_req = match active.get_mut(&request_id) {
            Some(r) => r,
            None => return Ok(None),
        };

        let state = &mut active_req.state;

        // Get all prompt tokens
        let input_tokens = state.request.tokens.clone();
        if input_tokens.is_empty() {
            return Err(anyhow!("Empty prompt tokens"));
        }

        // Create input tensor [1, seq_len]
        let input = Tensor::new(&input_tokens[..], &self.device)?.unsqueeze(0)?;

        // Forward pass - clear cache for new prefill
        let logits = {
            let mut model = self.model.lock();
            model.clear_kv_cache();
            model.forward(&input, 0)?
        };

        // Update cache owner to this request
        *self.current_cache_owner.lock() = Some(request_id);

        // Get last token logits
        let logits = logits.squeeze(0)?;
        let last_logits = if logits.dims().len() > 1 {
            logits.get(logits.dims()[0] - 1)?
        } else {
            logits
        };

        // Apply repeat penalty
        let penalized_logits = if self.config.repeat_penalty != 1.0 {
            let start_at = input_tokens.len().saturating_sub(self.config.repeat_last_n);
            candle_transformers::utils::apply_repeat_penalty(
                &last_logits,
                self.config.repeat_penalty,
                &input_tokens[start_at..],
            )?
        } else {
            last_logits
        };

        // Sample next token
        let next_token = active_req.logits_processor.sample(&penalized_logits)?;

        // Check if finished
        let is_eos = self.eos_token_id.as_ref()
            .map(|eos| eos.contains(next_token))
            .unwrap_or(false);
        let is_max_tokens = state.generated_tokens.len() + 1 >= state.request.params.max_tokens;
        let is_finished = is_eos || is_max_tokens;

        // Update state - mark prefill as done
        state.generated_tokens.push(next_token);
        state.is_prefill = false;

        Ok(Some((next_token, is_finished)))
    }

    /// Optimized decode processing with reduced allocations
    fn process_decode_optimized(&self, request_id: u64) -> Result<Option<(u32, bool)>> {
        let mut active = self.active_requests.lock();
        let active_req = match active.get_mut(&request_id) {
            Some(r) => r,
            None => return Ok(None),
        };

        // Check cache ownership
        let mut cache_owner = self.current_cache_owner.lock();
        let can_use_cache = *cache_owner == Some(request_id);

        // Prepare input - minimize allocations
        let seq_offset;
        let input = if can_use_cache {
            // Cache hit - single token input
            let last_token = active_req.state.generated_tokens.last()
                .or(active_req.state.request.tokens.last())
                .copied()
                .unwrap_or(0);
            seq_offset = active_req.state.request.tokens.len()
                + active_req.state.generated_tokens.len() - 1;
            Tensor::new(&[last_token], &self.device)?.unsqueeze(0)?
        } else {
            // Cache miss - full context
            seq_offset = 0;
            let prompt_len = active_req.state.request.tokens.len();
            let gen_len = active_req.state.generated_tokens.len();
            let total_len = prompt_len + gen_len;

            // Pre-allocate exact size
            let mut context = Vec::with_capacity(total_len);
            context.extend_from_slice(&active_req.state.request.tokens);
            context.extend(active_req.state.generated_tokens.iter().copied());
            Tensor::new(&context[..], &self.device)?.unsqueeze(0)?
        };

        // Forward pass
        let logits = {
            let mut model = self.model.lock();
            if !can_use_cache {
                model.clear_kv_cache();
            }
            model.forward(&input, seq_offset)?
        };

        // Update cache owner
        *cache_owner = Some(request_id);
        drop(cache_owner);

        // Extract last logits
        let logits = logits.squeeze(0)?;
        let last_logits = if logits.dims().len() > 1 {
            logits.get(logits.dims()[0] - 1)?
        } else {
            logits
        };

        // Apply repeat penalty - optimized window calculation
        let penalized_logits = if self.config.repeat_penalty != 1.0 {
            let prompt_len = active_req.state.request.tokens.len();
            let gen_len = active_req.state.generated_tokens.len();
            let total_len = prompt_len + gen_len;
            let start = total_len.saturating_sub(self.config.repeat_last_n);

            // Only allocate for the penalty window
            let penalty_tokens: Vec<u32> = if start < prompt_len {
                active_req.state.request.tokens[start..].iter()
                    .chain(active_req.state.generated_tokens.iter())
                    .copied()
                    .collect()
            } else {
                let gen_start = start - prompt_len;
                active_req.state.generated_tokens[gen_start..].iter().copied().collect()
            };

            candle_transformers::utils::apply_repeat_penalty(
                &last_logits,
                self.config.repeat_penalty,
                &penalty_tokens,
            )?
        } else {
            last_logits
        };

        // Sample
        let next_token = active_req.logits_processor.sample(&penalized_logits)?;

        // Check finish
        let is_eos = self.eos_token_id.as_ref()
            .map(|eos| eos.contains(next_token))
            .unwrap_or(false);
        let is_max_tokens = active_req.state.generated_tokens.len() + 1
            >= active_req.state.request.params.max_tokens;
        let is_finished = is_eos || is_max_tokens;

        // Update state
        active_req.state.generated_tokens.push(next_token);

        Ok(Some((next_token, is_finished)))
    }

    /// Process multiple tokens for a single request in a tight loop (EXPERIMENTAL)
    #[allow(dead_code)]
    fn process_decode_multi_token(&self, request_id: u64, max_tokens: usize) -> Result<Vec<(u32, bool)>> {
        let mut results = Vec::with_capacity(max_tokens);

        // Hold locks for the entire multi-token generation
        let mut active = self.active_requests.lock();
        let active_req = match active.get_mut(&request_id) {
            Some(r) => r,
            None => return Ok(results),
        };

        let mut cache_owner = self.current_cache_owner.lock();
        let mut model = self.model.lock();

        // Check initial cache state
        let mut can_use_cache = *cache_owner == Some(request_id);

        for _ in 0..max_tokens {
            // Check if already at max tokens
            if active_req.state.generated_tokens.len() >= active_req.state.request.params.max_tokens {
                break;
            }

            // Prepare input
            let (input_tokens, seq_offset): (Vec<u32>, usize) = if can_use_cache {
                let last_token = active_req.state.generated_tokens.last()
                    .or(active_req.state.request.tokens.last())
                    .copied()
                    .unwrap_or(0);
                let offset = active_req.state.request.tokens.len()
                    + active_req.state.generated_tokens.len() - 1;
                (vec![last_token], offset)
            } else {
                let mut context: Vec<u32> = active_req.state.request.tokens.clone();
                context.extend(active_req.state.generated_tokens.iter().copied());
                (context, 0)
            };

            // Create tensor and forward
            let input = Tensor::new(&input_tokens[..], &self.device)?.unsqueeze(0)?;

            if !can_use_cache {
                model.clear_kv_cache();
            }
            let logits = model.forward(&input, seq_offset)?;

            // Update cache state
            *cache_owner = Some(request_id);
            can_use_cache = true; // After first iteration, we own the cache

            // Extract logits
            let logits = logits.squeeze(0)?;
            let last_logits = if logits.dims().len() > 1 {
                logits.get(logits.dims()[0] - 1)?
            } else {
                logits
            };

            // Apply repeat penalty (optimized: use slice instead of collecting)
            let prompt_len = active_req.state.request.tokens.len();
            let gen_len = active_req.state.generated_tokens.len();
            let total_len = prompt_len + gen_len;
            let penalty_start = total_len.saturating_sub(self.config.repeat_last_n);

            let penalized_logits = if self.config.repeat_penalty != 1.0 && penalty_start < total_len {
                // Build penalty tokens from the relevant window
                let mut penalty_tokens = Vec::with_capacity(self.config.repeat_last_n);
                if penalty_start < prompt_len {
                    penalty_tokens.extend_from_slice(&active_req.state.request.tokens[penalty_start..]);
                    penalty_tokens.extend(active_req.state.generated_tokens.iter().copied());
                } else {
                    let gen_start = penalty_start - prompt_len;
                    penalty_tokens.extend(active_req.state.generated_tokens[gen_start..].iter().copied());
                }
                candle_transformers::utils::apply_repeat_penalty(
                    &last_logits,
                    self.config.repeat_penalty,
                    &penalty_tokens,
                )?
            } else {
                last_logits
            };

            // Sample
            let next_token = active_req.logits_processor.sample(&penalized_logits)?;

            // Check finish conditions
            let is_eos = self.eos_token_id.as_ref()
                .map(|eos| eos.contains(next_token))
                .unwrap_or(false);
            let is_max_tokens = active_req.state.generated_tokens.len() + 1
                >= active_req.state.request.params.max_tokens;
            let is_finished = is_eos || is_max_tokens;

            // Update state
            active_req.state.generated_tokens.push(next_token);
            results.push((next_token, is_finished));

            if is_finished {
                break;
            }
        }

        Ok(results)
    }

    /// Process a single decode request, leveraging KV cache if available
    #[allow(dead_code)]
    fn process_decode_single(&self, request_id: u64) -> Result<Option<(u32, bool)>> {
        let mut active = self.active_requests.lock();
        let active_req = match active.get_mut(&request_id) {
            Some(r) => r,
            None => return Ok(None),
        };

        // Check if this request owns the current cache
        let mut cache_owner = self.current_cache_owner.lock();
        let can_use_cache = *cache_owner == Some(request_id);

        let (input_tokens, seq_offset) = if can_use_cache {
            // Cache hit! Only need to process the last generated token
            let last_token = active_req.state.generated_tokens.last()
                .or(active_req.state.request.tokens.last())
                .copied()
                .unwrap_or(0);
            let offset = active_req.state.request.tokens.len()
                + active_req.state.generated_tokens.len() - 1;
            (vec![last_token], offset)
        } else {
            // Cache miss - need to replay full context
            let mut context: Vec<u32> = active_req.state.request.tokens.clone();
            context.extend(active_req.state.generated_tokens.iter().copied());
            (context, 0)
        };

        // Create input tensor
        let input = Tensor::new(&input_tokens[..], &self.device)?.unsqueeze(0)?;

        // Forward pass
        let logits = {
            let mut model = self.model.lock();
            if !can_use_cache {
                model.clear_kv_cache();
            }
            model.forward(&input, seq_offset)?
        };

        // Update cache owner
        *cache_owner = Some(request_id);
        drop(cache_owner);

        // Get last token logits
        let logits = logits.squeeze(0)?;
        let last_logits = if logits.dims().len() > 1 {
            logits.get(logits.dims()[0] - 1)?
        } else {
            logits
        };

        // Apply repeat penalty
        let all_tokens: Vec<u32> = active_req.state.request.tokens.iter()
            .chain(active_req.state.generated_tokens.iter())
            .copied()
            .collect();

        let penalized_logits = if self.config.repeat_penalty != 1.0 {
            let start_at = all_tokens.len().saturating_sub(self.config.repeat_last_n);
            candle_transformers::utils::apply_repeat_penalty(
                &last_logits,
                self.config.repeat_penalty,
                &all_tokens[start_at..],
            )?
        } else {
            last_logits
        };

        // Sample next token
        let next_token = active_req.logits_processor.sample(&penalized_logits)?;

        // Check if finished
        let is_eos = self.eos_token_id.as_ref()
            .map(|eos| eos.contains(next_token))
            .unwrap_or(false);
        let is_max_tokens = active_req.state.generated_tokens.len() + 1
            >= active_req.state.request.params.max_tokens;
        let is_finished = is_eos || is_max_tokens;

        // Update state
        active_req.state.generated_tokens.push(next_token);

        Ok(Some((next_token, is_finished)))
    }

    /// Process decode requests sequentially, using KV cache for efficiency
    /// When processing the same request consecutively, we can use the cached KV state
    #[allow(dead_code)]
    fn process_decode_sequential(&self, request_ids: &[u64]) -> Result<Vec<(u64, u32, bool)>> {
        let mut results = Vec::with_capacity(request_ids.len());

        for &request_id in request_ids {
            let mut active = self.active_requests.lock();
            let active_req = match active.get_mut(&request_id) {
                Some(r) => r,
                None => continue,
            };

            // Check if this request owns the current cache
            let mut cache_owner = self.current_cache_owner.lock();
            let can_use_cache = *cache_owner == Some(request_id);

            let (input_tokens, seq_offset) = if can_use_cache {
                // Cache is valid - only need to process the last generated token
                let last_token = active_req.state.generated_tokens.last()
                    .or(active_req.state.request.tokens.last())
                    .copied()
                    .unwrap_or(0);
                let offset = active_req.state.request.tokens.len()
                    + active_req.state.generated_tokens.len() - 1;
                (vec![last_token], offset)
            } else {
                // Cache miss - need to replay full context
                let mut context: Vec<u32> = active_req.state.request.tokens.clone();
                context.extend(active_req.state.generated_tokens.iter().copied());
                (context, 0)
            };

            // Create input tensor
            let input = Tensor::new(&input_tokens[..], &self.device)?.unsqueeze(0)?;

            // Forward pass
            let logits = {
                let mut model = self.model.lock();
                if !can_use_cache {
                    model.clear_kv_cache();
                }
                model.forward(&input, seq_offset)?
            };

            // Update cache owner
            *cache_owner = Some(request_id);
            drop(cache_owner);

            // Get last token logits
            let logits = logits.squeeze(0)?;
            let last_logits = if logits.dims().len() > 1 {
                logits.get(logits.dims()[0] - 1)?
            } else {
                logits
            };

            // Apply repeat penalty
            let all_tokens: Vec<u32> = active_req.state.request.tokens.iter()
                .chain(active_req.state.generated_tokens.iter())
                .copied()
                .collect();

            let penalized_logits = if self.config.repeat_penalty != 1.0 {
                let start_at = all_tokens.len().saturating_sub(self.config.repeat_last_n);
                candle_transformers::utils::apply_repeat_penalty(
                    &last_logits,
                    self.config.repeat_penalty,
                    &all_tokens[start_at..],
                )?
            } else {
                last_logits
            };

            // Sample next token
            let next_token = active_req.logits_processor.sample(&penalized_logits)?;

            // Check if finished
            let is_eos = self.eos_token_id.as_ref()
                .map(|eos| eos.contains(next_token))
                .unwrap_or(false);
            let is_max_tokens = active_req.state.generated_tokens.len() + 1
                >= active_req.state.request.params.max_tokens;
            let is_finished = is_eos || is_max_tokens;

            // Update state
            active_req.state.generated_tokens.push(next_token);

            results.push((request_id, next_token, is_finished));
        }

        Ok(results)
    }

    /// Process a single decode request with full context replay
    /// This maintains output quality by providing the full history
    #[allow(dead_code)]
    fn process_decode_request(&self, request_id: u64) -> Result<Option<(u32, bool)>> {
        let mut active = self.active_requests.lock();
        let active_req = match active.get_mut(&request_id) {
            Some(r) => r,
            None => return Ok(None),
        };

        // Build full context: prompt + all generated tokens so far
        let mut context: Vec<u32> = active_req.state.request.tokens.clone();
        context.extend(active_req.state.generated_tokens.iter().copied());

        // Create input tensor [1, seq_len]
        let input = Tensor::new(&context[..], &self.device)?.unsqueeze(0)?;

        // Forward pass
        let logits = {
            let mut model = self.model.lock();
            model.clear_kv_cache();
            model.forward(&input, 0)?
        };

        // Get last token logits
        let logits = logits.squeeze(0)?;
        let last_logits = if logits.dims().len() > 1 {
            logits.get(logits.dims()[0] - 1)?
        } else {
            logits
        };

        // Apply repeat penalty
        let penalized_logits = if self.config.repeat_penalty != 1.0 {
            let start_at = context.len().saturating_sub(self.config.repeat_last_n);
            candle_transformers::utils::apply_repeat_penalty(
                &last_logits,
                self.config.repeat_penalty,
                &context[start_at..],
            )?
        } else {
            last_logits
        };

        // Sample next token
        let next_token = active_req.logits_processor.sample(&penalized_logits)?;

        // Check if finished
        let is_eos = self.eos_token_id.as_ref()
            .map(|eos| eos.contains(next_token))
            .unwrap_or(false);
        let is_max_tokens = active_req.state.generated_tokens.len() + 1
            >= active_req.state.request.params.max_tokens;
        let is_finished = is_eos || is_max_tokens;

        // Update state
        active_req.state.generated_tokens.push(next_token);

        Ok(Some((next_token, is_finished)))
    }

    /// Process a batch of decode requests (EXPERIMENTAL - may have quality issues)
    /// Each request needs its full context (prompt + generated tokens) for quality
    #[allow(dead_code)]
    fn process_decode_batch(&self, request_ids: &[u64]) -> Result<Vec<(u64, u32, bool)>> {
        let batch_size = request_ids.len();
        if batch_size == 0 {
            return Ok(vec![]);
        }

        // Collect full context for each request (prompt + all generated tokens)
        let mut batch_contexts: Vec<Vec<u32>> = Vec::with_capacity(batch_size);
        let mut request_order = Vec::with_capacity(batch_size);

        {
            let active = self.active_requests.lock();
            for &request_id in request_ids {
                if let Some(active_req) = active.get(&request_id) {
                    // Full context = prompt + all generated tokens so far
                    let mut context: Vec<u32> = active_req.state.request.tokens.clone();
                    context.extend(active_req.state.generated_tokens.iter().copied());
                    batch_contexts.push(context);
                    request_order.push(request_id);
                }
            }
        }

        if batch_contexts.is_empty() {
            return Ok(vec![]);
        }

        // Find max context length and pad
        let max_len = batch_contexts.iter().map(|c| c.len()).max().unwrap_or(1);

        // Create padded batch tensor
        let mut padded_batch = Vec::with_capacity(batch_size * max_len);
        for context in &batch_contexts {
            padded_batch.extend_from_slice(context);
            // Pad remaining with zeros
            padded_batch.resize(padded_batch.len() + (max_len - context.len()), 0);
        }

        // Create batched input tensor [batch_size, max_len]
        let input = Tensor::new(&padded_batch[..], &self.device)?
            .reshape(&[batch_size, max_len])?;

        // Forward pass with full context
        let logits = {
            let mut model = self.model.lock();
            model.clear_kv_cache();
            model.forward(&input, 0)?
        };

        // Process results for each request
        let mut results = Vec::with_capacity(batch_size);
        let mut active = self.active_requests.lock();

        for (batch_idx, &request_id) in request_order.iter().enumerate() {
            if let Some(active_req) = active.get_mut(&request_id) {
                // Extract logits for this batch element
                let req_logits = logits.get(batch_idx)?;

                // Get logits at the last actual token position (not padding)
                let context_len = batch_contexts[batch_idx].len();
                let last_logits = if req_logits.dims().len() > 1 {
                    // Shape is [seq_len, vocab_size], get last real position
                    req_logits.get(context_len - 1)?
                } else {
                    req_logits
                };

                // Apply repeat penalty
                let penalized_logits = if self.config.repeat_penalty != 1.0 {
                    let all_tokens: Vec<u32> = active_req.state.request.tokens.iter()
                        .chain(active_req.state.generated_tokens.iter())
                        .copied()
                        .collect();
                    let start_at = all_tokens.len().saturating_sub(self.config.repeat_last_n);
                    candle_transformers::utils::apply_repeat_penalty(
                        &last_logits,
                        self.config.repeat_penalty,
                        &all_tokens[start_at..],
                    )?
                } else {
                    last_logits
                };

                // Sample next token
                let next_token = active_req.logits_processor.sample(&penalized_logits)?;

                // Check if finished
                let is_eos = self.eos_token_id.as_ref()
                    .map(|eos| eos.contains(next_token))
                    .unwrap_or(false);
                let is_max_tokens = active_req.state.generated_tokens.len() + 1
                    >= active_req.state.request.params.max_tokens;
                let is_finished = is_eos || is_max_tokens;

                // Update state
                active_req.state.generated_tokens.push(next_token);

                results.push((request_id, next_token, is_finished));
            }
        }

        Ok(results)
    }

    /// Process a single request and return the generated token
    fn process_single_request(&self, request_id: u64) -> Result<Option<(u32, bool)>> {
        let mut active = self.active_requests.lock();
        let active_req = match active.get_mut(&request_id) {
            Some(r) => r,
            None => return Ok(None),
        };

        let state = &mut active_req.state;

        // Determine what tokens to process
        let (input_tokens, seq_offset) = if state.is_prefill {
            // Process all prompt tokens during prefill
            let tokens = &state.request.tokens[..];
            if tokens.is_empty() {
                return Err(anyhow!("Empty prompt tokens"));
            }
            (tokens.to_vec(), 0)
        } else {
            // Decode phase - process last generated token
            let last_token = state.generated_tokens.last()
                .or(state.request.tokens.last())
                .copied()
                .unwrap_or(0);
            let offset = state.request.tokens.len() + state.generated_tokens.len() - 1;
            (vec![last_token], offset)
        };

        // Forward pass
        let input = Tensor::new(&input_tokens[..], &self.device)?.unsqueeze(0)?;

        let logits = {
            let mut model = self.model.lock();
            let mut cache_owner = self.current_cache_owner.lock();

            // Check if we need to invalidate the cache
            let need_cache_clear = match *cache_owner {
                None => state.is_prefill,  // First request
                Some(owner_id) => owner_id != request_id,  // Different request
            };

            if need_cache_clear {
                model.clear_kv_cache();
                // If switching to a non-prefill request, we need to replay its history
                // For now, we only support processing requests sequentially
            }

            *cache_owner = Some(request_id);
            drop(cache_owner);

            model.forward(&input, seq_offset)?
        };

        // Squeeze dimensions
        let logits = logits.squeeze(0)?;
        let logits = if logits.dims().len() > 1 {
            logits.squeeze(0)?
        } else {
            logits
        };

        // Apply repeat penalty
        let all_tokens: Vec<u32> = state.request.tokens.iter()
            .chain(state.generated_tokens.iter())
            .copied()
            .collect();

        let logits = if self.config.repeat_penalty != 1.0 {
            let start_at = all_tokens.len().saturating_sub(self.config.repeat_last_n);
            candle_transformers::utils::apply_repeat_penalty(
                &logits,
                self.config.repeat_penalty,
                &all_tokens[start_at..],
            )?
        } else {
            logits
        };

        // Sample next token
        let next_token = active_req.logits_processor.sample(&logits)?;

        // Check for finish conditions
        let is_eos = self.eos_token_id.as_ref()
            .map(|eos| eos.contains(next_token))
            .unwrap_or(false);
        let is_stop_token = state.request.params.stop_tokens.contains(&next_token);
        let is_max_tokens = state.generated_tokens.len() + 1 >= state.request.params.max_tokens;

        let is_finished = is_eos || is_stop_token || is_max_tokens;

        // If we just finished prefill, mark it
        if state.is_prefill {
            state.is_prefill = false;
        }

        // Add generated token
        state.generated_tokens.push(next_token);

        // If finished, clean up
        if is_finished {
            let mut stats = self.stats.lock();
            stats.completed_requests += 1;

            // Remove from active requests
            drop(active);  // Release lock first
            self.active_requests.lock().remove(&request_id);
        }

        Ok(Some((next_token, is_finished)))
    }

    /// Run continuous batching loop until all requests are done
    pub fn run_to_completion(&self) -> Result<()> {
        loop {
            let results = self.step()?;

            if results.is_empty() {
                // Check if there are any pending or running requests
                let scheduler_stats = self.scheduler.stats();
                if scheduler_stats.pending_requests == 0 && scheduler_stats.running_requests == 0 {
                    break;
                }

                // Brief sleep to avoid busy waiting
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
        }

        Ok(())
    }

    /// Get the generated text for a completed request
    pub fn get_generated_text(&self, request_id: u64) -> Result<Option<String>> {
        let active = self.active_requests.lock();

        if let Some(req) = active.get(&request_id) {
            let text = self.tokenizer.decode(&req.state.generated_tokens, true)
                .map_err(|e| anyhow!("Decode error: {}", e))?;
            Ok(Some(text))
        } else {
            Ok(None)
        }
    }

    /// Encode a prompt to tokens
    pub fn encode(&self, prompt: &str) -> Result<Vec<u32>> {
        self.tokenizer
            .encode(prompt, true)
            .map_err(|e| anyhow!("Tokenization error: {}", e))
            .map(|enc| enc.get_ids().to_vec())
    }

    /// Decode tokens to text
    pub fn decode(&self, tokens: &[u32]) -> Result<String> {
        self.tokenizer
            .decode(tokens, true)
            .map_err(|e| anyhow!("Decode error: {}", e))
    }

    /// Get engine statistics
    pub fn stats(&self) -> BatchEngineStats {
        self.stats.lock().clone()
    }

    /// Get scheduler statistics
    pub fn scheduler_stats(&self) -> crate::scheduler::SchedulerStats {
        self.scheduler.stats()
    }

    /// Get KV cache statistics
    pub fn cache_stats(&self) -> crate::kv_cache::CacheStats {
        self.kv_cache.lock().stats()
    }
}
