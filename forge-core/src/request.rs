use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::oneshot;

/// Unique identifier for a request
pub type RequestId = u64;

/// Request for inference
#[derive(Debug, Clone)]
pub struct InferenceRequest {
    pub id: RequestId,
    pub tokens: Vec<u32>,
    pub params: SamplingParams,
    pub response_tx: Arc<tokio::sync::Mutex<Option<oneshot::Sender<InferenceResponse>>>>,
}

/// Sampling parameters for generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SamplingParams {
    pub max_tokens: usize,
    pub temperature: f64,
    pub top_p: Option<f64>,
    pub top_k: Option<usize>,
    pub stop_tokens: Vec<u32>,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            max_tokens: 256,
            temperature: 0.7,
            top_p: Some(0.9),
            top_k: None,
            stop_tokens: vec![],
        }
    }
}

/// Response from inference
#[derive(Debug, Clone)]
pub struct InferenceResponse {
    pub id: RequestId,
    pub tokens: Vec<u32>,
    pub finished: bool,
    pub finish_reason: Option<FinishReason>,
}

/// Reason for completion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FinishReason {
    Length,
    StopToken,
    Error(String),
}

/// State of a request during batched inference
#[derive(Debug)]
pub struct RequestState {
    pub request: InferenceRequest,
    pub generated_tokens: Vec<u32>,
    pub is_prefill: bool,
    pub prefill_offset: usize,
    pub kv_cache_blocks: Vec<usize>,
}

impl RequestState {
    pub fn new(request: InferenceRequest) -> Self {
        Self {
            request,
            generated_tokens: Vec::new(),
            is_prefill: true,
            prefill_offset: 0,
            kv_cache_blocks: Vec::new(),
        }
    }

    pub fn total_tokens(&self) -> usize {
        self.request.tokens.len() + self.generated_tokens.len()
    }

    pub fn is_finished(&self) -> bool {
        self.generated_tokens.len() >= self.request.params.max_tokens
            || self.generated_tokens.last()
                .map(|t| self.request.params.stop_tokens.contains(t))
                .unwrap_or(false)
    }
}
