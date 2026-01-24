//! Forge Core - High-performance LLM inference engine
//!
//! This crate implements the core inference logic including:
//! - Real LLaMA model inference via candle-transformers
//! - Model loading from HuggingFace Hub
//! - Streaming text generation
//! - Temperature, top-p, top-k sampling

pub mod request;
pub mod kv_cache;
pub mod scheduler;
pub mod batcher;
pub mod model;
pub mod engine;

pub use request::{InferenceRequest, InferenceResponse, SamplingParams};
pub use request::FinishReason as RequestFinishReason;
pub use kv_cache::{PagedKVCache, CacheStats, PAGE_SIZE};
pub use scheduler::{Scheduler, ScheduledBatch, SchedulerStats};
pub use batcher::Sampler;
pub use model::{ModelConfig, LoadedModel, load_model, TokenOutputStream};
pub use engine::{InferenceEngine, EngineConfig, EngineStats, GenerationResult, FinishReason};
