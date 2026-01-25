//! Forge Core - High-performance LLM inference engine
//!
//! This crate implements the core inference logic including:
//! - Multi-architecture model support (LLaMA, Qwen2, Qwen3) via candle-transformers
//! - Model loading from HuggingFace Hub
//! - Streaming text generation
//! - Temperature, top-p, top-k sampling
//! - Continuous batching for high throughput

pub mod request;
pub mod kv_cache;
pub mod scheduler;
pub mod batcher;
pub mod model;
pub mod engine;
pub mod batch_engine;

pub use request::{InferenceRequest, InferenceResponse, SamplingParams};
pub use request::FinishReason as RequestFinishReason;
pub use kv_cache::{PagedKVCache, CacheStats, PAGE_SIZE};
pub use scheduler::{Scheduler, ScheduledBatch, SchedulerStats};
pub use batcher::Sampler;
pub use model::{ModelConfig, LoadedModel, load_model, TokenOutputStream, ModelArch, CausalLM, EosToken};
pub use engine::{InferenceEngine, EngineConfig, EngineStats, GenerationResult, FinishReason};
pub use batch_engine::{BatchInferenceEngine, BatchEngineConfig, BatchEngineStats};
