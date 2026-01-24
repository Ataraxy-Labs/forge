//! Forge Core - High-performance LLM inference engine
//!
//! This crate implements the core inference logic including:
//! - Paged KV cache for memory efficiency
//! - Continuous batching with chunked prefill
//! - Request scheduling and management
//! - Inference engine orchestration

pub mod request;
pub mod kv_cache;
pub mod scheduler;
pub mod batcher;
pub mod engine;

pub use request::{InferenceRequest, InferenceResponse, SamplingParams, FinishReason};
pub use kv_cache::{PagedKVCache, CacheStats, PAGE_SIZE};
pub use scheduler::{Scheduler, ScheduledBatch, SchedulerStats};
