use crate::request::{RequestState, InferenceRequest};
use crate::kv_cache::{PagedKVCache, PAGE_SIZE};
use parking_lot::Mutex;
use std::collections::VecDeque;
use std::sync::Arc;
use tracing::{debug, info};

/// Maximum batch size for inference
const MAX_BATCH_SIZE: usize = 256;

/// Maximum tokens to prefill in one step (chunked prefill)
const MAX_PREFILL_TOKENS: usize = 2048;

/// Request scheduler with continuous batching
pub struct Scheduler {
    /// Queue of pending requests waiting to be scheduled
    pending_queue: Arc<Mutex<VecDeque<InferenceRequest>>>,
    /// Currently running requests
    running_requests: Arc<Mutex<Vec<RequestState>>>,
    /// Reference to KV cache for page allocation
    kv_cache: Arc<Mutex<PagedKVCache>>,
}

impl Scheduler {
    pub fn new(kv_cache: Arc<Mutex<PagedKVCache>>) -> Self {
        Self {
            pending_queue: Arc::new(Mutex::new(VecDeque::new())),
            running_requests: Arc::new(Mutex::new(Vec::new())),
            kv_cache,
        }
    }

    /// Add a new request to the pending queue
    pub fn add_request(&self, request: InferenceRequest) {
        debug!("Adding request {} to queue", request.id);
        self.pending_queue.lock().push_back(request);
    }

    /// Get the next batch of requests to process
    /// Returns (batch, total_tokens, prefill_tokens)
    pub fn get_next_batch(&self) -> Option<ScheduledBatch> {
        let mut running = self.running_requests.lock();
        let mut pending = self.pending_queue.lock();
        let kv_cache = self.kv_cache.lock();

        // Remove finished requests and free their pages
        running.retain(|req| {
            if req.is_finished() {
                kv_cache.free_pages(&req.kv_cache_blocks);
                info!("Request {} finished, freed {} pages", 
                      req.request.id, req.kv_cache_blocks.len());
                false
            } else {
                true
            }
        });

        let mut batch_requests = Vec::new();
        let mut total_decode_tokens = 0;
        let mut total_prefill_tokens = 0;

        // First, add running requests (decode phase)
        for req in running.iter() {
            if !req.is_prefill {
                batch_requests.push(req.request.id);
                total_decode_tokens += 1;
            }
        }

        // Try to add new requests or continue prefilling
        while batch_requests.len() < MAX_BATCH_SIZE {
            // Try to continue prefilling existing requests
            let mut added_prefill = false;
            for req in running.iter_mut() {
                if req.is_prefill && total_prefill_tokens < MAX_PREFILL_TOKENS {
                    let remaining = req.request.tokens.len() - req.prefill_offset;
                    let chunk_size = remaining.min(MAX_PREFILL_TOKENS - total_prefill_tokens);
                    
                    if chunk_size > 0 {
                        total_prefill_tokens += chunk_size;
                        req.prefill_offset += chunk_size;
                        
                        // Allocate pages if needed
                        let pages_needed = (req.prefill_offset + PAGE_SIZE - 1) / PAGE_SIZE;
                        while req.kv_cache_blocks.len() < pages_needed {
                            if let Some(page) = kv_cache.allocate_page() {
                                req.kv_cache_blocks.push(page);
                            } else {
                                // Out of memory, stop scheduling
                                return self.create_batch(
                                    &running,
                                    batch_requests,
                                    total_decode_tokens,
                                    total_prefill_tokens,
                                );
                            }
                        }

                        // Check if prefill is complete
                        if req.prefill_offset >= req.request.tokens.len() {
                            req.is_prefill = false;
                            debug!("Request {} prefill complete", req.request.id);
                        }

                        added_prefill = true;
                        break;
                    }
                }
            }

            if added_prefill {
                continue;
            }

            // Try to add a new request from pending queue
            if let Some(new_req) = pending.pop_front() {
                if kv_cache.num_free_pages() > 0 {
                    let mut req_state = RequestState::new(new_req.clone());
                    
                    // Start prefilling
                    let chunk_size = new_req.tokens.len().min(MAX_PREFILL_TOKENS - total_prefill_tokens);
                    if chunk_size > 0 {
                        req_state.prefill_offset = chunk_size;
                        total_prefill_tokens += chunk_size;

                        // Allocate initial pages
                        let pages_needed = (chunk_size + PAGE_SIZE - 1) / PAGE_SIZE;
                        for _ in 0..pages_needed {
                            if let Some(page) = kv_cache.allocate_page() {
                                req_state.kv_cache_blocks.push(page);
                            } else {
                                // Out of memory, put back and stop
                                pending.push_front(new_req);
                                return self.create_batch(
                                    &running,
                                    batch_requests,
                                    total_decode_tokens,
                                    total_prefill_tokens,
                                );
                            }
                        }

                        if req_state.prefill_offset >= new_req.tokens.len() {
                            req_state.is_prefill = false;
                        }

                        batch_requests.push(req_state.request.id);
                        running.push(req_state);
                        debug!("Added new request {} to batch", new_req.id);
                    }
                } else {
                    // No memory available, put back
                    pending.push_front(new_req);
                    break;
                }
            } else {
                // No more pending requests
                break;
            }
        }

        if batch_requests.is_empty() {
            None
        } else {
            self.create_batch(&running, batch_requests, total_decode_tokens, total_prefill_tokens)
        }
    }

    fn create_batch(
        &self,
        _running: &[RequestState],
        request_ids: Vec<u64>,
        decode_tokens: usize,
        prefill_tokens: usize,
    ) -> Option<ScheduledBatch> {
        if request_ids.is_empty() {
            return None;
        }

        Some(ScheduledBatch {
            request_ids,
            total_decode_tokens: decode_tokens,
            total_prefill_tokens: prefill_tokens,
        })
    }

    /// Update request state after inference step
    pub fn update_requests(&self, generated_tokens: Vec<(u64, u32)>) {
        let mut running = self.running_requests.lock();
        let kv_cache = self.kv_cache.lock();

        for (req_id, token) in generated_tokens {
            if let Some(req) = running.iter_mut().find(|r| r.request.id == req_id) {
                req.generated_tokens.push(token);
                
                // Allocate new page if needed
                let total_tokens = req.total_tokens();
                let pages_needed = (total_tokens + PAGE_SIZE - 1) / PAGE_SIZE;
                while req.kv_cache_blocks.len() < pages_needed {
                    if let Some(page) = kv_cache.allocate_page() {
                        req.kv_cache_blocks.push(page);
                    } else {
                        // Out of memory - this shouldn't happen if scheduler is correct
                        tracing::error!("OOM during generation for request {}", req_id);
                        break;
                    }
                }
            }
        }
    }

    /// Get statistics
    pub fn stats(&self) -> SchedulerStats {
        let running = self.running_requests.lock();
        let pending = self.pending_queue.lock();

        SchedulerStats {
            pending_requests: pending.len(),
            running_requests: running.len(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ScheduledBatch {
    pub request_ids: Vec<u64>,
    pub total_decode_tokens: usize,
    pub total_prefill_tokens: usize,
}

#[derive(Debug, Clone)]
pub struct SchedulerStats {
    pub pending_requests: usize,
    pub running_requests: usize,
}
