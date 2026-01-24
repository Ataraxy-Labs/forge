use forge_core::*;
use candle_core::{Device, DType};
use std::sync::Arc;
use tokio::sync::{oneshot, Mutex};
use parking_lot::Mutex as ParkingMutex;

#[test]
fn test_request_creation() {
    let (tx, _rx) = oneshot::channel();
    
    let request = InferenceRequest {
        id: 1,
        tokens: vec![1, 2, 3, 4, 5],
        params: SamplingParams::default(),
        response_tx: Arc::new(Mutex::new(Some(tx))),
    };
    
    assert_eq!(request.id, 1);
    assert_eq!(request.tokens.len(), 5);
}

#[test]
fn test_request_state() {
    let (tx, _rx) = oneshot::channel();
    
    let request = InferenceRequest {
        id: 1,
        tokens: vec![1, 2, 3],
        params: SamplingParams {
            max_tokens: 10,
            ..Default::default()
        },
        response_tx: Arc::new(Mutex::new(Some(tx))),
    };
    
    let mut state = request::RequestState::new(request);
    
    assert_eq!(state.total_tokens(), 3);
    assert!(state.is_prefill);
    assert_eq!(state.generated_tokens.len(), 0);
    
    state.generated_tokens.push(100);
    state.generated_tokens.push(101);
    
    assert_eq!(state.total_tokens(), 5);
    assert_eq!(state.generated_tokens.len(), 2);
}

#[test]
fn test_request_finish_max_tokens() {
    let (tx, _rx) = oneshot::channel();
    
    let request = InferenceRequest {
        id: 1,
        tokens: vec![1],
        params: SamplingParams {
            max_tokens: 3,
            ..Default::default()
        },
        response_tx: Arc::new(Mutex::new(Some(tx))),
    };
    
    let mut state = request::RequestState::new(request);
    assert!(!state.is_finished());
    
    state.generated_tokens.push(10);
    assert!(!state.is_finished());
    
    state.generated_tokens.push(11);
    assert!(!state.is_finished());
    
    state.generated_tokens.push(12);
    assert!(state.is_finished());
}

#[test]
fn test_request_stop_token() {
    let (tx, _rx) = oneshot::channel();
    
    let request = InferenceRequest {
        id: 1,
        tokens: vec![1],
        params: SamplingParams {
            max_tokens: 100,
            stop_tokens: vec![999],
            ..Default::default()
        },
        response_tx: Arc::new(Mutex::new(Some(tx))),
    };
    
    let mut state = request::RequestState::new(request);
    
    state.generated_tokens.push(10);
    assert!(!state.is_finished());
    
    state.generated_tokens.push(999);
    assert!(state.is_finished());
}

#[test]
fn test_kv_cache_creation() {
    let device = Device::Cpu;
    
    let cache = PagedKVCache::new(
        32,
        32,
        128,
        1024,
        DType::F16,
        device,
    ).unwrap();
    
    let stats = cache.stats();
    assert_eq!(stats.total_pages, 1024);
    assert_eq!(stats.free_pages, 1024);
    assert_eq!(stats.used_pages, 0);
}

#[test]
fn test_kv_cache_allocation() {
    let device = Device::Cpu;
    
    let cache = PagedKVCache::new(
        32, 32, 128, 10, DType::F16, device
    ).unwrap();
    
    let page1 = cache.allocate_page();
    assert!(page1.is_some());
    assert_eq!(cache.num_free_pages(), 9);
    
    let page2 = cache.allocate_page();
    assert!(page2.is_some());
    assert_eq!(cache.num_free_pages(), 8);
    
    cache.free_page(page1.unwrap());
    assert_eq!(cache.num_free_pages(), 9);
}

#[test]
fn test_kv_cache_exhaustion() {
    let device = Device::Cpu;
    
    let cache = PagedKVCache::new(
        32, 32, 128, 3, DType::F16, device
    ).unwrap();
    
    let p1 = cache.allocate_page();
    let p2 = cache.allocate_page();
    let p3 = cache.allocate_page();
    
    assert!(p1.is_some());
    assert!(p2.is_some());
    assert!(p3.is_some());
    assert_eq!(cache.num_free_pages(), 0);
    
    let p4 = cache.allocate_page();
    assert!(p4.is_none());
}

#[test]
fn test_scheduler_creation() {
    let device = Device::Cpu;
    let cache = Arc::new(ParkingMutex::new(
        PagedKVCache::new(32, 32, 128, 1024, DType::F16, device).unwrap()
    ));
    
    let scheduler = Scheduler::new(cache);
    let stats = scheduler.stats();
    
    assert_eq!(stats.pending_requests, 0);
    assert_eq!(stats.running_requests, 0);
}

#[test]
fn test_scheduler_add_request() {
    let device = Device::Cpu;
    let cache = Arc::new(ParkingMutex::new(
        PagedKVCache::new(32, 32, 128, 1024, DType::F16, device).unwrap()
    ));
    
    let scheduler = Scheduler::new(cache);
    
    let (tx, _rx) = oneshot::channel();
    let request = InferenceRequest {
        id: 1,
        tokens: vec![1, 2, 3, 4, 5],
        params: SamplingParams::default(),
        response_tx: Arc::new(Mutex::new(Some(tx))),
    };
    
    scheduler.add_request(request);
    
    let stats = scheduler.stats();
    assert_eq!(stats.pending_requests, 1);
}

#[test]
fn test_page_size_constant() {
    assert_eq!(PAGE_SIZE, 16);
}

#[test]
fn test_sampling_params_defaults() {
    let params = SamplingParams::default();
    
    assert_eq!(params.max_tokens, 256);
    assert_eq!(params.temperature, 0.7);
    assert_eq!(params.top_p, Some(0.9));
    assert_eq!(params.top_k, None);
    assert_eq!(params.stop_tokens.len(), 0);
}
