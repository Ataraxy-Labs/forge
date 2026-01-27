//! Forge Server - OpenAI-compatible HTTP API for the Forge inference engine

use axum::{
    extract::State,
    http::StatusCode,
    routing::{get, post},
    Json, Router,
};
use forge_core::{InferenceEngine, SamplingParams, EngineStats, EngineConfig, ModelConfig, FinishReason};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tower_http::cors::CorsLayer;
use tracing::{info, Level};
use tracing_subscriber;

#[derive(Clone)]
struct AppState {
    engine: Arc<InferenceEngine>,
}

#[derive(Debug, Deserialize)]
struct CompletionRequest {
    prompt: String,
    #[serde(default = "default_max_tokens")]
    max_tokens: usize,
    #[serde(default = "default_temperature")]
    temperature: f64,
    #[serde(default)]
    top_p: Option<f64>,
    #[serde(default)]
    top_k: Option<usize>,
    #[serde(default)]
    #[allow(dead_code)]
    stop: Vec<String>,
    #[serde(default)]
    seed: Option<u64>,
    #[serde(default)]
    #[allow(dead_code)]
    stream: bool,
}

fn default_max_tokens() -> usize { 100 }
fn default_temperature() -> f64 { 0.7 }

#[derive(Debug, Serialize)]
struct CompletionResponse {
    id: String,
    object: String,
    created: u64,
    model: String,
    choices: Vec<Choice>,
    usage: Usage,
}

#[derive(Debug, Serialize)]
struct Choice {
    index: usize,
    text: String,
    finish_reason: String,
}

#[derive(Debug, Serialize)]
struct Usage {
    prompt_tokens: usize,
    completion_tokens: usize,
    total_tokens: usize,
}

#[derive(Debug, Serialize)]
struct HealthResponse {
    status: String,
    engine: EngineStats,
}

#[derive(Debug, Serialize)]
struct ErrorResponse {
    error: String,
}

async fn health_check(State(state): State<AppState>) -> Json<HealthResponse> {
    let stats = state.engine.stats();
    Json(HealthResponse {
        status: "ok".to_string(),
        engine: stats,
    })
}

async fn completions(
    State(state): State<AppState>,
    Json(req): Json<CompletionRequest>,
) -> Result<Json<CompletionResponse>, (StatusCode, Json<ErrorResponse>)> {
    info!("Completion request: prompt='{}' max_tokens={}",
          if req.prompt.len() > 50 { &req.prompt[..50] } else { &req.prompt },
          req.max_tokens);

    let params = SamplingParams {
        max_tokens: req.max_tokens,
        temperature: req.temperature,
        top_p: req.top_p,
        top_k: req.top_k,
        stop_tokens: vec![], // Would need tokenizer to convert stop strings to tokens
        seed: req.seed,
    };

    // Generate
    let result = state.engine.generate(&req.prompt, &params)
        .map_err(|e| {
            (StatusCode::INTERNAL_SERVER_ERROR, Json(ErrorResponse {
                error: e.to_string(),
            }))
        })?;

    let finish_reason = match result.finish_reason {
        FinishReason::Stop => "stop",
        FinishReason::Length => "length",
        FinishReason::Error => "error",
    };

    let response = CompletionResponse {
        id: format!("cmpl-{}", uuid_simple()),
        object: "text_completion".to_string(),
        created: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs(),
        model: state.engine.stats().model_id,
        choices: vec![Choice {
            index: 0,
            text: result.text,
            finish_reason: finish_reason.to_string(),
        }],
        usage: Usage {
            prompt_tokens: result.prompt_tokens,
            completion_tokens: result.generated_tokens,
            total_tokens: result.prompt_tokens + result.generated_tokens,
        },
    };

    info!("Generated {} tokens at {:.2} tok/s",
          result.generated_tokens,
          result.tokens_per_second);

    Ok(Json(response))
}

fn uuid_simple() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap();
    format!("{:x}{:x}", now.as_secs(), now.subsec_nanos())
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct ModelRequest {
    model_id: Option<String>,
}

async fn list_models(State(state): State<AppState>) -> Json<serde_json::Value> {
    let stats = state.engine.stats();

    // All supported models
    let available_models = vec![
        // Currently loaded (marked as ready)
        serde_json::json!({
            "id": stats.model_id,
            "object": "model",
            "owned_by": "forge",
            "ready": true,
            "permission": []
        }),
        // LLaMA family
        serde_json::json!({
            "id": "HuggingFaceTB/SmolLM2-135M",
            "object": "model",
            "owned_by": "huggingface",
            "ready": stats.model_id == "HuggingFaceTB/SmolLM2-135M",
            "permission": []
        }),
        serde_json::json!({
            "id": "HuggingFaceTB/SmolLM2-360M",
            "object": "model",
            "owned_by": "huggingface",
            "ready": stats.model_id == "HuggingFaceTB/SmolLM2-360M",
            "permission": []
        }),
        serde_json::json!({
            "id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "object": "model",
            "owned_by": "huggingface",
            "ready": stats.model_id == "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "permission": []
        }),
        // Qwen2 family
        serde_json::json!({
            "id": "Qwen/Qwen2-0.5B",
            "object": "model",
            "owned_by": "qwen",
            "ready": stats.model_id == "Qwen/Qwen2-0.5B",
            "permission": []
        }),
        serde_json::json!({
            "id": "Qwen/Qwen2-1.5B",
            "object": "model",
            "owned_by": "qwen",
            "ready": stats.model_id == "Qwen/Qwen2-1.5B",
            "permission": []
        }),
        serde_json::json!({
            "id": "Qwen/Qwen2-7B",
            "object": "model",
            "owned_by": "qwen",
            "ready": stats.model_id == "Qwen/Qwen2-7B",
            "permission": []
        }),
        // Qwen3 family
        serde_json::json!({
            "id": "Qwen/Qwen3-0.6B",
            "object": "model",
            "owned_by": "qwen",
            "ready": stats.model_id == "Qwen/Qwen3-0.6B",
            "permission": []
        }),
        serde_json::json!({
            "id": "Qwen/Qwen3-1.7B",
            "object": "model",
            "owned_by": "qwen",
            "ready": stats.model_id == "Qwen/Qwen3-1.7B",
            "permission": []
        }),
        serde_json::json!({
            "id": "Qwen/Qwen3-4B",
            "object": "model",
            "owned_by": "qwen",
            "ready": stats.model_id == "Qwen/Qwen3-4B",
            "permission": []
        }),
    ];

    // Deduplicate (in case current model is already in the list)
    let mut seen = std::collections::HashSet::new();
    let unique_models: Vec<_> = available_models
        .into_iter()
        .filter(|m| {
            let id = m["id"].as_str().unwrap_or("");
            seen.insert(id.to_string())
        })
        .collect();

    Json(serde_json::json!({
        "object": "list",
        "data": unique_models
    }))
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_max_level(Level::INFO)
        .init();

    // Parse model selection from environment or use default
    let model_id = std::env::var("FORGE_MODEL")
        .unwrap_or_else(|_| "HuggingFaceTB/SmolLM2-135M".to_string());

    info!("Selected model: {}", model_id);

    // Create model config with optimal settings
    let model_config = ModelConfig {
        model_id: model_id.clone(),
        revision: "main".to_string(),
        dtype: candle_core::DType::F16,  // Use F16 for GPU efficiency
        use_flash_attn: true,  // Enable Flash Attention
    };

    // Create engine config
    let engine_config = EngineConfig::with_model(model_config);

    // Check for device preference from environment
    let force_device = std::env::var("FORGE_DEVICE").ok();

    // Determine device (GPU if available, otherwise CPU)
    let device = match force_device.as_deref() {
        Some("cpu") | Some("CPU") => {
            info!("FORGE_DEVICE=cpu: Using CPU for inference");
            candle_core::Device::Cpu
        }
        Some("cuda") | Some("CUDA") | Some("gpu") | Some("GPU") => {
            if cfg!(feature = "cuda") {
                match candle_core::Device::new_cuda(0) {
                    Ok(cuda_device) => {
                        info!("FORGE_DEVICE=cuda: Using CUDA GPU for inference");
                        cuda_device
                    }
                    Err(e) => {
                        info!("CUDA requested but not available ({}), falling back to CPU", e);
                        candle_core::Device::Cpu
                    }
                }
            } else {
                info!("CUDA requested but not compiled in, using CPU");
                candle_core::Device::Cpu
            }
        }
        Some("metal") | Some("METAL") => {
            if cfg!(feature = "metal") {
                match candle_core::Device::new_metal(0) {
                    Ok(metal_device) => {
                        info!("FORGE_DEVICE=metal: Using Metal GPU for inference");
                        metal_device
                    }
                    Err(e) => {
                        info!("Metal requested but not available ({}), falling back to CPU", e);
                        candle_core::Device::Cpu
                    }
                }
            } else {
                info!("Metal requested but not compiled in, using CPU");
                candle_core::Device::Cpu
            }
        }
        _ => {
            // Auto-detect: try CUDA, then Metal, then CPU
            if cfg!(feature = "cuda") {
                match candle_core::Device::new_cuda(0) {
                    Ok(cuda_device) => {
                        info!("Auto-detected CUDA GPU for inference");
                        cuda_device
                    }
                    Err(e) => {
                        info!("CUDA not available ({}), trying Metal...", e);
                        if cfg!(feature = "metal") {
                            match candle_core::Device::new_metal(0) {
                                Ok(metal_device) => {
                                    info!("Auto-detected Metal GPU for inference");
                                    metal_device
                                }
                                Err(e) => {
                                    info!("Metal not available ({}), falling back to CPU", e);
                                    candle_core::Device::Cpu
                                }
                            }
                        } else {
                            info!("Falling back to CPU");
                            candle_core::Device::Cpu
                        }
                    }
                }
            } else if cfg!(feature = "metal") {
                match candle_core::Device::new_metal(0) {
                    Ok(metal_device) => {
                        info!("Auto-detected Metal GPU for inference");
                        metal_device
                    }
                    Err(e) => {
                        info!("Metal not available ({}), falling back to CPU", e);
                        candle_core::Device::Cpu
                    }
                }
            } else {
                info!("Using CPU for inference (no GPU features enabled)");
                candle_core::Device::Cpu
            }
        }
    };

    // Create inference engine
    info!("Initializing inference engine...");
    let engine = InferenceEngine::new(engine_config, device)?;

    let stats = engine.stats();
    info!("Engine initialized successfully!");
    info!("  Model: {}", stats.model_id);
    info!("  Vocab size: {}", stats.vocab_size);
    info!("  Hidden size: {}", stats.hidden_size);
    info!("  Layers: {}", stats.num_layers);
    info!("  Device: {}", stats.device);

    let state = AppState {
        engine: Arc::new(engine),
    };

    // Build router
    let app = Router::new()
        .route("/health", get(health_check))
        .route("/v1/completions", post(completions))
        .route("/v1/models", get(list_models))
        .layer(CorsLayer::permissive())
        .with_state(state);

    // Start server
    let port = std::env::var("FORGE_PORT")
        .unwrap_or_else(|_| "8080".to_string())
        .parse::<u16>()
        .unwrap_or(8080);

    let addr = format!("0.0.0.0:{}", port);
    info!("Starting server on {}", addr);

    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
