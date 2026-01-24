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
    stop: Vec<String>,
    #[serde(default)]
    seed: Option<u64>,
    #[serde(default)]
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
struct ModelRequest {
    model_id: Option<String>,
}

async fn list_models(State(state): State<AppState>) -> Json<serde_json::Value> {
    let stats = state.engine.stats();
    Json(serde_json::json!({
        "object": "list",
        "data": [{
            "id": stats.model_id,
            "object": "model",
            "owned_by": "forge",
            "permission": []
        }]
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

    // Create model config
    let model_config = ModelConfig {
        model_id: model_id.clone(),
        revision: "main".to_string(),
        dtype: candle_core::DType::F32,
        use_flash_attn: false,
    };

    // Create engine config
    let engine_config = EngineConfig::with_model(model_config);

    // Create inference engine
    info!("Initializing inference engine...");
    let engine = InferenceEngine::new(engine_config, candle_core::Device::Cpu)?;

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
