use axum::{
    extract::State,
    http::StatusCode,
    routing::{get, post},
    Json, Router,
};
use forge_core::{InferenceEngine, SamplingParams, EngineStats};
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
}

fn default_max_tokens() -> usize { 50 }
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
    info!("Completion request: prompt='{}' max_tokens={}", req.prompt, req.max_tokens);

    // Simple tokenization (in real impl, use proper tokenizer)
    let tokens: Vec<u32> = req.prompt.chars().map(|c| c as u32).collect();
    let prompt_tokens = tokens.len();

    let params = SamplingParams {
        max_tokens: req.max_tokens,
        temperature: req.temperature,
        top_p: req.top_p,
        top_k: req.top_k,
        stop_tokens: vec![],
    };

    // Generate
    let output_tokens = state.engine.generate(&tokens, &params)
        .map_err(|e| {
            (StatusCode::INTERNAL_SERVER_ERROR, Json(ErrorResponse {
                error: e.to_string(),
            }))
        })?;

    // Convert tokens back to text (simplified - just use token IDs as chars)
    let text: String = output_tokens.iter()
        .filter_map(|&t| char::from_u32(t))
        .collect();

    let completion_tokens = output_tokens.len();

    let response = CompletionResponse {
        id: format!("cmpl-{}", uuid_simple()),
        object: "text_completion".to_string(),
        created: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs(),
        model: "forge-mock".to_string(),
        choices: vec![Choice {
            index: 0,
            text,
            finish_reason: "length".to_string(),
        }],
        usage: Usage {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
        },
    };

    Ok(Json(response))
}

fn uuid_simple() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap();
    format!("{:x}{:x}", now.as_secs(), now.subsec_nanos())
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_max_level(Level::INFO)
        .init();

    // Create inference engine
    info!("Initializing inference engine...");

    let engine = InferenceEngine::new_cpu()?;
    info!("Engine initialized: {:?}", engine.stats());

    let state = AppState {
        engine: Arc::new(engine),
    };

    // Build router
    let app = Router::new()
        .route("/health", get(health_check))
        .route("/v1/completions", post(completions))
        .layer(CorsLayer::permissive())
        .with_state(state);

    // Start server
    let addr = "0.0.0.0:8080";
    info!("Starting server on {}", addr);

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
