use axum::{
    Json, Router,
    extract::{Path, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{delete, get, post},
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::Mutex;
use tower_http::cors::CorsLayer;
use uuid::Uuid;

use crate::error::{MinervaError, MinervaResult};
#[allow(unused_imports)]
use crate::models::{
    ChatCompletionRequest, ChatCompletionResponse, ChatMessage, Choice, ModelInfo, ModelRegistry,
    ModelsListResponse, Usage,
};
use crate::observability::{
    endpoints::{HealthEndpointResponse, ReadinessResponse, MetricsResponse, RequestMetrics, ResponseTimeMetrics, ErrorMetrics, CacheMetrics},
    metrics::MetricsCollector,
};

pub type SharedModelRegistry = Arc<Mutex<ModelRegistry>>;

/// Request to load a model
#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(dead_code)]
pub struct ModelLoadRequest {
    pub model_id: String,
    pub model_path: String,
}

/// Response for model operations
#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(dead_code)]
pub struct ModelOperationResponse {
    pub success: bool,
    pub message: String,
    pub model_id: Option<String>,
}

/// Model statistics response
#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(dead_code)]
pub struct ModelStatsResponse {
    pub loaded_models: Vec<String>,
    pub total_loaded: usize,
    pub estimated_memory_mb: u64,
}

#[derive(Clone)]
#[allow(dead_code)]
pub struct ServerState {
    pub model_registry: SharedModelRegistry,
    pub metrics: Arc<MetricsCollector>,
}

impl ServerState {
    #[allow(dead_code)]
    pub fn new() -> Self {
        Self {
            model_registry: Arc::new(Mutex::new(ModelRegistry::new())),
            metrics: Arc::new(MetricsCollector::new()),
        }
    }
}

impl Default for ServerState {
    fn default() -> Self {
        Self::new()
    }
}

impl ServerState {
    /// Create server state and load discovered models
    #[allow(dead_code)]
    pub fn with_discovered_models(models_dir: std::path::PathBuf) -> MinervaResult<Self> {
        let mut registry = ModelRegistry::new();
        registry.discover(&models_dir)?;

        Ok(Self {
            model_registry: Arc::new(Mutex::new(registry)),
            metrics: Arc::new(MetricsCollector::new()),
        })
    }
}

#[allow(dead_code)]
pub async fn create_server(state: ServerState) -> Router {
    Router::new()
        .route("/v1/models", get(list_models))
        .route("/v1/models/:id/load", post(load_model))
        .route("/v1/models/:id/preload", post(preload_model))
        .route("/v1/models/:id", delete(unload_model))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/health", get(health_check_enhanced))
        .route("/ready", get(readiness_check))
        .route("/metrics", get(metrics_endpoint))
        .route("/v1/models/stats", get(model_stats))
        .with_state(state)
        .layer(CorsLayer::permissive())
}

/// Enhanced health check endpoint
async fn health_check_enhanced() -> impl IntoResponse {
    let mut resp = HealthEndpointResponse {
        timestamp: chrono::Local::now().to_rfc3339(),
        ..Default::default()
    };
    resp.calculate_status();
    Json(resp)
}

/// Readiness probe endpoint
async fn readiness_check() -> impl IntoResponse {
    let resp = ReadinessResponse::ready();
    Json(resp)
}

/// Metrics endpoint
async fn metrics_endpoint(State(state): State<ServerState>) -> impl IntoResponse {
    let metrics = state.metrics.snapshot();
    let uptime = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    let resp = MetricsResponse {
        timestamp: chrono::Local::now().to_rfc3339(),
        uptime_seconds: uptime,
        requests: RequestMetrics {
            total: metrics.total_requests,
            successful: metrics.successful_requests,
            failed: metrics.failed_requests,
            rps: metrics.rps,
        },
        response_times: ResponseTimeMetrics {
            avg_ms: metrics.avg_response_time_ms,
            min_ms: metrics.min_response_time_ms,
            max_ms: metrics.max_response_time_ms,
            p50_ms: metrics.p50_response_time_ms,
            p95_ms: metrics.p95_response_time_ms,
            p99_ms: metrics.p99_response_time_ms,
        },
        errors: ErrorMetrics {
            total: metrics.failed_requests,
            rate_percent: metrics.error_rate_percent,
            top_error: None,
        },
        cache: CacheMetrics {
            hits: metrics.cache_hits,
            misses: metrics.cache_misses,
            hit_rate_percent: metrics.cache_hit_rate_percent,
        },
    };

    Json(resp)
}

async fn list_models(State(state): State<ServerState>) -> MinervaResult<Json<ModelsListResponse>> {
    let registry = state.model_registry.lock().await;
    let models = registry.list_models();

    Ok(Json(ModelsListResponse {
        object: "list".to_string(),
        data: models,
    }))
}

async fn chat_completions(
    State(state): State<ServerState>,
    Json(req): Json<ChatCompletionRequest>,
) -> MinervaResult<Response> {
    let registry = state.model_registry.lock().await;

    registry
        .get_model(&req.model)
        .ok_or_else(|| MinervaError::ModelNotFound(format!("Model '{}' not found", req.model)))?;

    let is_streaming = req.stream.unwrap_or(false);

    if is_streaming {
        Ok(create_streaming_response(req).into_response())
    } else {
        Ok(create_completion_response(req).await.into_response())
    }
}

async fn create_completion_response(
    req: ChatCompletionRequest,
) -> MinervaResult<Json<ChatCompletionResponse>> {
    let completion_id = format!("chatcmpl-{}", Uuid::new_v4());
    let created = chrono::Utc::now().timestamp();

    // Build prompt from messages
    let prompt = build_chat_prompt(&req.messages);

    // For now, use mock response
    // In Phase 3.5, this will call the actual inference engine
    let response_content = format!(
        "Minerva inference response to: \"{}\" - Mock response for testing",
        prompt.chars().take(50).collect::<String>()
    );

    // Estimate token counts (actual tokenization in Phase 3.5)
    let prompt_tokens = estimate_tokens(&prompt);
    let completion_tokens = estimate_tokens(&response_content);

    Ok(Json(ChatCompletionResponse {
        id: completion_id,
        object: "chat.completion".to_string(),
        created,
        model: req.model,
        choices: vec![Choice {
            index: 0,
            message: ChatMessage {
                role: "assistant".to_string(),
                content: response_content,
            },
            finish_reason: "stop".to_string(),
        }],
        usage: Usage {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
        },
    }))
}

/// Build a chat prompt from messages
fn build_chat_prompt(messages: &[ChatMessage]) -> String {
    messages
        .iter()
        .map(|msg| format!("{}: {}", msg.role, msg.content))
        .collect::<Vec<_>>()
        .join("\n")
}

/// Estimate token count (rough approximation)
/// Real tokenization will use llama.cpp in Phase 3.5
fn estimate_tokens(text: &str) -> usize {
    // Rough estimate: ~4 characters per token
    (text.len() / 4).max(1)
}

fn create_streaming_response(_req: ChatCompletionRequest) -> impl IntoResponse {
    (StatusCode::OK, "streaming not yet implemented")
}

/// Load a model into memory
#[allow(dead_code)]
async fn load_model(
    State(_state): State<ServerState>,
    Path(_id): Path<String>,
    Json(_req): Json<ModelLoadRequest>,
) -> MinervaResult<Json<ModelOperationResponse>> {
    Ok(Json(ModelOperationResponse {
        success: true,
        message: "Model loading not yet implemented".to_string(),
        model_id: Some(_id),
    }))
}

/// Preload a model without marking as used
#[allow(dead_code)]
async fn preload_model(
    State(_state): State<ServerState>,
    Path(_id): Path<String>,
    Json(_req): Json<ModelLoadRequest>,
) -> MinervaResult<Json<ModelOperationResponse>> {
    Ok(Json(ModelOperationResponse {
        success: true,
        message: "Model preloading not yet implemented".to_string(),
        model_id: Some(_id),
    }))
}

/// Unload a model from memory
#[allow(dead_code)]
async fn unload_model(
    State(_state): State<ServerState>,
    Path(_id): Path<String>,
) -> MinervaResult<Json<ModelOperationResponse>> {
    Ok(Json(ModelOperationResponse {
        success: true,
        message: "Model unloading not yet implemented".to_string(),
        model_id: Some(_id),
    }))
}

/// Get model statistics
#[allow(dead_code)]
async fn model_stats(State(_state): State<ServerState>) -> MinervaResult<Json<ModelStatsResponse>> {
    Ok(Json(ModelStatsResponse {
        loaded_models: vec![],
        total_loaded: 0,
        estimated_memory_mb: 0,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_registry_empty() {
        let registry = ModelRegistry::new();
        let models = registry.list_models();
        assert_eq!(models.len(), 0);
    }

    #[test]
    fn test_model_registry_add_and_retrieve() {
        let mut registry = ModelRegistry::new();
        let model = ModelInfo {
            id: "test-model".to_string(),
            object: "model".to_string(),
            created: 1704067200,
            owned_by: "local".to_string(),
            context_window: Some(4096),
            max_output_tokens: Some(2048),
        };

        let path = std::path::PathBuf::from("/tmp/test-model.gguf");
        registry.add_model(model.clone(), path);
        let retrieved = registry.get_model("test-model");
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().id, "test-model");
    }

    #[test]
    fn test_model_registry_remove() {
        let mut registry = ModelRegistry::new();
        let model = ModelInfo {
            id: "test-model".to_string(),
            object: "model".to_string(),
            created: 1704067200,
            owned_by: "local".to_string(),
            context_window: None,
            max_output_tokens: None,
        };

        let path = std::path::PathBuf::from("/tmp/test-model.gguf");
        registry.add_model(model, path);
        assert_eq!(registry.list_models().len(), 1);

        registry.remove_model("test-model");
        assert_eq!(registry.list_models().len(), 0);
    }

    #[tokio::test]
    async fn test_list_models_endpoint() {
        let state = ServerState::new();
        let response = list_models(State(state)).await;

        assert!(response.is_ok());
        let Json(models_response) = response.unwrap();
        assert_eq!(models_response.data.len(), 0);
        assert_eq!(models_response.object, "list");
    }

    #[tokio::test]
    async fn test_chat_completions_model_not_found() {
        let state = ServerState::new();
        let req = ChatCompletionRequest {
            model: "nonexistent-model".to_string(),
            messages: vec![ChatMessage {
                role: "user".to_string(),
                content: "Hello".to_string(),
            }],
            temperature: None,
            max_tokens: None,
            stream: None,
            top_p: None,
            frequency_penalty: None,
            presence_penalty: None,
        };

        let response = chat_completions(State(state), Json(req)).await;
        assert!(response.is_err());
    }
}
