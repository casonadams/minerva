pub mod handlers;
pub mod chat;
pub mod streaming;
pub mod validation;

use axum::{Json, Router, extract::Path, extract::State, routing::{delete, get, post}, response::IntoResponse};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::Mutex;
use tower_http::cors::CorsLayer;

use crate::error::MinervaResult;
use crate::middleware::RateLimiter;
use crate::models::ModelRegistry;
use crate::observability::metrics::MetricsCollector;

pub type SharedModelRegistry = Arc<Mutex<ModelRegistry>>;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(dead_code)]
pub struct ModelLoadRequest {
    pub model_id: String,
    pub model_path: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(dead_code)]
pub struct ModelOperationResponse {
    pub success: bool,
    pub message: String,
    pub model_id: Option<String>,
}

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
    pub rate_limiter: Arc<RateLimiter>,
}

impl ServerState {
    #[allow(dead_code)]
    pub fn new() -> Self {
        Self {
            model_registry: Arc::new(Mutex::new(ModelRegistry::new())),
            metrics: Arc::new(MetricsCollector::new()),
            rate_limiter: Arc::new(RateLimiter::new(100.0, 10.0)),
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
    pub fn with_discovered_models(
        models_dir: std::path::PathBuf,
    ) -> MinervaResult<Self> {
        let mut registry = ModelRegistry::new();
        registry.discover(&models_dir)?;

        Ok(Self {
            model_registry: Arc::new(Mutex::new(registry)),
            metrics: Arc::new(MetricsCollector::new()),
            rate_limiter: Arc::new(RateLimiter::new(100.0, 10.0)),
        })
    }
}

#[allow(dead_code)]
pub async fn create_server(state: ServerState) -> Router {
    Router::new()
        .route("/v1/models", get(handlers::list_models))
        .route("/v1/models/:id/load", post(load_model))
        .route("/v1/models/:id/preload", post(preload_model))
        .route("/v1/models/:id", delete(unload_model))
        .route("/v1/chat/completions", post(handlers::chat_completions))
        .route("/health", get(health_check_enhanced))
        .route("/ready", get(readiness_check))
        .route("/metrics", get(metrics_endpoint))
        .route("/v1/models/stats", get(model_stats))
        .with_state(state)
        .layer(CorsLayer::permissive())
}

async fn health_check_enhanced() -> impl IntoResponse {
    use crate::observability::health::HealthEndpointResponse;

    let mut resp = HealthEndpointResponse {
        timestamp: chrono::Local::now().to_rfc3339(),
        ..Default::default()
    };
    resp.calculate_status();
    Json(resp)
}

async fn readiness_check() -> impl IntoResponse {
    use crate::observability::endpoints::ReadinessResponse;

    let resp = ReadinessResponse::ready();
    Json(resp)
}

async fn metrics_endpoint(
    State(state): State<ServerState>,
) -> impl IntoResponse {
    use crate::observability::endpoints::{
        CacheMetrics, ErrorMetrics, MetricsResponse, RequestMetrics, ResponseTimeMetrics,
    };

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

#[allow(dead_code)]
async fn model_stats(
    State(_state): State<ServerState>,
) -> MinervaResult<Json<ModelStatsResponse>> {
    Ok(Json(ModelStatsResponse {
        loaded_models: vec![],
        total_loaded: 0,
        estimated_memory_mb: 0,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::ChatMessage;
    use axum::http::HeaderMap;

    #[test]
    fn test_model_registry_empty() {
        let registry = ModelRegistry::new();
        let models = registry.list_models();
        assert_eq!(models.len(), 0);
    }

    #[test]
    fn test_model_registry_add_and_retrieve() {
        use crate::models::ModelInfo;
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

    #[tokio::test]
    async fn test_list_models_endpoint() {
        let state = ServerState::new();
        let response = handlers::list_models(State(state)).await;

        assert!(response.is_ok());
        let Json(models_response) = response.unwrap();
        assert_eq!(models_response.data.len(), 0);
        assert_eq!(models_response.object, "list");
    }

    #[tokio::test]
    async fn test_chat_completions_model_not_found() {
        let state = ServerState::new();
        let req = crate::models::ChatCompletionRequest {
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

        let headers = HeaderMap::new();
        let response = handlers::chat_completions(
            State(state),
            headers,
            Json(req),
        )
        .await;
        assert!(response.is_err());
    }
}
