/// HTTP Server Configuration & Routing
pub mod chat;
pub mod endpoints;
pub mod handlers;
pub mod server_state;
pub mod streaming;
pub mod validation;

use axum::{Router, routing::{delete, get, post}};
use tower_http::cors::CorsLayer;
pub use self::server_state::ServerState;
use self::endpoints::{
    health_check_enhanced, readiness_check, metrics_endpoint,
    load_model, preload_model, unload_model, model_stats,
};

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::ChatMessage;
    use axum::http::HeaderMap;
    use axum::Json;
    use crate::models::ModelRegistry;

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
        let response = handlers::list_models(axum::extract::State(state)).await;

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
            axum::extract::State(state),
            headers,
            Json(req),
        )
        .await;
        assert!(response.is_err());
    }
}
