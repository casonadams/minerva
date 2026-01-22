use axum::{
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use serde_json::json;
use std::sync::Arc;
use tokio::sync::Mutex;
use tower_http::cors::CorsLayer;
use uuid::Uuid;

use crate::error::{MinervaError, MinervaResult};
use crate::models::{
    ChatCompletionRequest, ChatCompletionResponse, Choice, ChatMessage, ModelsListResponse,
    ModelRegistry, Usage,
};

pub type SharedModelRegistry = Arc<Mutex<ModelRegistry>>;

#[derive(Clone)]
pub struct ServerState {
    pub model_registry: SharedModelRegistry,
}

impl ServerState {
    pub fn new() -> Self {
        Self {
            model_registry: Arc::new(Mutex::new(ModelRegistry::new())),
        }
    }
}

pub async fn create_server(state: ServerState) -> Router {
    Router::new()
        .route("/v1/models", get(list_models))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/health", get(health_check))
        .with_state(state)
        .layer(CorsLayer::permissive())
}

async fn health_check() -> impl IntoResponse {
    Json(json!({
        "status": "ok",
        "timestamp": chrono::Utc::now().to_rfc3339()
    }))
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

    registry.get_model(&req.model).ok_or_else(|| {
        MinervaError::ModelNotFound(format!("Model '{}' not found", req.model))
    })?;

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

    let response_content = "This is a mock response from Minerva. Actual LLM inference will be implemented in Phase 3.".to_string();

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
            prompt_tokens: 10,
            completion_tokens: 20,
            total_tokens: 30,
        },
    }))
}

fn create_streaming_response(_req: ChatCompletionRequest) -> impl IntoResponse {
    (StatusCode::OK, "streaming not yet implemented")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_health_check() {
        let response = health_check().await;
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_list_models_empty() {
        let state = ServerState::new();
        let response = list_models(State(state)).await;
        
        assert!(response.is_ok());
        let Json(models_response) = response.unwrap();
        assert_eq!(models_response.data.len(), 0);
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
