use axum::{Json, response::IntoResponse};
use crate::error::MinervaResult;
use crate::models::ChatCompletionRequest;
use axum::http::HeaderMap;
use crate::server::ServerState;
use super::chat::create_completion_response;
use super::streaming::create_streaming_response;
use super::validation::validate_chat_request;

pub async fn list_models(
    axum::extract::State(state): axum::extract::State<ServerState>,
) -> MinervaResult<Json<crate::models::ModelsListResponse>> {
    let registry = state.model_registry.lock().await;
    let models = registry.list_models();

    Ok(Json(crate::models::ModelsListResponse {
        object: "list".to_string(),
        data: models,
    }))
}

pub async fn chat_completions(
    axum::extract::State(state): axum::extract::State<ServerState>,
    headers: HeaderMap,
    Json(req): Json<ChatCompletionRequest>,
) -> MinervaResult<axum::response::Response> {
    let client_id = headers
        .get("x-client-id")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("anonymous");

    validate_chat_request(&req)?;

    if !state.rate_limiter.allow_request(client_id, 1.0).await {
        let retry = state.rate_limiter.retry_after(client_id, 1.0).await;
        return Err(crate::error::MinervaError::InvalidRequest(format!(
            "Rate limit exceeded. Retry after {} seconds",
            retry
        )));
    }

    let registry = state.model_registry.lock().await;
    registry
        .get_model(&req.model)
        .ok_or_else(|| crate::error::MinervaError::ModelNotFound(
            format!("Model '{}' not found", req.model)
        ))?;

    let is_streaming = req.stream.unwrap_or(false);

    if is_streaming {
        Ok(create_streaming_response(req).into_response())
    } else {
        Ok(create_completion_response(req).await?.into_response())
    }
}
