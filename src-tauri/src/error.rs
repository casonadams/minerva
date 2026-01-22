use axum::{
    Json,
    http::StatusCode,
    response::{IntoResponse, Response},
};
use serde_json::json;
use thiserror::Error;

#[derive(Error, Debug)]
#[allow(dead_code)]
pub enum MinervaError {
    #[error("Model not found: {0}")]
    ModelNotFound(String),

    #[error("Server error: {0}")]
    ServerError(String),

    #[error("Invalid request: {0}")]
    InvalidRequest(String),

    #[error("Inference error: {0}")]
    InferenceError(String),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("JSON error: {0}")]
    JsonError(#[from] serde_json::Error),

    #[error("Model loading error: {0}")]
    ModelLoadingError(String),

    #[error("Context limit exceeded: max {max}, required {required}")]
    ContextLimitExceeded { max: usize, required: usize },

    #[error("Generation timeout")]
    GenerationTimeout,

    #[error("Out of memory")]
    OutOfMemory,
}

impl IntoResponse for MinervaError {
    fn into_response(self) -> Response {
        let (status, error_code, message) = match self {
            MinervaError::ModelNotFound(msg) => (StatusCode::NOT_FOUND, "model_not_found", msg),
            MinervaError::InvalidRequest(msg) => (StatusCode::BAD_REQUEST, "invalid_request", msg),
            MinervaError::InferenceError(msg) => {
                (StatusCode::INTERNAL_SERVER_ERROR, "inference_error", msg)
            }
            MinervaError::ModelLoadingError(msg) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                "model_loading_error",
                msg,
            ),
            MinervaError::ContextLimitExceeded { max, required } => (
                StatusCode::BAD_REQUEST,
                "context_limit_exceeded",
                format!(
                    "Context limit exceeded: model supports {}, request requires {}",
                    max, required
                ),
            ),
            MinervaError::GenerationTimeout => (
                StatusCode::REQUEST_TIMEOUT,
                "generation_timeout",
                "Generation request timed out".to_string(),
            ),
            MinervaError::OutOfMemory => (
                StatusCode::INTERNAL_SERVER_ERROR,
                "out_of_memory",
                "Out of memory during inference".to_string(),
            ),
            _ => (
                StatusCode::INTERNAL_SERVER_ERROR,
                "server_error",
                self.to_string(),
            ),
        };

        let body = Json(json!({
            "error": {
                "message": message,
                "type": error_code,
                "code": error_code,
                "param": null
            }
        }));

        (status, body).into_response()
    }
}

#[allow(dead_code)]
pub type MinervaResult<T> = Result<T, MinervaError>;
