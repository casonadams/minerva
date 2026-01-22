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

    #[error("Out of memory: {0}")]
    OutOfMemory(String),

    /// Phase 3.5b: GPU out of memory - can fallback to CPU
    #[error("GPU out of memory: {0}, falling back to CPU")]
    GpuOutOfMemory(String),

    /// Phase 3.5b: GPU context lost - needs reinitialization
    #[error("GPU context lost: {0}")]
    GpuContextLost(String),

    /// Phase 3.5b: Model corrupted or incompatible
    #[error("Model corrupted or incompatible: {0}")]
    ModelCorrupted(String),

    /// Phase 3.5b: Streaming error - can retry
    #[error("Streaming error: {0}")]
    StreamingError(String),
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
            MinervaError::OutOfMemory(msg) => {
                (StatusCode::INTERNAL_SERVER_ERROR, "out_of_memory", msg)
            }
            MinervaError::GpuOutOfMemory(msg) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                "gpu_out_of_memory",
                format!("GPU memory exhausted, falling back to CPU: {}", msg),
            ),
            MinervaError::GpuContextLost(msg) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                "gpu_context_lost",
                format!("GPU context lost, will reinitialize: {}", msg),
            ),
            MinervaError::ModelCorrupted(msg) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                "model_corrupted",
                format!("Model file corrupted: {}", msg),
            ),
            MinervaError::StreamingError(msg) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                "streaming_error",
                format!("Streaming error (retryable): {}", msg),
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
