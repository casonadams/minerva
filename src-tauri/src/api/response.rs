//! API response handling

use super::types::{ApiError, ApiErrorResponse};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;

/// HTTP response implementation for API errors
impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let status = match self.code.as_str() {
            "invalid_model_id" | "invalid_parameter" | "invalid_request_error" => {
                StatusCode::BAD_REQUEST
            }
            "model_not_found" => StatusCode::NOT_FOUND,
            "rate_limit_exceeded" => StatusCode::TOO_MANY_REQUESTS,
            "server_error" => StatusCode::INTERNAL_SERVER_ERROR,
            _ => StatusCode::INTERNAL_SERVER_ERROR,
        };

        let response = ApiErrorResponse { error: self };
        (status, Json(response)).into_response()
    }
}

impl IntoResponse for ApiErrorResponse {
    fn into_response(self) -> Response {
        let status = match self.error.code.as_str() {
            "invalid_model_id" | "invalid_parameter" | "invalid_request_error" => {
                StatusCode::BAD_REQUEST
            }
            "model_not_found" => StatusCode::NOT_FOUND,
            "rate_limit_exceeded" => StatusCode::TOO_MANY_REQUESTS,
            "server_error" => StatusCode::INTERNAL_SERVER_ERROR,
            _ => StatusCode::INTERNAL_SERVER_ERROR,
        };

        (status, Json(self)).into_response()
    }
}
