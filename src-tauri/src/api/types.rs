//! API protocol types

use serde::{Deserialize, Serialize};

/// Standard API error response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiErrorResponse {
    pub error: ApiError,
}

/// Error details
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiError {
    pub message: String,
    pub code: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub type_: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub param: Option<String>,
}

/// Standard API response wrapper
#[derive(Debug, Serialize)]
pub struct ApiResponse<T> {
    pub data: T,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub meta: Option<ResponseMetadata>,
}

/// Response metadata
#[derive(Debug, Serialize)]
pub struct ResponseMetadata {
    pub request_id: String,
    pub timestamp: String,
    pub version: String,
}

impl<T: Serialize> ApiResponse<T> {
    pub fn new(data: T) -> Self {
        Self {
            data,
            meta: Some(ResponseMetadata {
                request_id: uuid::Uuid::new_v4().to_string(),
                timestamp: chrono::Utc::now().to_rfc3339(),
                version: "0.1.0".to_string(),
            }),
        }
    }

    pub fn without_meta(data: T) -> Self {
        Self { data, meta: None }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_api_response_with_metadata() {
        let data = vec!["test"];
        let response = ApiResponse::new(data);
        assert!(response.meta.is_some(), "Meta should be present");
    }

    #[test]
    fn test_api_response_without_metadata() {
        let data = vec!["test"];
        let response = ApiResponse::<Vec<&str>>::without_meta(data);
        assert!(response.meta.is_none(), "Meta should not be present");
    }
}
