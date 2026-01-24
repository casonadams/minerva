/// API Response Types
///
/// Lean, focused response structures for inference API.
/// Follows OpenAI API format for compatibility.
use serde::{Deserialize, Serialize};

/// Standard error response format
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorResponse {
    pub error: ErrorDetail,
}

/// Error details matching OpenAI format
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorDetail {
    pub message: String,
    pub code: String,
    pub param: Option<String>,
}

/// Single token prediction response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenResponse {
    pub token_id: u32,
    pub token: String,
    pub logit: f32,
    pub probability: f32,
}

/// Inference completion response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceResponse {
    pub id: String,
    pub model: String,
    pub tokens: Vec<TokenResponse>,
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
    pub created: u64,
    pub finish_reason: String,
}

/// Streaming response chunk (SSE format)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamChunk {
    pub id: String,
    pub model: String,
    pub token: TokenResponse,
    pub index: usize,
    pub created: u64,
    pub finish_reason: Option<String>,
}

/// Model information response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfoResponse {
    pub id: String,
    pub name: String,
    pub model_type: String,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    pub max_seq_len: usize,
    pub loaded: bool,
    pub memory_mb: u64,
}

/// List of loaded models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelsResponse {
    pub object: String,
    pub data: Vec<ModelInfoResponse>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_response_format() {
        let err = ErrorResponse {
            error: ErrorDetail {
                message: "test error".to_string(),
                code: "test_code".to_string(),
                param: None,
            },
        };
        let json = serde_json::to_string(&err).unwrap();
        assert!(json.contains("test error"));
    }

    #[test]
    fn test_model_info_response() {
        let info = ModelInfoResponse {
            id: "test".to_string(),
            name: "Test Model".to_string(),
            model_type: "llama".to_string(),
            vocab_size: 32000,
            hidden_size: 4096,
            num_layers: 32,
            num_attention_heads: 32,
            intermediate_size: 11008,
            max_seq_len: 2048,
            loaded: true,
            memory_mb: 5000,
        };
        let json = serde_json::to_string(&info).unwrap();
        assert!(json.contains("Test Model"));
    }
}
