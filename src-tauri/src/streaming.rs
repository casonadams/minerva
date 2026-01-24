//! Streaming Response Support
//! Handles SSE (Server-Sent Events) and streaming responses for token generation

use serde::{Deserialize, Serialize};

/// Streaming response format (SSE)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamingDelta {
    pub content: Option<String>,
    pub finish_reason: Option<String>,
}

/// Chat completion streaming response event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionStreamEvent {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub model: String,
    pub choices: Vec<StreamingChoice>,
}

/// Streaming choice
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamingChoice {
    pub index: u32,
    pub delta: StreamingDelta,
    pub finish_reason: Option<String>,
}

/// Streaming configuration
#[derive(Debug, Clone)]
pub struct StreamingConfig {
    /// Keep-alive interval in milliseconds
    pub keep_alive_ms: u64,
    /// Maximum chunk size in tokens
    pub max_chunk_size: usize,
}

impl Default for StreamingConfig {
    fn default() -> Self {
        Self {
            keep_alive_ms: 15000, // 15 seconds
            max_chunk_size: 50,
        }
    }
}

/// Formats a single streaming event as JSON
pub fn format_streaming_event(event: &ChatCompletionStreamEvent) -> String {
    serde_json::to_string(event).unwrap_or_else(|_| "{}".to_string())
}

/// Creates a list of streaming events from tokens
pub fn create_streaming_events(
    request_id: String,
    model: String,
    tokens: Vec<String>,
) -> Vec<ChatCompletionStreamEvent> {
    let mut events = Vec::new();

    // Add token events
    for token in tokens {
        events.push(ChatCompletionStreamEvent {
            id: request_id.clone(),
            object: "text_completion.chunk".to_string(),
            created: chrono::Utc::now().timestamp(),
            model: model.clone(),
            choices: vec![StreamingChoice {
                index: 0,
                delta: StreamingDelta {
                    content: Some(token),
                    finish_reason: None,
                },
                finish_reason: None,
            }],
        });
    }

    // Add finish event
    events.push(ChatCompletionStreamEvent {
        id: request_id,
        object: "text_completion.chunk".to_string(),
        created: chrono::Utc::now().timestamp(),
        model,
        choices: vec![StreamingChoice {
            index: 0,
            delta: StreamingDelta {
                content: None,
                finish_reason: Some("stop".to_string()),
            },
            finish_reason: Some("stop".to_string()),
        }],
    });

    events
}

/// Validates streaming parameters
pub struct StreamingValidator;

impl StreamingValidator {
    /// Validate stream parameter
    pub fn validate_stream(_stream: bool) -> Result<(), String> {
        // Stream is valid if it's a boolean (always valid)
        Ok(())
    }

    /// Validate streaming chunk size
    pub fn validate_chunk_size(size: usize) -> Result<(), String> {
        if size == 0 {
            return Err("Chunk size must be greater than 0".to_string());
        }
        if size > 1000 {
            return Err("Chunk size must not exceed 1000 tokens".to_string());
        }
        Ok(())
    }

    /// Check if streaming is compatible with model
    pub fn is_streaming_supported(model: &str) -> bool {
        // All models support streaming by default
        !model.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_streaming_config_default() {
        let config = StreamingConfig::default();
        assert_eq!(config.keep_alive_ms, 15000);
        assert_eq!(config.max_chunk_size, 50);
    }

    #[test]
    fn test_streaming_choice_creation() {
        let choice = StreamingChoice {
            index: 0,
            delta: StreamingDelta {
                content: Some("hello".to_string()),
                finish_reason: None,
            },
            finish_reason: None,
        };

        assert_eq!(choice.index, 0);
        assert_eq!(choice.delta.content, Some("hello".to_string()));
        assert!(choice.finish_reason.is_none());
    }

    #[test]
    fn test_streaming_delta_with_content() {
        let delta = StreamingDelta {
            content: Some("test".to_string()),
            finish_reason: None,
        };

        assert_eq!(delta.content, Some("test".to_string()));
        assert!(delta.finish_reason.is_none());
    }

    #[test]
    fn test_streaming_delta_finish() {
        let delta = StreamingDelta {
            content: None,
            finish_reason: Some("stop".to_string()),
        };

        assert!(delta.content.is_none());
        assert_eq!(delta.finish_reason, Some("stop".to_string()));
    }

    #[test]
    fn test_validator_chunk_size_valid() {
        assert!(StreamingValidator::validate_chunk_size(1).is_ok());
        assert!(StreamingValidator::validate_chunk_size(50).is_ok());
        assert!(StreamingValidator::validate_chunk_size(1000).is_ok());
    }

    #[test]
    fn test_validator_chunk_size_invalid() {
        assert!(StreamingValidator::validate_chunk_size(0).is_err());
        assert!(StreamingValidator::validate_chunk_size(1001).is_err());
    }

    #[test]
    fn test_validator_stream_always_valid() {
        assert!(StreamingValidator::validate_stream(true).is_ok());
        assert!(StreamingValidator::validate_stream(false).is_ok());
    }

    #[test]
    fn test_validator_streaming_supported() {
        assert!(StreamingValidator::is_streaming_supported("gpt-4"));
        assert!(StreamingValidator::is_streaming_supported("llama-2"));
        assert!(!StreamingValidator::is_streaming_supported(""));
    }

    #[test]
    fn test_chat_completion_stream_event_creation() {
        let event = ChatCompletionStreamEvent {
            id: "test-id".to_string(),
            object: "text_completion.chunk".to_string(),
            created: 123456789,
            model: "gpt-4".to_string(),
            choices: vec![StreamingChoice {
                index: 0,
                delta: StreamingDelta {
                    content: Some("hello".to_string()),
                    finish_reason: None,
                },
                finish_reason: None,
            }],
        };

        assert_eq!(event.id, "test-id");
        assert_eq!(event.object, "text_completion.chunk");
        assert_eq!(event.model, "gpt-4");
        assert_eq!(event.choices.len(), 1);
    }

    #[test]
    fn test_streaming_event_serialization() {
        let event = ChatCompletionStreamEvent {
            id: "test".to_string(),
            object: "chunk".to_string(),
            created: 100,
            model: "test-model".to_string(),
            choices: vec![StreamingChoice {
                index: 0,
                delta: StreamingDelta {
                    content: Some("data".to_string()),
                    finish_reason: None,
                },
                finish_reason: None,
            }],
        };

        let json = serde_json::to_string(&event).expect("Should serialize");
        assert!(
            json.contains("test-id") || json.contains("test"),
            "Should contain ID"
        );
        assert!(json.contains("chunk"), "Should contain object type");
    }
}
