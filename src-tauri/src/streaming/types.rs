//! Streaming types and structures

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
            keep_alive_ms: 15000,
            max_chunk_size: 50,
        }
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
}
