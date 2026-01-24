use crate::models::ChatCompletionChunk;
use uuid::Uuid;

/// Streaming response builder for chat completions
#[derive(Debug)]
#[allow(dead_code)]
pub struct StreamingResponse {
    completion_id: String,
    model: String,
    created: i64,
}

impl StreamingResponse {
    /// Create a new streaming response builder
    #[allow(dead_code)]
    pub fn new(model: String) -> Self {
        Self {
            completion_id: format!("chatcmpl-{}", Uuid::new_v4()),
            model,
            created: chrono::Utc::now().timestamp(),
        }
    }

    /// Build a chunk for a token
    #[allow(dead_code)]
    pub fn chunk(&self, token: &str, index: usize) -> ChatCompletionChunk {
        ChatCompletionChunk {
            id: self.completion_id.clone(),
            object: "chat.completion.chunk".to_string(),
            created: self.created,
            model: self.model.clone(),
            choices: vec![crate::models::ChoiceDelta {
                index,
                delta: crate::models::DeltaMessage {
                    role: None,
                    content: Some(token.to_string()),
                },
                finish_reason: None,
            }],
        }
    }

    /// Build a final chunk to signal end of stream
    #[allow(dead_code)]
    pub fn chunk_end(&self, index: usize) -> ChatCompletionChunk {
        ChatCompletionChunk {
            id: self.completion_id.clone(),
            object: "chat.completion.chunk".to_string(),
            created: self.created,
            model: self.model.clone(),
            choices: vec![crate::models::ChoiceDelta {
                index,
                delta: crate::models::DeltaMessage {
                    role: None,
                    content: None,
                },
                finish_reason: Some("stop".to_string()),
            }],
        }
    }

    /// Convert chunk to SSE (Server-Sent Event) format
    #[allow(dead_code)]
    pub fn to_sse_string(chunk: &ChatCompletionChunk) -> String {
        let json = serde_json::to_string(chunk).unwrap_or_else(|_| "{}".to_string());
        format!("data: {}\n\n", json)
    }

    /// Get completion ID
    #[allow(dead_code)]
    pub fn completion_id(&self) -> &str {
        &self.completion_id
    }

    /// Get model name
    #[allow(dead_code)]
    pub fn model(&self) -> &str {
        &self.model
    }

    /// Get creation timestamp
    #[allow(dead_code)]
    pub fn created(&self) -> i64 {
        self.created
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_streaming_response_creation() {
        let response = StreamingResponse::new("test-model".to_string());
        assert_eq!(response.model(), "test-model");
        assert!(!response.completion_id().is_empty());
        assert!(response.created() > 0);
    }

    #[test]
    fn test_streaming_response_chunk() {
        let response = StreamingResponse::new("test-model".to_string());
        let chunk = response.chunk("hello ", 0);

        assert_eq!(chunk.object, "chat.completion.chunk");
        assert_eq!(chunk.choices.len(), 1);
        assert_eq!(chunk.choices[0].index, 0);
        assert_eq!(chunk.choices[0].delta.content, Some("hello ".to_string()));
        assert_eq!(chunk.choices[0].finish_reason, None);
    }

    #[test]
    fn test_streaming_response_end_chunk() {
        let response = StreamingResponse::new("test-model".to_string());
        let chunk = response.chunk_end(0);

        assert_eq!(chunk.choices[0].delta.content, None);
        assert_eq!(chunk.choices[0].finish_reason, Some("stop".to_string()));
    }

    #[test]
    fn test_sse_format() {
        let response = StreamingResponse::new("test-model".to_string());
        let chunk = response.chunk("token", 0);
        let sse = StreamingResponse::to_sse_string(&chunk);

        assert!(sse.starts_with("data: "));
        assert!(sse.ends_with("\n\n"));
        assert!(sse.contains("chat.completion.chunk"));
    }
}
