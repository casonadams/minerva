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

/// Simulates token streaming for testing
#[derive(Debug)]
#[allow(dead_code)]
pub struct MockTokenStream {
    tokens: Vec<String>,
    current_index: usize,
}

impl MockTokenStream {
    /// Create a mock stream with predefined tokens
    #[allow(dead_code)]
    pub fn new(text: &str) -> Self {
        // Split text into words (simple tokenization for testing)
        let tokens: Vec<String> = text
            .split_whitespace()
            .map(|word| format!("{} ", word))
            .collect();

        Self {
            tokens,
            current_index: 0,
        }
    }

    /// Get next token
    #[allow(dead_code)]
    pub fn next_token(&mut self) -> Option<String> {
        if self.current_index < self.tokens.len() {
            let token = self.tokens[self.current_index].clone();
            self.current_index += 1;
            Some(token)
        } else {
            None
        }
    }

    /// Check if there are more tokens
    #[allow(dead_code)]
    pub fn has_next(&self) -> bool {
        self.current_index < self.tokens.len()
    }

    /// Get total token count
    #[allow(dead_code)]
    pub fn total_tokens(&self) -> usize {
        self.tokens.len()
    }

    /// Get current position
    #[allow(dead_code)]
    pub fn position(&self) -> usize {
        self.current_index
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

    #[test]
    fn test_mock_token_stream_creation() {
        let stream = MockTokenStream::new("hello world test");
        assert_eq!(stream.total_tokens(), 3);
        assert!(stream.has_next());
    }

    #[test]
    fn test_mock_token_stream_iteration() {
        let mut stream = MockTokenStream::new("hello world");
        assert_eq!(stream.position(), 0);

        let token1 = stream.next_token();
        assert!(token1.is_some());
        assert_eq!(token1.unwrap(), "hello ");
        assert_eq!(stream.position(), 1);

        let token2 = stream.next_token();
        assert!(token2.is_some());
        assert_eq!(token2.unwrap(), "world ");
        assert_eq!(stream.position(), 2);

        let token3 = stream.next_token();
        assert!(token3.is_none());
        assert!(!stream.has_next());
    }

    #[test]
    fn test_mock_token_stream_empty() {
        let stream = MockTokenStream::new("");
        assert_eq!(stream.total_tokens(), 0);
        assert!(!stream.has_next());
    }
}
