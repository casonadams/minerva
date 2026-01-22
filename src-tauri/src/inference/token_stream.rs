use std::sync::{Arc, Mutex};

/// Real token stream from llama.cpp
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct TokenStream {
    tokens: Arc<Mutex<Vec<String>>>,
    current_index: usize,
}

impl TokenStream {
    #[allow(dead_code)]
    /// Create new token stream
    pub fn new() -> Self {
        Self {
            tokens: Arc::new(Mutex::new(Vec::new())),
            current_index: 0,
        }
    }

    #[allow(dead_code)]
    /// Add token to stream (called from llama.cpp callback)
    pub fn push_token(&self, token: String) {
        let mut tokens = self.tokens.lock().unwrap();
        tokens.push(token);
    }

    #[allow(dead_code)]
    /// Get next token
    pub fn next_token(&mut self) -> Option<String> {
        let tokens = self.tokens.lock().unwrap();
        if self.current_index < tokens.len() {
            let token = tokens[self.current_index].clone();
            self.current_index += 1;
            Some(token)
        } else {
            None
        }
    }

    #[allow(dead_code)]
    /// Check if more tokens available
    pub fn has_next(&self) -> bool {
        let tokens = self.tokens.lock().unwrap();
        self.current_index < tokens.len()
    }

    #[allow(dead_code)]
    /// Get total tokens received so far
    pub fn total_tokens(&self) -> usize {
        self.tokens.lock().unwrap().len()
    }

    #[allow(dead_code)]
    /// Get current position
    pub fn position(&self) -> usize {
        self.current_index
    }

    #[allow(dead_code)]
    /// Reset stream position
    pub fn reset(&mut self) {
        self.current_index = 0;
    }

    #[allow(clippy::inherent_to_string)]
    #[allow(dead_code)]
    /// Get all tokens as string
    pub fn to_string(&self) -> String {
        let tokens = self.tokens.lock().unwrap();
        tokens.join("")
    }
}

impl Default for TokenStream {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_token_stream_creation() {
        let stream = TokenStream::new();
        assert_eq!(stream.total_tokens(), 0);
        assert!(!stream.has_next());
    }

    #[test]
    fn test_token_stream_push_and_next() {
        let stream = TokenStream::new();
        stream.push_token("hello ".to_string());
        stream.push_token("world".to_string());

        let mut stream = stream;
        assert_eq!(stream.next_token(), Some("hello ".to_string()));
        assert_eq!(stream.next_token(), Some("world".to_string()));
        assert_eq!(stream.next_token(), None);
    }

    #[test]
    fn test_token_stream_position() {
        let stream = TokenStream::new();
        stream.push_token("a".to_string());
        stream.push_token("b".to_string());

        let mut stream = stream;
        assert_eq!(stream.position(), 0);
        stream.next_token();
        assert_eq!(stream.position(), 1);
        stream.next_token();
        assert_eq!(stream.position(), 2);
    }

    #[test]
    fn test_token_stream_reset() {
        let stream = TokenStream::new();
        stream.push_token("test".to_string());

        let mut stream = stream;
        stream.next_token();
        assert_eq!(stream.position(), 1);
        stream.reset();
        assert_eq!(stream.position(), 0);
    }

    #[test]
    fn test_token_stream_to_string() {
        let stream = TokenStream::new();
        stream.push_token("hello".to_string());
        stream.push_token(" ".to_string());
        stream.push_token("world".to_string());

        assert_eq!(stream.to_string(), "hello world");
    }
}
