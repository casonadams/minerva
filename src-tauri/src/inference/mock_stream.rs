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
