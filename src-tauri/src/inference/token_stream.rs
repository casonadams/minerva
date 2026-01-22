use std::sync::{Arc, Mutex};

/// Token stream callback for real-time streaming
///
/// This callback is invoked whenever a new token is generated.
/// Used for real-time streaming to clients via Server-Sent Events (SSE).
pub type TokenCallback = Arc<dyn Fn(String) + Send + Sync>;

/// Real token stream from llama.cpp
#[derive(Clone)]
#[allow(dead_code)]
pub struct TokenStream {
    tokens: Arc<Mutex<Vec<String>>>,
    current_index: Arc<Mutex<usize>>,
    callback: Option<Arc<Mutex<Option<TokenCallback>>>>,
}

impl std::fmt::Debug for TokenStream {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TokenStream")
            .field("tokens_count", &self.tokens.lock().unwrap().len())
            .field("current_index", &self.current_index.lock().unwrap())
            .field("has_callback", &self.callback.is_some())
            .finish()
    }
}

impl TokenStream {
    /// Create new token stream without callback
    pub fn new() -> Self {
        Self {
            tokens: Arc::new(Mutex::new(Vec::new())),
            current_index: Arc::new(Mutex::new(0)),
            callback: None,
        }
    }

    /// Create new token stream with callback for real-time streaming
    pub fn with_callback(callback: TokenCallback) -> Self {
        Self {
            tokens: Arc::new(Mutex::new(Vec::new())),
            current_index: Arc::new(Mutex::new(0)),
            callback: Some(Arc::new(Mutex::new(Some(callback)))),
        }
    }

    /// Add token to stream (called from llama.cpp callback)
    #[allow(clippy::collapsible_if)]
    pub fn push_token(&self, token: String) {
        let mut tokens = self.tokens.lock().unwrap();
        tokens.push(token.clone());

        // Invoke callback if registered
        if let Some(callback_opt) = &self.callback {
            if let Ok(callback_guard) = callback_opt.lock() {
                if let Some(callback) = callback_guard.as_ref() {
                    callback(token);
                }
            }
        }
    }

    /// Get next token
    pub fn next_token(&mut self) -> Option<String> {
        let tokens = self.tokens.lock().unwrap();
        let mut index = self.current_index.lock().unwrap();

        if *index < tokens.len() {
            let token = tokens[*index].clone();
            *index += 1;
            Some(token)
        } else {
            None
        }
    }

    /// Check if more tokens available
    pub fn has_next(&self) -> bool {
        let tokens = self.tokens.lock().unwrap();
        let index = self.current_index.lock().unwrap();
        *index < tokens.len()
    }

    /// Get total tokens received so far
    pub fn total_tokens(&self) -> usize {
        self.tokens.lock().unwrap().len()
    }

    /// Get current position
    pub fn position(&self) -> usize {
        *self.current_index.lock().unwrap()
    }

    /// Reset stream position
    pub fn reset(&mut self) {
        *self.current_index.lock().unwrap() = 0;
    }

    #[allow(clippy::inherent_to_string)]
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

    #[test]
    fn test_token_stream_with_callback() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let call_count = Arc::new(AtomicUsize::new(0));
        let call_count_clone = call_count.clone();

        let callback: TokenCallback = Arc::new(move |_token: String| {
            call_count_clone.fetch_add(1, Ordering::Relaxed);
        });

        let stream = TokenStream::with_callback(callback);

        stream.push_token("token1".to_string());
        stream.push_token("token2".to_string());
        stream.push_token("token3".to_string());

        // Verify callback was called for each token
        assert_eq!(call_count.load(Ordering::Relaxed), 3);
        assert_eq!(stream.total_tokens(), 3);
    }

    #[test]
    fn test_token_stream_callback_content() {
        let received_tokens = Arc::new(Mutex::new(Vec::new()));
        let received_tokens_clone = received_tokens.clone();

        let callback: TokenCallback = Arc::new(move |token: String| {
            received_tokens_clone.lock().unwrap().push(token);
        });

        let stream = TokenStream::with_callback(callback);

        stream.push_token("hello".to_string());
        stream.push_token(" ".to_string());
        stream.push_token("world".to_string());

        let tokens = received_tokens.lock().unwrap();
        assert_eq!(*tokens, vec!["hello", " ", "world"]);
    }

    #[test]
    fn test_token_stream_no_callback() {
        let stream = TokenStream::new();
        stream.push_token("token1".to_string());
        stream.push_token("token2".to_string());

        assert_eq!(stream.total_tokens(), 2);
        assert!(stream.has_next());
    }
}
