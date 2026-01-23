/// Streaming Response Delivery for Phase 5
///
/// This module provides streaming response capabilities for progressive token delivery.
/// Enables better UX by streaming tokens as they become available instead of waiting for batch completion.
use std::sync::{Arc, mpsc};

use parking_lot::Mutex;

/// Token stream event
#[derive(Clone, Debug)]
pub enum StreamEvent {
    Token(String),
    Delta(String),
    Done,
    Error(String),
}

impl StreamEvent {
    pub fn token(value: String) -> Self {
        StreamEvent::Token(value)
    }

    pub fn delta(value: String) -> Self {
        StreamEvent::Delta(value)
    }

    pub fn done() -> Self {
        StreamEvent::Done
    }

    pub fn error(msg: String) -> Self {
        StreamEvent::Error(msg)
    }
}

/// Stream buffer for collecting tokens
#[derive(Clone)]
pub struct StreamBuffer {
    buffer: Arc<Mutex<Vec<String>>>,
    capacity: usize,
}

impl StreamBuffer {
    pub fn new(capacity: usize) -> Self {
        Self {
            buffer: Arc::new(Mutex::new(Vec::with_capacity(capacity))),
            capacity,
        }
    }

    /// Add token to buffer
    pub fn add_token(&self, token: String) {
        let mut buf = self.buffer.lock();
        buf.push(token);
    }

    /// Get all tokens and clear
    pub fn flush(&self) -> Vec<String> {
        let mut buf = self.buffer.lock();
        buf.drain(..).collect()
    }

    /// Get current buffer size
    pub fn size(&self) -> usize {
        self.buffer.lock().len()
    }

    /// Check if buffer is full
    pub fn is_full(&self) -> bool {
        self.buffer.lock().len() >= self.capacity
    }

    /// Get all tokens without clearing
    pub fn peek(&self) -> Vec<String> {
        self.buffer.lock().clone()
    }
}

/// Streaming response handler
pub struct StreamingResponse {
    tx: mpsc::Sender<StreamEvent>,
    #[allow(dead_code)]
    rx: Arc<Mutex<mpsc::Receiver<StreamEvent>>>,
    buffer: StreamBuffer,
}

impl StreamingResponse {
    /// Create a new streaming response with buffer
    pub fn new(buffer_capacity: usize) -> Self {
        let (tx, _rx) = mpsc::channel();
        Self {
            tx,
            rx: Arc::new(Mutex::new(mpsc::channel().1)),
            buffer: StreamBuffer::new(buffer_capacity),
        }
    }

    /// Create with specific channel
    pub fn with_sender(tx: mpsc::Sender<StreamEvent>, buffer_capacity: usize) -> Self {
        Self {
            tx,
            rx: Arc::new(Mutex::new(mpsc::channel().1)),
            buffer: StreamBuffer::new(buffer_capacity),
        }
    }

    /// Send token event
    pub fn send_token(&self, token: String) -> Result<(), String> {
        self.buffer.add_token(token.clone());

        if self.buffer.is_full() {
            let tokens = self.buffer.flush();
            let combined = tokens.join("");
            self.tx
                .send(StreamEvent::token(combined))
                .map_err(|e| e.to_string())
        } else {
            self.tx
                .send(StreamEvent::delta(token))
                .map_err(|e| e.to_string())
        }
    }

    /// Send delta event (partial token)
    pub fn send_delta(&self, delta: String) -> Result<(), String> {
        self.tx
            .send(StreamEvent::delta(delta))
            .map_err(|e| e.to_string())
    }

    /// Mark streaming as complete
    pub fn finish(&self) -> Result<(), String> {
        // Flush any remaining tokens
        let remaining = self.buffer.flush();
        if !remaining.is_empty() {
            let combined = remaining.join("");
            self.tx
                .send(StreamEvent::token(combined))
                .map_err(|e| e.to_string())?;
        }

        self.tx.send(StreamEvent::done()).map_err(|e| e.to_string())
    }

    /// Send error
    pub fn error(&self, msg: String) -> Result<(), String> {
        self.tx
            .send(StreamEvent::error(msg))
            .map_err(|e| e.to_string())
    }
}

/// Token stream for collecting streamed responses
pub struct TokenStream {
    receiver: mpsc::Receiver<StreamEvent>,
    tokens: Vec<String>,
    is_complete: bool,
}

impl TokenStream {
    pub fn new(receiver: mpsc::Receiver<StreamEvent>) -> Self {
        Self {
            receiver,
            tokens: Vec::new(),
            is_complete: false,
        }
    }

    /// Get next event from stream
    pub fn next_event(&mut self) -> Option<StreamEvent> {
        match self.receiver.recv() {
            Ok(event) => {
                match &event {
                    StreamEvent::Token(t) => self.tokens.push(t.clone()),
                    StreamEvent::Delta(d) => {
                        // Accumulate delta if previous token exists
                        if let Some(last) = self.tokens.last_mut() {
                            last.push_str(d);
                        } else {
                            self.tokens.push(d.clone());
                        }
                    }
                    StreamEvent::Done => self.is_complete = true,
                    StreamEvent::Error(_) => self.is_complete = true,
                }
                Some(event)
            }
            Err(_) => None,
        }
    }

    /// Collect all events
    pub fn collect_all(&mut self) -> Result<String, String> {
        while let Some(event) = self.next_event() {
            if let StreamEvent::Error(msg) = event {
                return Err(msg);
            }
        }
        Ok(self.tokens.join(""))
    }

    /// Get collected tokens
    pub fn get_tokens(&self) -> &[String] {
        &self.tokens
    }

    /// Get complete output
    pub fn get_output(&self) -> String {
        self.tokens.join("")
    }

    /// Check if streaming is complete
    pub fn is_complete(&self) -> bool {
        self.is_complete
    }
}

/// Backpressure handler for stream throttling
pub struct BackpressureHandler {
    max_buffer_size: usize,
    current_buffer_size: usize,
}

impl BackpressureHandler {
    pub fn new(max_buffer_size: usize) -> Self {
        Self {
            max_buffer_size,
            current_buffer_size: 0,
        }
    }

    /// Check if sender can send (no backpressure)
    pub fn can_send(&self) -> bool {
        self.current_buffer_size < self.max_buffer_size
    }

    /// Update buffer size
    pub fn update_buffer_size(&mut self, size: usize) {
        self.current_buffer_size = size;
    }

    /// Apply backpressure (wait for buffer to drain)
    pub fn apply_backpressure(&mut self) {
        if self.current_buffer_size >= self.max_buffer_size {
            // Simulate waiting for consumer to drain buffer
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
    }

    /// Get buffer fill percentage
    pub fn buffer_fill_percent(&self) -> f32 {
        (self.current_buffer_size as f32 / self.max_buffer_size as f32) * 100.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stream_event_creation() {
        let event = StreamEvent::token("hello".to_string());
        match event {
            StreamEvent::Token(t) => assert_eq!(t, "hello"),
            _ => panic!("Wrong event type"),
        }
    }

    #[test]
    fn test_stream_buffer_creation() {
        let buffer = StreamBuffer::new(4);
        assert_eq!(buffer.size(), 0);
    }

    #[test]
    fn test_stream_buffer_add_token() {
        let buffer = StreamBuffer::new(4);
        buffer.add_token("hello".to_string());
        assert_eq!(buffer.size(), 1);
    }

    #[test]
    fn test_stream_buffer_full() {
        let buffer = StreamBuffer::new(2);
        buffer.add_token("token1".to_string());
        buffer.add_token("token2".to_string());
        assert!(buffer.is_full());
    }

    #[test]
    fn test_stream_buffer_flush() {
        let buffer = StreamBuffer::new(4);
        buffer.add_token("a".to_string());
        buffer.add_token("b".to_string());

        let tokens = buffer.flush();
        assert_eq!(tokens.len(), 2);
        assert_eq!(buffer.size(), 0);
    }

    #[test]
    fn test_streaming_response_send_token() {
        let (response, receiver) = mpsc::channel::<StreamEvent>();
        let stream = StreamingResponse {
            tx: response,
            rx: Arc::new(Mutex::new(receiver)),
            buffer: StreamBuffer::new(4),
        };

        let result = stream.send_token("test".to_string());
        assert!(result.is_ok());
    }

    #[test]
    fn test_streaming_response_finish() {
        let (tx, _rx) = mpsc::channel::<StreamEvent>();
        let stream = StreamingResponse {
            tx,
            rx: Arc::new(Mutex::new(mpsc::channel().1)),
            buffer: StreamBuffer::new(4),
        };

        let result = stream.finish();
        assert!(result.is_ok());
    }

    #[test]
    fn test_token_stream_creation() {
        let (_tx, rx) = mpsc::channel();
        let stream = TokenStream::new(rx);
        assert!(!stream.is_complete());
    }

    #[test]
    fn test_token_stream_collect() {
        let (tx, rx) = mpsc::channel();
        let mut stream = TokenStream::new(rx);

        let _ = tx.send(StreamEvent::token("hello".to_string()));
        let _ = tx.send(StreamEvent::token("world".to_string()));
        let _ = tx.send(StreamEvent::done());
        drop(tx);

        let output = stream.collect_all();
        assert!(output.is_ok());
        assert_eq!(stream.get_output(), "helloworld");
    }

    #[test]
    fn test_backpressure_handler_creation() {
        let handler = BackpressureHandler::new(1000);
        assert!(handler.can_send());
    }

    #[test]
    fn test_backpressure_handler_full() {
        let mut handler = BackpressureHandler::new(100);
        handler.update_buffer_size(100);
        assert!(!handler.can_send());
    }

    #[test]
    fn test_backpressure_handler_fill_percent() {
        let mut handler = BackpressureHandler::new(1000);
        handler.update_buffer_size(500);
        let fill = handler.buffer_fill_percent();
        assert!(fill > 49.0 && fill < 51.0); // ~50%
    }
}
