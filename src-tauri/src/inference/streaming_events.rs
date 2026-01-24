/// Streaming Event Types - Phase 10 Day 5
///
/// Event types and structures for real-time token streaming.
use serde::{Deserialize, Serialize};
use std::time::{SystemTime, UNIX_EPOCH};

// ============================================================================
// Event Types
// ============================================================================

/// Streaming event type
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum StreamEventType {
    Start,
    Token,
    Complete,
    Error,
    Cancelled,
}

impl std::fmt::Display for StreamEventType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StreamEventType::Start => write!(f, "start"),
            StreamEventType::Token => write!(f, "token"),
            StreamEventType::Complete => write!(f, "complete"),
            StreamEventType::Error => write!(f, "error"),
            StreamEventType::Cancelled => write!(f, "cancelled"),
        }
    }
}

/// Stream event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamEvent {
    pub event: StreamEventType,
    pub stream_id: String,
    pub token_id: Option<u32>,
    pub token: Option<String>,
    pub index: u32,
    pub total_tokens: u32,
    pub tokens_per_second: f32,
    pub error: Option<String>,
    pub timestamp: u64,
}

impl StreamEvent {
    /// Start event
    pub fn start(stream_id: String) -> Self {
        Self {
            event: StreamEventType::Start,
            stream_id,
            token_id: None,
            token: None,
            index: 0,
            total_tokens: 0,
            tokens_per_second: 0.0,
            error: None,
            timestamp: current_timestamp(),
        }
    }

    /// Token event
    pub fn token(
        stream_id: String,
        token_id: u32,
        token: String,
        index: u32,
        total: u32,
        tps: f32,
    ) -> Self {
        Self {
            event: StreamEventType::Token,
            stream_id,
            token_id: Some(token_id),
            token: Some(token),
            index,
            total_tokens: total,
            tokens_per_second: tps,
            error: None,
            timestamp: current_timestamp(),
        }
    }

    /// Complete event
    pub fn complete(stream_id: String, total: u32, tps: f32) -> Self {
        Self {
            event: StreamEventType::Complete,
            stream_id,
            token_id: None,
            token: None,
            index: total,
            total_tokens: total,
            tokens_per_second: tps,
            error: None,
            timestamp: current_timestamp(),
        }
    }

    /// Error event
    pub fn error(stream_id: String, error: String) -> Self {
        Self {
            event: StreamEventType::Error,
            stream_id,
            token_id: None,
            token: None,
            index: 0,
            total_tokens: 0,
            tokens_per_second: 0.0,
            error: Some(error),
            timestamp: current_timestamp(),
        }
    }

    /// Format as SSE
    pub fn as_sse(&self) -> String {
        let json = serde_json::to_string(self).unwrap_or_default();
        format!("event: {}\ndata: {}\n\n", self.event, json)
    }
}

fn current_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_start_event() {
        let e = StreamEvent::start("s1".to_string());
        assert_eq!(e.event, StreamEventType::Start);
    }

    #[test]
    fn test_token_event() {
        let e = StreamEvent::token("s1".to_string(), 42, "hi".to_string(), 0, 1, 25.0);
        assert_eq!(e.event, StreamEventType::Token);
        assert_eq!(e.token_id, Some(42));
    }

    #[test]
    fn test_complete_event() {
        let e = StreamEvent::complete("s1".to_string(), 100, 30.0);
        assert_eq!(e.event, StreamEventType::Complete);
    }

    #[test]
    fn test_error_event() {
        let e = StreamEvent::error("s1".to_string(), "err".to_string());
        assert_eq!(e.event, StreamEventType::Error);
        assert_eq!(e.error, Some("err".to_string()));
    }

    #[test]
    fn test_as_sse() {
        let e = StreamEvent::start("s1".to_string());
        let sse = e.as_sse();
        assert!(sse.contains("event: start"));
    }
}
