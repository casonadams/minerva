/// Streaming Event Types - Phase 10 Day 5
///
/// Event types and structures for real-time token streaming.
pub use crate::inference::stream_event_type::StreamEventType;
use serde::{Deserialize, Serialize};
use std::time::{SystemTime, UNIX_EPOCH};

/// Token event data
#[derive(Debug, Clone)]
pub struct TokenData {
    pub token_id: u32,
    pub token: String,
    pub index: u32,
    pub total: u32,
    pub tps: f32,
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
    pub fn token(stream_id: String, data: TokenData) -> Self {
        Self {
            event: StreamEventType::Token,
            stream_id,
            token_id: Some(data.token_id),
            token: Some(data.token),
            index: data.index,
            total_tokens: data.total,
            tokens_per_second: data.tps,
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
        let e = StreamEvent::token(
            "s1".to_string(),
            TokenData {
                token_id: 42,
                token: "hi".to_string(),
                index: 0,
                total: 1,
                tps: 25.0,
            },
        );
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
