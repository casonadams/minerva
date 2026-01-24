use serde::{Deserialize, Serialize};

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
