//! Streaming Response Support
//! Handles SSE (Server-Sent Events) and streaming responses for token generation

pub mod handler;
pub mod types;
pub mod validator;

pub use handler::{create_streaming_events, format_streaming_event};
pub use types::{ChatCompletionStreamEvent, StreamingChoice, StreamingConfig, StreamingDelta};
pub use validator::StreamingValidator;
