use std::sync::Arc;

/// Token stream callback for real-time streaming
///
/// This callback is invoked whenever a new token is generated.
/// Used for real-time streaming to clients via Server-Sent Events (SSE).
pub type TokenCallback = Arc<dyn Fn(String) + Send + Sync>;
