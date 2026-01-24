/// Retry logic with exponential backoff and jitter
///
/// Implements the retry pattern with:
/// - Configurable max attempts
/// - Exponential backoff (2^n * base_ms)
/// - Jitter to prevent thundering herd
/// - Full jitter algorithm (random between 0 and backoff)
pub use crate::resilience::retry_config::RetryConfig;
pub use crate::resilience::retry_state::RetryState;
