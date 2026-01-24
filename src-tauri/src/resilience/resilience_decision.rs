use super::ErrorClass;

/// Resilience decision for an operation
#[derive(Debug, Clone)]
pub struct ResilienceDecision {
    /// Error classification
    pub error_class: ErrorClass,
    /// Should retry?
    pub should_retry: bool,
    /// Should fallback?
    pub should_fallback: bool,
    /// Should fail fast?
    pub should_fail_fast: bool,
    /// Time to wait before retry
    pub retry_delay_ms: Option<u64>,
}
