//! Error recovery types

/// Recovery strategy for different error types
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RecoveryStrategy {
    /// Retry the operation (e.g., streaming, timeout)
    Retry { max_attempts: u32, backoff_ms: u64 },
    /// Fallback to CPU if GPU fails
    FallbackToCpu,
    /// Reinitialize GPU context
    ReinitializeGpu,
    /// Reload the model
    ReloadModel,
    /// Skip and continue (non-critical)
    SkipAndContinue,
    /// Fatal error - stop
    Fatal,
}
