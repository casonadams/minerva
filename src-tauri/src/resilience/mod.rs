/// Error Recovery & Resilience Patterns for Phase 7 Step 2
///
/// This module provides production-grade resilience patterns:
/// - Retry logic with exponential backoff and jitter
/// - Circuit breaker pattern for failing operations
/// - Fallback mechanisms for graceful degradation
/// - Error categorization (recoverable vs fatal)
/// - Timeout management and deadline propagation
pub mod circuit_breaker;
pub mod circuit_breaker_config;
pub mod circuit_breaker_transitions;
pub mod circuit_state_transitions;
pub mod component_health;
pub mod coordinator;
pub mod coordinator_decision;
pub mod fallback;
pub mod fallback_health;
pub mod fallback_strategy;
pub mod health;
pub mod health_status;
pub mod resilience_decision;
pub mod retry;
pub mod retry_config;
pub mod retry_state;
pub mod timeout;
pub mod timeout_context;
pub mod timeout_manager;
pub mod timeout_stats;

use crate::error::MinervaError;
use std::time::Duration;

/// Error classification for resilience decisions
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorClass {
    /// Recoverable: transient failure, can be retried
    Transient,
    /// Recoverable: resource exhaustion, can fallback
    ResourceExhausted,
    /// Unrecoverable: permanent failure, should not retry
    Permanent,
    /// Fatal: stop all operations
    Fatal,
}

impl ErrorClass {
    /// Classify an error for resilience handling
    pub fn classify(error: &MinervaError) -> Self {
        match error {
            // Transient errors: retry with backoff
            MinervaError::StreamingError(_) | MinervaError::GenerationTimeout => {
                ErrorClass::Transient
            }

            // Resource exhaustion: try fallback
            MinervaError::GpuOutOfMemory(_)
            | MinervaError::OutOfMemory(_)
            | MinervaError::GpuContextLost(_) => ErrorClass::ResourceExhausted,

            // Permanent errors: don't retry
            MinervaError::ModelNotFound(_)
            | MinervaError::InvalidRequest(_)
            | MinervaError::ModelCorrupted(_) => ErrorClass::Permanent,

            // Fatal: stop
            MinervaError::ContextLimitExceeded { .. } => ErrorClass::Fatal,

            // Default: permanent
            _ => ErrorClass::Permanent,
        }
    }

    /// Is this error recoverable at all?
    pub fn is_recoverable(self) -> bool {
        matches!(self, ErrorClass::Transient | ErrorClass::ResourceExhausted)
    }
}

/// Timeout configuration for operations
#[derive(Debug, Clone, Copy)]
pub struct TimeoutConfig {
    /// Maximum time for a single operation
    pub operation_timeout: Duration,
    /// Maximum time for entire request (includes retries)
    pub total_timeout: Duration,
}

impl Default for TimeoutConfig {
    fn default() -> Self {
        Self {
            operation_timeout: Duration::from_secs(30),
            total_timeout: Duration::from_secs(60),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_classify_transient_error() {
        let err = MinervaError::StreamingError("network".to_string());
        assert_eq!(ErrorClass::classify(&err), ErrorClass::Transient);
    }

    #[test]
    fn test_classify_resource_exhausted() {
        let err = MinervaError::GpuOutOfMemory("16GB".to_string());
        assert_eq!(ErrorClass::classify(&err), ErrorClass::ResourceExhausted);
    }

    #[test]
    fn test_classify_permanent_error() {
        let err = MinervaError::InvalidRequest("bad input".to_string());
        assert_eq!(ErrorClass::classify(&err), ErrorClass::Permanent);
    }

    #[test]
    fn test_classify_fatal_error() {
        let err = MinervaError::ContextLimitExceeded {
            max: 2048,
            required: 8192,
        };
        assert_eq!(ErrorClass::classify(&err), ErrorClass::Fatal);
    }

    #[test]
    fn test_is_recoverable() {
        assert!(ErrorClass::Transient.is_recoverable());
        assert!(ErrorClass::ResourceExhausted.is_recoverable());
        assert!(!ErrorClass::Permanent.is_recoverable());
        assert!(!ErrorClass::Fatal.is_recoverable());
    }

    #[test]
    fn test_timeout_config_default() {
        let cfg = TimeoutConfig::default();
        assert_eq!(cfg.operation_timeout, Duration::from_secs(30));
        assert_eq!(cfg.total_timeout, Duration::from_secs(60));
    }
}
