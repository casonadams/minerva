//! Error recovery handler

use super::types::RecoveryStrategy;
use crate::error::MinervaError;
use std::time::Duration;

/// Error recovery handler
pub struct ErrorRecovery;

impl ErrorRecovery {
    /// Determine recovery strategy for an error
    pub fn strategy_for(error: &MinervaError) -> RecoveryStrategy {
        match error {
            MinervaError::StreamingError(_) => RecoveryStrategy::Retry {
                max_attempts: 3,
                backoff_ms: 100,
            },
            MinervaError::GpuOutOfMemory(_) => RecoveryStrategy::FallbackToCpu,
            MinervaError::GpuContextLost(_) => RecoveryStrategy::ReinitializeGpu,
            MinervaError::ModelCorrupted(_) => RecoveryStrategy::ReloadModel,
            MinervaError::GenerationTimeout => RecoveryStrategy::Retry {
                max_attempts: 2,
                backoff_ms: 500,
            },
            MinervaError::ContextLimitExceeded { .. } => RecoveryStrategy::Fatal,
            _ => RecoveryStrategy::Fatal,
        }
    }

    /// Get human-readable recovery message
    pub fn recovery_message(strategy: RecoveryStrategy) -> &'static str {
        match strategy {
            RecoveryStrategy::Retry { .. } => "Retrying operation with backoff...",
            RecoveryStrategy::FallbackToCpu => "GPU unavailable, falling back to CPU inference...",
            RecoveryStrategy::ReinitializeGpu => "Reinitializing GPU context...",
            RecoveryStrategy::ReloadModel => "Reloading model from disk...",
            RecoveryStrategy::SkipAndContinue => "Skipping operation and continuing...",
            RecoveryStrategy::Fatal => "Fatal error - stopping operation.",
        }
    }

    /// Calculate backoff delay for retry attempt
    pub fn backoff_delay(attempt: u32, base_ms: u64) -> Duration {
        let delay_ms = base_ms * u64::pow(2, attempt);
        Duration::from_millis(delay_ms)
    }

    /// Check if error is recoverable
    pub fn is_recoverable(error: &MinervaError) -> bool {
        !matches!(
            error,
            MinervaError::ModelNotFound(_) | MinervaError::InvalidRequest(_)
        )
    }

    /// Check if error indicates resource exhaustion (can fallback)
    pub fn is_resource_exhaustion(error: &MinervaError) -> bool {
        matches!(
            error,
            MinervaError::OutOfMemory(_) | MinervaError::GpuOutOfMemory(_)
        )
    }

    /// Check if error is GPU-related
    pub fn is_gpu_error(error: &MinervaError) -> bool {
        matches!(
            error,
            MinervaError::GpuOutOfMemory(_) | MinervaError::GpuContextLost(_)
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_streaming_error_recovery() {
        let err = MinervaError::StreamingError("connection lost".to_string());
        let strategy = ErrorRecovery::strategy_for(&err);
        assert!(matches!(strategy, RecoveryStrategy::Retry { .. }));
    }

    #[test]
    fn test_gpu_oom_fallback() {
        let err = MinervaError::GpuOutOfMemory("16GB exceeded".to_string());
        let strategy = ErrorRecovery::strategy_for(&err);
        assert_eq!(strategy, RecoveryStrategy::FallbackToCpu);
    }

    #[test]
    fn test_gpu_context_lost() {
        let err = MinervaError::GpuContextLost("device removed".to_string());
        let strategy = ErrorRecovery::strategy_for(&err);
        assert_eq!(strategy, RecoveryStrategy::ReinitializeGpu);
    }

    #[test]
    fn test_model_corrupted() {
        let err = MinervaError::ModelCorrupted("invalid header".to_string());
        let strategy = ErrorRecovery::strategy_for(&err);
        assert_eq!(strategy, RecoveryStrategy::ReloadModel);
    }

    #[test]
    fn test_context_limit_fatal() {
        let err = MinervaError::ContextLimitExceeded {
            max: 2048,
            required: 4096,
        };
        let strategy = ErrorRecovery::strategy_for(&err);
        assert_eq!(strategy, RecoveryStrategy::Fatal);
    }

    #[test]
    fn test_backoff_calculation() {
        assert_eq!(
            ErrorRecovery::backoff_delay(0, 100),
            Duration::from_millis(100)
        );
        assert_eq!(
            ErrorRecovery::backoff_delay(1, 100),
            Duration::from_millis(200)
        );
        assert_eq!(
            ErrorRecovery::backoff_delay(2, 100),
            Duration::from_millis(400)
        );
    }

    #[test]
    fn test_is_recoverable() {
        let recoverable = MinervaError::StreamingError("test".to_string());
        assert!(ErrorRecovery::is_recoverable(&recoverable));
        let not_recoverable = MinervaError::InvalidRequest("test".to_string());
        assert!(!ErrorRecovery::is_recoverable(&not_recoverable));
    }

    #[test]
    fn test_is_gpu_error() {
        let gpu_err = MinervaError::GpuOutOfMemory("test".to_string());
        assert!(ErrorRecovery::is_gpu_error(&gpu_err));
        let other_err = MinervaError::StreamingError("test".to_string());
        assert!(!ErrorRecovery::is_gpu_error(&other_err));
    }

    #[test]
    fn test_recovery_messages() {
        let msg = ErrorRecovery::recovery_message(RecoveryStrategy::FallbackToCpu);
        assert!(msg.contains("CPU"));
        let msg = ErrorRecovery::recovery_message(RecoveryStrategy::Retry {
            max_attempts: 3,
            backoff_ms: 100,
        });
        assert!(msg.contains("Retrying"));
    }
}
