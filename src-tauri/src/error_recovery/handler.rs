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
