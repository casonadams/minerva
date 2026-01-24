//! Error recovery tests

use super::handler::ErrorRecovery;
use super::types::RecoveryStrategy;
use crate::error::MinervaError;
use std::time::Duration;

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
