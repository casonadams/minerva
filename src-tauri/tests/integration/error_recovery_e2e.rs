// Error Recovery and End-to-End Pipeline Integration Tests

use minerva_lib::inference::llama_adapter::GenerationParams;
use std::fs;
use std::path::PathBuf;
use tempfile::TempDir;

fn setup_temp_model() -> (TempDir, PathBuf) {
    let temp_dir = TempDir::new().unwrap();
    let model_path = temp_dir.path().join("test_model.gguf");
    fs::write(&model_path, "dummy").unwrap();
    (temp_dir, model_path)
}

// Error Recovery Tests

#[test]
fn test_error_recovery_gpu_oom() {
    use minerva_lib::error::MinervaError;
    use minerva_lib::error_recovery::ErrorRecovery;

    let err = MinervaError::GpuOutOfMemory("requested: 5000MB, available: 1000MB".to_string());

    assert!(ErrorRecovery::is_recoverable(&err));
    let strategy = ErrorRecovery::strategy_for(&err);
    assert!(!format!("{:?}", strategy).is_empty());
}

#[test]
fn test_error_recovery_gpu_context_lost() {
    use minerva_lib::error::MinervaError;
    use minerva_lib::error_recovery::ErrorRecovery;

    let err = MinervaError::GpuContextLost("device removed".to_string());
    assert!(ErrorRecovery::is_recoverable(&err));
}

#[test]
fn test_error_recovery_model_corrupted() {
    use minerva_lib::error::MinervaError;
    use minerva_lib::error_recovery::ErrorRecovery;

    let err = MinervaError::ModelCorrupted("test.gguf".to_string());
    assert!(ErrorRecovery::is_recoverable(&err));
}

#[test]
fn test_error_recovery_gpu_oom_fallback() {
    use minerva_lib::error::MinervaError;
    use minerva_lib::error_recovery::ErrorRecovery;

    let err = MinervaError::GpuOutOfMemory("requested: 5000MB, available: 1000MB".to_string());

    let strategy = ErrorRecovery::strategy_for(&err);
    assert!(!format!("{:?}", strategy).is_empty());
}

#[test]
fn test_error_recovery_backoff_calculation() {
    use minerva_lib::error_recovery::ErrorRecovery;

    let backoff1 = ErrorRecovery::backoff_delay(0, 100);
    let backoff2 = ErrorRecovery::backoff_delay(1, 100);

    assert!(backoff2 > backoff1);
}

#[test]
fn test_error_recovery_is_gpu_error() {
    use minerva_lib::error::MinervaError;
    use minerva_lib::error_recovery::ErrorRecovery;

    let gpu_err = MinervaError::GpuOutOfMemory("requested: 5000MB, available: 1000MB".to_string());

    assert!(ErrorRecovery::is_gpu_error(&gpu_err));
}

#[test]
fn test_error_recovery_is_recoverable() {
    use minerva_lib::error::MinervaError;
    use minerva_lib::error_recovery::ErrorRecovery;

    let recoverable =
        MinervaError::GpuOutOfMemory("requested: 5000MB, available: 1000MB".to_string());
    assert!(ErrorRecovery::is_recoverable(&recoverable));
}

#[test]
fn test_error_recovery_streaming_error_recovery() {
    use minerva_lib::error::MinervaError;
    use minerva_lib::error_recovery::ErrorRecovery;

    let err = MinervaError::StreamingError("test".to_string());
    assert!(ErrorRecovery::is_recoverable(&err));
}

#[test]
fn test_error_recovery_context_limit_fatal() {
    use minerva_lib::error::MinervaError;
    use minerva_lib::error_recovery::RecoveryStrategy;

    let err = MinervaError::ContextLimitExceeded {
        max: 2048,
        required: 4096,
    };

    // This should be fatal strategy
    use minerva_lib::error_recovery::ErrorRecovery;
    let strategy = ErrorRecovery::strategy_for(&err);
    assert!(matches!(strategy, RecoveryStrategy::Fatal));
}

// End-to-End Pipeline Tests

#[test]
fn test_full_inference_pipeline() {
    use minerva_lib::inference::InferenceEngine;

    let (_temp, model_path) = setup_temp_model();
    let mut engine = InferenceEngine::new(model_path);

    assert!(engine.load_model().is_ok());
    assert!(engine.is_loaded());

    let response = engine.generate("Hello").unwrap();
    assert!(!response.is_empty());

    engine.unload_model();
    assert!(!engine.is_loaded());
}

#[test]
fn test_inference_engine_lifecycle() {
    use minerva_lib::inference::InferenceEngine;

    let (_temp, model_path) = setup_temp_model();
    let mut engine = InferenceEngine::new(model_path);

    assert!(!engine.is_loaded());
    assert!(engine.load_model().is_ok());
    assert!(engine.is_loaded());
    engine.unload_model();
    assert!(!engine.is_loaded());
}

#[test]
fn test_full_inference_with_error_handling() {
    use minerva_lib::error_recovery::ErrorRecovery;
    use minerva_lib::inference::llama_adapter::{InferenceBackend, MockBackend};

    let (_temp, model_path) = setup_temp_model();
    let mut backend = MockBackend::new();

    assert!(backend.load_model(&model_path, 2048).is_ok());
    assert!(backend.is_loaded());

    let params = GenerationParams {
        max_tokens: 100,
        temperature: 0.7,
        top_p: 0.9,
    };
    let result = backend.generate("test prompt", params);
    assert!(result.is_ok());

    if let Err(err) = result {
        assert!(ErrorRecovery::is_recoverable(&err));
    }
}

#[test]
fn test_performance_metrics_tracking() {
    use minerva_lib::inference::benchmarks::{PerformanceMetrics, PerformanceMetricsInput};
    use std::time::Duration;

    let input = PerformanceMetricsInput {
        duration: Duration::from_millis(150),
        token_count: 3,
        memory_bytes: 1_000_000,
        gpu_used: true,
    };
    let metrics = PerformanceMetrics::new(input);

    assert!(metrics.duration.as_millis() > 0);
    assert_eq!(metrics.token_count, 3);
    assert!(metrics.tokens_per_sec > 0.0);
}

#[test]
fn test_context_exceeds_limits() {
    use minerva_lib::error::MinervaError;
    use minerva_lib::error_recovery::ErrorRecovery;

    let err = MinervaError::ContextLimitExceeded {
        max: 2048,
        required: 4096,
    };

    let strategy = ErrorRecovery::strategy_for(&err);
    use minerva_lib::error_recovery::RecoveryStrategy;
    assert!(matches!(strategy, RecoveryStrategy::Fatal));

    assert!(ErrorRecovery::is_recoverable(&err));
}
