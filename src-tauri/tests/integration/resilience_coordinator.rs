use minerva::error::MinervaError;
use minerva::resilience::{
    circuit_breaker::CircuitBreakerConfig, coordinator_decision::CoordinatorDecision,
    CircuitBreaker, DecisionContext, ErrorClass, TimeoutContext,
};
use std::time::Duration;

#[test]
fn test_decision_transient_error() {
    let cb = CircuitBreaker::new(CircuitBreakerConfig::default());
    let err = MinervaError::StreamingError("test".to_string());
    let mut retry = None;

    let decision = CoordinatorDecision::decide(DecisionContext {
        circuit_breaker: &cb,
        retry_state: &mut retry,
        timeout_context: &None,
        error: &err,
    });
    assert_eq!(decision.error_class, ErrorClass::Transient);
}

#[test]
fn test_decision_permanent_error() {
    let cb = CircuitBreaker::new(CircuitBreakerConfig::default());
    let err = MinervaError::InvalidRequest("test".to_string());
    let mut retry = None;

    let decision = CoordinatorDecision::decide(DecisionContext {
        circuit_breaker: &cb,
        retry_state: &mut retry,
        timeout_context: &None,
        error: &err,
    });
    assert_eq!(decision.error_class, ErrorClass::Permanent);
    assert!(decision.should_fail_fast);
}

#[test]
fn test_decision_circuit_open() {
    let cfg = CircuitBreakerConfig {
        failure_threshold: 1,
        ..Default::default()
    };
    let cb = CircuitBreaker::new(cfg);
    cb.record_failure();

    let err = MinervaError::StreamingError("test".to_string());
    let mut retry = None;
    let decision = CoordinatorDecision::decide(DecisionContext {
        circuit_breaker: &cb,
        retry_state: &mut retry,
        timeout_context: &None,
        error: &err,
    });

    assert!(decision.should_fail_fast);
}

#[test]
fn test_decision_timeout_exceeded() {
    let cb = CircuitBreaker::new(CircuitBreakerConfig::default());
    let ctx = TimeoutContext::new(Duration::from_millis(1), Duration::from_secs(10));
    std::thread::sleep(Duration::from_millis(10));

    let err = MinervaError::StreamingError("test".to_string());
    let mut retry = None;
    let decision = CoordinatorDecision::decide(DecisionContext {
        circuit_breaker: &cb,
        retry_state: &mut retry,
        timeout_context: &Some(ctx),
        error: &err,
    });

    assert!(decision.should_fail_fast);
}

#[test]
fn test_decision_fallback_available() {
    let cb = CircuitBreaker::new(CircuitBreakerConfig::default());
    let err = MinervaError::GpuOutOfMemory("test".to_string());
    let mut retry = None;

    let decision = CoordinatorDecision::decide(DecisionContext {
        circuit_breaker: &cb,
        retry_state: &mut retry,
        timeout_context: &None,
        error: &err,
    });
    assert!(decision.should_fallback);
}
