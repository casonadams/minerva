use super::{
    ErrorClass, circuit_breaker::CircuitBreaker, fallback::FallbackDecision,
    resilience_decision::ResilienceDecision, retry::RetryState, timeout::TimeoutContext,
};
use crate::error::MinervaError;

/// Context for making resilience decisions
pub struct DecisionContext<'a> {
    pub circuit_breaker: &'a CircuitBreaker,
    pub retry_state: &'a mut Option<RetryState>,
    pub timeout_context: &'a Option<TimeoutContext>,
    pub error: &'a MinervaError,
}

/// Decision logic for resilience coordination
pub struct CoordinatorDecision;

impl CoordinatorDecision {
    /// Make resilience decision for an error
    pub fn decide(ctx: DecisionContext) -> ResilienceDecision {
        let DecisionContext {
            circuit_breaker,
            retry_state,
            timeout_context,
            error,
        } = ctx;
        let error_class = ErrorClass::classify(error);

        // Check if circuit is open
        if !circuit_breaker.allow_request() {
            return ResilienceDecision {
                error_class,
                should_retry: false,
                should_fallback: false,
                should_fail_fast: true,
                retry_delay_ms: None,
            };
        }

        // Check timeout
        if let Some(ctx) = timeout_context
            && ctx.is_deadline_exceeded()
        {
            return ResilienceDecision {
                error_class,
                should_retry: false,
                should_fallback: false,
                should_fail_fast: true,
                retry_delay_ms: None,
            };
        }

        // Determine retry strategy
        let should_retry = if let Some(retry) = retry_state {
            retry.can_retry()
        } else {
            false
        };

        // Fallback available?
        let should_fallback = !matches!(
            FallbackDecision::strategy_for(error),
            super::fallback::FallbackStrategy::None
        );

        // Calculate retry delay if applicable
        let retry_delay_ms = if should_retry {
            retry_state
                .clone()
                .map(|mut retry| retry.next_delay().as_millis() as u64)
        } else {
            None
        };

        ResilienceDecision {
            error_class,
            should_retry,
            should_fallback,
            should_fail_fast: !error_class.is_recoverable(),
            retry_delay_ms,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resilience::circuit_breaker::CircuitBreakerConfig;
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
}
