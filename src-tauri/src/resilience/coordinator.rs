use super::{
    circuit_breaker::CircuitBreaker, fallback::FallbackDecision, retry::RetryState,
    timeout::TimeoutContext, ErrorClass, resilience_decision::ResilienceDecision,
};
use crate::error::MinervaError;

/// Resilience coordinator for orchestrating patterns
pub struct ResilienceCoordinator {
    circuit_breaker: CircuitBreaker,
    retry_state: Option<RetryState>,
    timeout_context: Option<TimeoutContext>,
}

impl ResilienceCoordinator {
    /// Create new coordinator
    pub fn new(circuit_breaker: CircuitBreaker) -> Self {
        Self {
            circuit_breaker,
            retry_state: None,
            timeout_context: None,
        }
    }

    /// Set timeout context
    pub fn with_timeout(mut self, ctx: TimeoutContext) -> Self {
        self.timeout_context = Some(ctx);
        self
    }

    /// Set retry state
    pub fn with_retry(mut self, retry: RetryState) -> Self {
        self.retry_state = Some(retry);
        self
    }

    /// Make resilience decision for an error
    pub fn decide(&mut self, error: &MinervaError) -> ResilienceDecision {
        let error_class = ErrorClass::classify(error);

        // Check if circuit is open
        if !self.circuit_breaker.allow_request() {
            return ResilienceDecision {
                error_class,
                should_retry: false,
                should_fallback: false,
                should_fail_fast: true,
                retry_delay_ms: None,
            };
        }

        // Check timeout
        if let Some(ref ctx) = self.timeout_context
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
        let should_retry = if let Some(ref mut retry) = self.retry_state {
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
            self.retry_state
                .as_mut()
                .map(|retry| retry.next_delay().as_millis() as u64)
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

    /// Record success
    pub fn record_success(&self) {
        self.circuit_breaker.record_success();
    }

    /// Record failure
    pub fn record_failure(&self) {
        self.circuit_breaker.record_failure();
    }

    /// Get circuit breaker state
    pub fn circuit_state(&self) -> super::circuit_breaker::CircuitState {
        self.circuit_breaker.state()
    }

    /// Is timed out?
    pub fn is_timed_out(&self) -> bool {
        if let Some(ref ctx) = self.timeout_context {
            ctx.is_deadline_exceeded()
        } else {
            false
        }
    }

    /// Get remaining attempts
    pub fn remaining_attempts(&self) -> Option<u32> {
        self.retry_state.as_ref().map(|r| r.remaining())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resilience::circuit_breaker::CircuitBreakerConfig;
    use std::time::Duration;

    #[test]
    fn test_coordinator_creation() {
        let cb = CircuitBreaker::new(CircuitBreakerConfig::default());
        let coord = ResilienceCoordinator::new(cb);
        assert!(!coord.is_timed_out());
    }

    #[test]
    fn test_coordinator_transient_error() {
        let cb = CircuitBreaker::new(CircuitBreakerConfig::default());
        let mut coord = ResilienceCoordinator::new(cb);

        let err = MinervaError::StreamingError("test".to_string());
        let decision = coord.decide(&err);

        assert_eq!(decision.error_class, ErrorClass::Transient);
    }

    #[test]
    fn test_coordinator_permanent_error() {
        let cb = CircuitBreaker::new(CircuitBreakerConfig::default());
        let mut coord = ResilienceCoordinator::new(cb);

        let err = MinervaError::InvalidRequest("test".to_string());
        let decision = coord.decide(&err);

        assert_eq!(decision.error_class, ErrorClass::Permanent);
        assert!(decision.should_fail_fast);
    }

    #[test]
    fn test_coordinator_circuit_open() {
        let cfg = CircuitBreakerConfig {
            failure_threshold: 1,
            ..Default::default()
        };
        let cb = CircuitBreaker::new(cfg);
        cb.record_failure();

        let mut coord = ResilienceCoordinator::new(cb);
        let err = MinervaError::StreamingError("test".to_string());
        let decision = coord.decide(&err);

        assert!(decision.should_fail_fast);
    }

    #[test]
    fn test_coordinator_with_timeout() {
        let cb = CircuitBreaker::new(CircuitBreakerConfig::default());
        let ctx = TimeoutContext::new(Duration::from_millis(1), Duration::from_secs(10));

        let coord = ResilienceCoordinator::new(cb).with_timeout(ctx);
        std::thread::sleep(Duration::from_millis(10));

        assert!(coord.is_timed_out());
    }

    #[test]
    fn test_coordinator_record_success() {
        let cb = CircuitBreaker::new(CircuitBreakerConfig::default());
        cb.record_failure();
        cb.record_failure();

        let coord = ResilienceCoordinator::new(cb);
        coord.record_success();

        let cb = &coord.circuit_breaker;
        assert_eq!(cb.failures(), 0);
    }

    #[test]
    fn test_coordinator_record_failure() {
        let cb = CircuitBreaker::new(CircuitBreakerConfig::default());
        let coord = ResilienceCoordinator::new(cb);

        coord.record_failure();
        assert_eq!(coord.circuit_breaker.failures(), 1);
    }

    #[test]
    fn test_coordinator_fallback_available() {
        let cb = CircuitBreaker::new(CircuitBreakerConfig::default());
        let mut coord = ResilienceCoordinator::new(cb);

        let err = MinervaError::GpuOutOfMemory("test".to_string());
        let decision = coord.decide(&err);

        assert!(decision.should_fallback);
    }
}
