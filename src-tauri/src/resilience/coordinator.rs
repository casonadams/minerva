use super::{
    circuit_breaker::CircuitBreaker, coordinator_decision::CoordinatorDecision,
    resilience_decision::ResilienceDecision, retry::RetryState, timeout::TimeoutContext,
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
        use crate::resilience::coordinator_decision::DecisionContext;
        CoordinatorDecision::decide(DecisionContext {
            circuit_breaker: &self.circuit_breaker,
            retry_state: &mut self.retry_state,
            timeout_context: &self.timeout_context,
            error,
        })
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
}
