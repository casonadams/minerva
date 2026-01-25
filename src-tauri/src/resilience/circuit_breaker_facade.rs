use super::circuit_breaker_config::CircuitBreakerConfig;
use super::circuit_breaker_transitions::{CircuitBreakerStateMachine, CircuitState};
use std::sync::Arc;

/// Circuit breaker state machine wrapper and facade
pub struct CircuitBreaker {
    state: Arc<CircuitBreakerStateMachine>,
}

impl CircuitBreaker {
    /// Create new circuit breaker with config
    pub fn new(config: CircuitBreakerConfig) -> Self {
        Self {
            state: Arc::new(CircuitBreakerStateMachine::new(config)),
        }
    }

    /// Get current state
    pub fn state(&self) -> CircuitState {
        self.state.state()
    }

    /// Check if operation is allowed
    pub fn allow_request(&self) -> bool {
        self.state.allow_request()
    }

    /// Record successful operation
    pub fn record_success(&self) {
        self.state.record_success();
    }

    /// Record failed operation
    pub fn record_failure(&self) {
        self.state.record_failure();
    }

    /// Get current failure count
    pub fn failures(&self) -> u32 {
        self.state.failures()
    }

    /// Reset circuit breaker
    pub fn reset(&self) {
        self.state.reset();
    }
}

impl Clone for CircuitBreaker {
    fn clone(&self) -> Self {
        Self {
            state: Arc::clone(&self.state),
        }
    }
}
