use super::circuit_breaker_config::CircuitBreakerConfig;
use super::circuit_breaker_request_handler::CircuitBreakerRequestHandler;
use super::circuit_breaker_state_recorder::CircuitBreakerStateRecorder;
use super::circuit_state_transitions::StateTransitionHelper;

/// Circuit breaker states
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CircuitState {
    /// Normal operation, requests allowed
    Closed,
    /// Failing, reject requests immediately
    Open,
    /// Testing recovery, single request allowed
    HalfOpen,
}

/// State machine for circuit breaker transitions
pub struct CircuitBreakerStateMachine {
    transitions: StateTransitionHelper,
    config: CircuitBreakerConfig,
}

impl CircuitBreakerStateMachine {
    /// Create new state machine
    pub fn new(config: CircuitBreakerConfig) -> Self {
        Self {
            transitions: StateTransitionHelper::new(),
            config,
        }
    }

    /// Get current state
    pub fn state(&self) -> CircuitState {
        match self.transitions.get_state() {
            0 => CircuitState::Closed,
            1 => CircuitState::Open,
            _ => CircuitState::HalfOpen,
        }
    }

    /// Check if operation is allowed
    pub fn allow_request(&self) -> bool {
        CircuitBreakerRequestHandler::should_allow_request(
            self.state(),
            &self.transitions,
            &self.config,
        )
    }

    /// Record successful operation
    pub fn record_success(&self) {
        CircuitBreakerStateRecorder::record_success(self.state(), &self.transitions, &self.config);
    }

    /// Record failed operation
    pub fn record_failure(&self) {
        CircuitBreakerStateRecorder::record_failure(self.state(), &self.transitions, &self.config);
    }

    /// Get current failure count
    pub fn failures(&self) -> u32 {
        self.transitions.get_failures()
    }

    /// Reset state machine
    pub fn reset(&self) {
        self.transitions.reset_all();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_initial_state_closed() {
        let sm = CircuitBreakerStateMachine::new(CircuitBreakerConfig::default());
        assert_eq!(sm.state(), CircuitState::Closed);
    }

    #[test]
    fn test_open_on_failure_threshold() {
        let cfg = CircuitBreakerConfig {
            failure_threshold: 2,
            ..Default::default()
        };
        let sm = CircuitBreakerStateMachine::new(cfg);
        sm.record_failure();
        assert_eq!(sm.state(), CircuitState::Closed);
        sm.record_failure();
        assert_eq!(sm.state(), CircuitState::Open);
    }

    #[test]
    fn test_half_open_transition() {
        let cfg = CircuitBreakerConfig {
            failure_threshold: 1,
            timeout_secs: 0,
            half_open_max_calls: 1,
        };
        let sm = CircuitBreakerStateMachine::new(cfg);
        sm.record_failure();
        assert!(sm.allow_request());
        assert_eq!(sm.state(), CircuitState::HalfOpen);
    }

    #[test]
    fn test_reset() {
        let cfg = CircuitBreakerConfig::default();
        let sm = CircuitBreakerStateMachine::new(cfg);
        sm.record_failure();
        sm.record_failure();
        sm.reset();
        assert_eq!(sm.state(), CircuitState::Closed);
        assert_eq!(sm.failures(), 0);
    }
}
