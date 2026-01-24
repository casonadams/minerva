use super::circuit_breaker_config::CircuitBreakerConfig;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

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
    state: AtomicU32,
    failures: AtomicU32,
    successes: AtomicU32,
    opened_at: AtomicU64,
    config: CircuitBreakerConfig,
}

impl CircuitBreakerStateMachine {
    /// Create new state machine
    pub fn new(config: CircuitBreakerConfig) -> Self {
        Self {
            state: AtomicU32::new(0), // Closed
            failures: AtomicU32::new(0),
            successes: AtomicU32::new(0),
            opened_at: AtomicU64::new(0),
            config,
        }
    }

    /// Get current state
    pub fn state(&self) -> CircuitState {
        match self.state.load(Ordering::SeqCst) {
            0 => CircuitState::Closed,
            1 => CircuitState::Open,
            _ => CircuitState::HalfOpen,
        }
    }

    /// Check if operation is allowed
    pub fn allow_request(&self) -> bool {
        match self.state() {
            CircuitState::Closed => true,
            CircuitState::Open => {
                if let Ok(elapsed) = self.time_since_open() {
                    if elapsed >= Duration::from_secs(self.config.timeout_secs) {
                        self.transition_to_half_open();
                        true
                    } else {
                        false
                    }
                } else {
                    false
                }
            }
            CircuitState::HalfOpen => {
                let successes = self.successes.load(Ordering::SeqCst);
                successes < self.config.half_open_max_calls
            }
        }
    }

    /// Record successful operation
    pub fn record_success(&self) {
        match self.state() {
            CircuitState::Closed => {
                self.failures.store(0, Ordering::SeqCst);
            }
            CircuitState::HalfOpen => {
                let successes = self.successes.fetch_add(1, Ordering::SeqCst) + 1;
                if successes >= self.config.half_open_max_calls {
                    self.transition_to_closed();
                }
            }
            _ => {}
        }
    }

    /// Record failed operation
    pub fn record_failure(&self) {
        match self.state() {
            CircuitState::Closed => {
                let failures = self.failures.fetch_add(1, Ordering::SeqCst) + 1;
                if failures >= self.config.failure_threshold {
                    self.transition_to_open();
                }
            }
            CircuitState::HalfOpen => {
                self.transition_to_open();
            }
            _ => {}
        }
    }

    /// Get current failure count
    pub fn failures(&self) -> u32 {
        self.failures.load(Ordering::SeqCst)
    }

    /// Reset state machine
    pub fn reset(&self) {
        self.failures.store(0, Ordering::SeqCst);
        self.successes.store(0, Ordering::SeqCst);
        self.opened_at.store(0, Ordering::SeqCst);
        self.state.store(0, Ordering::SeqCst);
    }

    fn transition_to_open(&self) {
        self.state.store(1, Ordering::SeqCst);
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        self.opened_at.store(now, Ordering::SeqCst);
    }

    fn transition_to_half_open(&self) {
        self.state.store(2, Ordering::SeqCst);
        self.successes.store(0, Ordering::SeqCst);
    }

    fn transition_to_closed(&self) {
        self.state.store(0, Ordering::SeqCst);
        self.failures.store(0, Ordering::SeqCst);
        self.successes.store(0, Ordering::SeqCst);
    }

    fn time_since_open(&self) -> Result<Duration, std::time::SystemTimeError> {
        let opened_at = self.opened_at.load(Ordering::SeqCst);
        let now = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();
        Ok(Duration::from_secs(now.saturating_sub(opened_at)))
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
}
