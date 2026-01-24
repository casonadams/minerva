pub use super::circuit_breaker_config::CircuitBreakerConfig;

/// Circuit Breaker Pattern
///
/// Protects against cascading failures by:
/// - Closed: Normal operation, requests pass through
/// - Open: Too many failures, requests fail immediately
/// - Half-Open: Testing if service recovered, single request allowed
///
/// Transitions:
/// - Closed → Open: Failure threshold exceeded
/// - Open → Half-Open: Timeout elapsed
/// - Half-Open → Closed: Test request succeeds
/// - Half-Open → Open: Test request fails
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// State of the circuit breaker
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CircuitState {
    /// Normal operation, requests allowed
    Closed,
    /// Failing, reject requests immediately
    Open,
    /// Testing recovery, single request allowed
    HalfOpen,
}

/// Circuit breaker state machine
pub struct CircuitBreaker {
    state: Arc<CircuitBreakerState>,
}

struct CircuitBreakerState {
    // Current state: 0=Closed, 1=Open, 2=HalfOpen
    state: AtomicU32,
    // Consecutive failure count
    failures: AtomicU32,
    // Successful calls during half-open
    successes: AtomicU32,
    // Timestamp when opened (unix seconds)
    opened_at: AtomicU64,
    config: CircuitBreakerConfig,
}

impl CircuitBreaker {
    /// Create new circuit breaker with config
    pub fn new(config: CircuitBreakerConfig) -> Self {
        Self {
            state: Arc::new(CircuitBreakerState {
                state: AtomicU32::new(0), // Closed
                failures: AtomicU32::new(0),
                successes: AtomicU32::new(0),
                opened_at: AtomicU64::new(0),
                config,
            }),
        }
    }

    /// Get current state
    pub fn state(&self) -> CircuitState {
        match self.state.state.load(Ordering::SeqCst) {
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
                // Check if timeout elapsed
                if let Ok(elapsed) = self.time_since_open() {
                    if elapsed >= Duration::from_secs(self.state.config.timeout_secs) {
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
                let successes = self.state.successes.load(Ordering::SeqCst);
                successes < self.state.config.half_open_max_calls
            }
        }
    }

    /// Record successful operation
    pub fn record_success(&self) {
        match self.state() {
            CircuitState::Closed => {
                // Reset failures on success
                self.state.failures.store(0, Ordering::SeqCst);
            }
            CircuitState::HalfOpen => {
                // Track successes during half-open
                let successes = self.state.successes.fetch_add(1, Ordering::SeqCst) + 1;
                if successes >= self.state.config.half_open_max_calls {
                    // Recovered, close circuit
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
                let failures = self.state.failures.fetch_add(1, Ordering::SeqCst) + 1;
                if failures >= self.state.config.failure_threshold {
                    self.transition_to_open();
                }
            }
            CircuitState::HalfOpen => {
                // Failed during recovery attempt, open again
                self.transition_to_open();
            }
            _ => {}
        }
    }

    /// Get current failure count
    pub fn failures(&self) -> u32 {
        self.state.failures.load(Ordering::SeqCst)
    }

    /// Reset circuit breaker
    pub fn reset(&self) {
        self.state.failures.store(0, Ordering::SeqCst);
        self.state.successes.store(0, Ordering::SeqCst);
        self.state.opened_at.store(0, Ordering::SeqCst);
        self.state.state.store(0, Ordering::SeqCst);
    }

    // Private helpers
    fn transition_to_open(&self) {
        self.state.state.store(1, Ordering::SeqCst);
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        self.state.opened_at.store(now, Ordering::SeqCst);
    }

    fn transition_to_half_open(&self) {
        self.state.state.store(2, Ordering::SeqCst);
        self.state.successes.store(0, Ordering::SeqCst);
    }

    fn transition_to_closed(&self) {
        self.state.state.store(0, Ordering::SeqCst);
        self.state.failures.store(0, Ordering::SeqCst);
        self.state.successes.store(0, Ordering::SeqCst);
    }

    fn time_since_open(&self) -> Result<Duration, std::time::SystemTimeError> {
        let opened_at = self.state.opened_at.load(Ordering::SeqCst);
        let now = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();
        Ok(Duration::from_secs(now.saturating_sub(opened_at)))
    }
}

impl Clone for CircuitBreaker {
    fn clone(&self) -> Self {
        Self {
            state: Arc::clone(&self.state),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_circuit_breaker_creation() {
        let cb = CircuitBreaker::new(CircuitBreakerConfig::default());
        assert_eq!(cb.state(), CircuitState::Closed);
        assert_eq!(cb.failures(), 0);
    }

    #[test]
    fn test_circuit_breaker_allow_request_closed() {
        let cb = CircuitBreaker::new(CircuitBreakerConfig::default());
        assert!(cb.allow_request());
    }

    #[test]
    fn test_circuit_breaker_failure_threshold() {
        let cfg = CircuitBreakerConfig {
            failure_threshold: 3,
            ..Default::default()
        };
        let cb = CircuitBreaker::new(cfg);

        cb.record_failure();
        assert_eq!(cb.state(), CircuitState::Closed);
        assert_eq!(cb.failures(), 1);

        cb.record_failure();
        cb.record_failure();
        assert_eq!(cb.state(), CircuitState::Open);
    }

    #[test]
    fn test_circuit_breaker_open_rejects_requests() {
        let cfg = CircuitBreakerConfig {
            failure_threshold: 1,
            ..Default::default()
        };
        let cb = CircuitBreaker::new(cfg);

        cb.record_failure();
        assert_eq!(cb.state(), CircuitState::Open);
        assert!(!cb.allow_request());
    }

    #[test]
    fn test_circuit_breaker_success_resets() {
        let cfg = CircuitBreakerConfig {
            failure_threshold: 5,
            ..Default::default()
        };
        let cb = CircuitBreaker::new(cfg);

        cb.record_failure();
        cb.record_failure();
        assert_eq!(cb.failures(), 2);

        cb.record_success();
        assert_eq!(cb.failures(), 0);
    }

    #[test]
    fn test_circuit_breaker_half_open_transition() {
        let cfg = CircuitBreakerConfig {
            failure_threshold: 1,
            timeout_secs: 0, // Immediate transition
            half_open_max_calls: 1,
        };
        let cb = CircuitBreaker::new(cfg);

        // Open circuit
        cb.record_failure();
        assert_eq!(cb.state(), CircuitState::Open);

        // Should allow request (timeout passed)
        assert!(cb.allow_request());
        assert_eq!(cb.state(), CircuitState::HalfOpen);
    }

    #[test]
    fn test_circuit_breaker_half_open_recovery() {
        let cfg = CircuitBreakerConfig {
            failure_threshold: 1,
            timeout_secs: 0,
            half_open_max_calls: 1,
        };
        let cb = CircuitBreaker::new(cfg);

        cb.record_failure();
        assert!(cb.allow_request()); // Transition to half-open
        cb.record_success();
        assert_eq!(cb.state(), CircuitState::Closed);
    }

    #[test]
    fn test_circuit_breaker_half_open_failure() {
        let cfg = CircuitBreakerConfig {
            failure_threshold: 1,
            timeout_secs: 0,
            half_open_max_calls: 1,
        };
        let cb = CircuitBreaker::new(cfg);

        cb.record_failure();
        assert!(cb.allow_request()); // Transition to half-open
        cb.record_failure();
        assert_eq!(cb.state(), CircuitState::Open);
    }

    #[test]
    fn test_circuit_breaker_reset() {
        let cb = CircuitBreaker::new(CircuitBreakerConfig::default());

        cb.record_failure();
        cb.record_failure();
        assert_eq!(cb.failures(), 2);

        cb.reset();
        assert_eq!(cb.state(), CircuitState::Closed);
        assert_eq!(cb.failures(), 0);
    }

    #[test]
    fn test_circuit_breaker_cloneable() {
        let cb1 = CircuitBreaker::new(CircuitBreakerConfig::default());
        let cb2 = cb1.clone();

        cb1.record_failure();
        // Both should see the same state
        assert_eq!(cb2.failures(), 1);
    }
}
