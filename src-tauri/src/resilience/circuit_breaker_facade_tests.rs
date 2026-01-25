#[cfg(test)]
mod tests {
    use crate::resilience::circuit_breaker_config::CircuitBreakerConfig;
    use crate::resilience::circuit_breaker_facade::CircuitBreaker;
    use crate::resilience::circuit_breaker_transitions::CircuitState;

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
            timeout_secs: 0,
            half_open_max_calls: 1,
        };
        let cb = CircuitBreaker::new(cfg);

        cb.record_failure();
        assert_eq!(cb.state(), CircuitState::Open);

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
        assert!(cb.allow_request());
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
        assert!(cb.allow_request());
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
        assert_eq!(cb2.failures(), 1);
    }
}
