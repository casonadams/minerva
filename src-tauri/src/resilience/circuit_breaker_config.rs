/// Configuration for circuit breaker
#[derive(Debug, Clone, Copy)]
pub struct CircuitBreakerConfig {
    /// Number of consecutive failures to open circuit
    pub failure_threshold: u32,
    /// Time to wait before attempting half-open
    pub timeout_secs: u64,
    /// Maximum requests during half-open state
    pub half_open_max_calls: u32,
}

impl Default for CircuitBreakerConfig {
    fn default() -> Self {
        Self {
            failure_threshold: 5,
            timeout_secs: 30,
            half_open_max_calls: 1,
        }
    }
}

impl CircuitBreakerConfig {
    /// Create config for GPU operations (strict)
    pub fn for_gpu() -> Self {
        Self {
            failure_threshold: 3,
            timeout_secs: 60,
            half_open_max_calls: 1,
        }
    }

    /// Create config for network operations (lenient)
    pub fn for_network() -> Self {
        Self {
            failure_threshold: 5,
            timeout_secs: 10,
            half_open_max_calls: 2,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let cfg = CircuitBreakerConfig::default();
        assert_eq!(cfg.failure_threshold, 5);
        assert_eq!(cfg.timeout_secs, 30);
        assert_eq!(cfg.half_open_max_calls, 1);
    }

    #[test]
    fn test_gpu_config() {
        let cfg = CircuitBreakerConfig::for_gpu();
        assert_eq!(cfg.failure_threshold, 3);
        assert_eq!(cfg.timeout_secs, 60);
    }

    #[test]
    fn test_network_config() {
        let cfg = CircuitBreakerConfig::for_network();
        assert_eq!(cfg.timeout_secs, 10);
        assert_eq!(cfg.half_open_max_calls, 2);
    }
}
