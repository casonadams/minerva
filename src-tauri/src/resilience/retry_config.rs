/// Retry configuration
#[derive(Debug, Clone, Copy)]
pub struct RetryConfig {
    /// Maximum number of retry attempts (0 = no retries, 1 = fail-fast)
    pub max_attempts: u32,
    /// Base delay in milliseconds for first retry
    pub base_delay_ms: u64,
    /// Maximum delay in milliseconds (caps backoff growth)
    pub max_delay_ms: u64,
    /// Whether to apply jitter to delays
    pub use_jitter: bool,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_attempts: 3,
            base_delay_ms: 100,
            max_delay_ms: 10_000,
            use_jitter: true,
        }
    }
}

impl RetryConfig {
    /// Create config with custom max attempts
    pub fn with_attempts(max_attempts: u32) -> Self {
        Self {
            max_attempts,
            ..Default::default()
        }
    }

    /// Create config for aggressive retries (more attempts, longer delays)
    pub fn aggressive() -> Self {
        Self {
            max_attempts: 5,
            base_delay_ms: 100,
            max_delay_ms: 30_000,
            use_jitter: true,
        }
    }

    /// Create config for conservative retries (fewer attempts, shorter delays)
    pub fn conservative() -> Self {
        Self {
            max_attempts: 2,
            base_delay_ms: 50,
            max_delay_ms: 1_000,
            use_jitter: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_retry_config_default() {
        let cfg = RetryConfig::default();
        assert_eq!(cfg.max_attempts, 3);
        assert_eq!(cfg.base_delay_ms, 100);
        assert_eq!(cfg.max_delay_ms, 10_000);
        assert!(cfg.use_jitter);
    }

    #[test]
    fn test_retry_config_with_attempts() {
        let cfg = RetryConfig::with_attempts(5);
        assert_eq!(cfg.max_attempts, 5);
        assert_eq!(cfg.base_delay_ms, 100);
    }

    #[test]
    fn test_retry_config_aggressive() {
        let cfg = RetryConfig::aggressive();
        assert_eq!(cfg.max_attempts, 5);
        assert_eq!(cfg.base_delay_ms, 100);
        assert_eq!(cfg.max_delay_ms, 30_000);
    }

    #[test]
    fn test_retry_config_conservative() {
        let cfg = RetryConfig::conservative();
        assert_eq!(cfg.max_attempts, 2);
        assert_eq!(cfg.base_delay_ms, 50);
    }
}
