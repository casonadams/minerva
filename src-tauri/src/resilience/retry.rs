/// Retry logic with exponential backoff and jitter
///
/// Implements the retry pattern with:
/// - Configurable max attempts
/// - Exponential backoff (2^n * base_ms)
/// - Jitter to prevent thundering herd
/// - Full jitter algorithm (random between 0 and backoff)
use std::time::Duration;

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

/// Retry state tracker
pub struct RetryState {
    attempt: u32,
    config: RetryConfig,
}

impl RetryState {
    /// Create new retry state with config
    pub fn new(config: RetryConfig) -> Self {
        Self { attempt: 0, config }
    }

    /// Get current attempt number (0-based)
    pub fn attempt(&self) -> u32 {
        self.attempt
    }

    /// Get remaining attempts
    pub fn remaining(&self) -> u32 {
        self.config.max_attempts.saturating_sub(self.attempt)
    }

    /// Can attempt again?
    pub fn can_retry(&self) -> bool {
        self.attempt < self.config.max_attempts
    }

    /// Move to next attempt, return delay to wait
    pub fn next_delay(&mut self) -> Duration {
        let delay = Self::calculate_delay(self.attempt, &self.config);
        self.attempt += 1;
        delay
    }

    /// Calculate backoff delay for given attempt
    fn calculate_delay(attempt: u32, config: &RetryConfig) -> Duration {
        // Exponential backoff: base * 2^attempt
        let exponential_ms = config.base_delay_ms.saturating_mul(
            2u64.checked_pow(attempt)
                .unwrap_or(u64::MAX / config.base_delay_ms),
        );

        // Cap at max_delay_ms
        let capped_ms = exponential_ms.min(config.max_delay_ms);

        // Apply full jitter if enabled: random(0, capped_ms)
        let final_ms = if config.use_jitter {
            (capped_ms as f64 * rand::random::<f64>()) as u64
        } else {
            capped_ms
        };

        Duration::from_millis(final_ms)
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
    fn test_retry_config_aggressive() {
        let cfg = RetryConfig::aggressive();
        assert_eq!(cfg.max_attempts, 5);
        assert!(cfg.max_delay_ms > 10_000);
    }

    #[test]
    fn test_retry_config_conservative() {
        let cfg = RetryConfig::conservative();
        assert_eq!(cfg.max_attempts, 2);
        assert!(cfg.max_delay_ms < 10_000);
    }

    #[test]
    fn test_retry_state_creation() {
        let state = RetryState::new(RetryConfig::default());
        assert_eq!(state.attempt(), 0);
        assert_eq!(state.remaining(), 3);
        assert!(state.can_retry());
    }

    #[test]
    fn test_retry_state_progress() {
        let mut state = RetryState::new(RetryConfig::with_attempts(3));
        assert_eq!(state.attempt(), 0);
        assert!(state.can_retry());

        state.next_delay();
        assert_eq!(state.attempt(), 1);
        assert_eq!(state.remaining(), 2);
        assert!(state.can_retry());

        state.next_delay();
        state.next_delay();
        assert_eq!(state.attempt(), 3);
        assert!(!state.can_retry());
    }

    #[test]
    fn test_retry_exponential_backoff() {
        // Without jitter, backoff should be predictable
        let cfg = RetryConfig {
            max_attempts: 5,
            base_delay_ms: 100,
            max_delay_ms: 100_000,
            use_jitter: false,
        };

        let delay0 = RetryState::calculate_delay(0, &cfg);
        let delay1 = RetryState::calculate_delay(1, &cfg);
        let delay2 = RetryState::calculate_delay(2, &cfg);

        // Each should roughly double (exactly without jitter)
        assert_eq!(delay0.as_millis(), 100);
        assert_eq!(delay1.as_millis(), 200);
        assert_eq!(delay2.as_millis(), 400);
    }

    #[test]
    fn test_retry_backoff_capped() {
        let cfg = RetryConfig {
            max_attempts: 10,
            base_delay_ms: 100,
            max_delay_ms: 1000,
            use_jitter: false,
        };

        // Eventually should cap at max_delay_ms
        let delay5 = RetryState::calculate_delay(5, &cfg);
        let delay10 = RetryState::calculate_delay(10, &cfg);

        assert!(delay5.as_millis() <= 1000);
        assert!(delay10.as_millis() <= 1000);
    }

    #[test]
    fn test_retry_jitter_within_bounds() {
        let cfg = RetryConfig {
            max_attempts: 5,
            base_delay_ms: 100,
            max_delay_ms: 1000,
            use_jitter: true,
        };

        // With jitter, delays should be within bounds but vary
        for _ in 0..10 {
            let delay = RetryState::calculate_delay(2, &cfg);
            assert!(delay.as_millis() <= 400); // max is 2^2 * 100
        }
    }

    #[test]
    fn test_retry_with_zero_attempts() {
        let state = RetryState::new(RetryConfig::with_attempts(0));
        assert!(!state.can_retry());
        assert_eq!(state.remaining(), 0);
    }

    #[test]
    fn test_retry_saturation_on_overflow() {
        let cfg = RetryConfig {
            max_attempts: 100,
            base_delay_ms: u64::MAX / 2,
            max_delay_ms: u64::MAX,
            use_jitter: false,
        };

        // Should not panic on overflow, should cap at max_delay_ms
        let delay = RetryState::calculate_delay(50, &cfg);
        assert!(delay.as_millis() <= u64::MAX as u128);
    }
}
