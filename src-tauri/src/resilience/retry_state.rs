use super::retry_config::RetryConfig;
use std::time::Duration;

/// Retry state tracker
#[derive(Clone)]
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
    fn test_retry_state_creation() {
        let state = RetryState::new(RetryConfig::default());
        assert_eq!(state.attempt(), 0);
        assert_eq!(state.remaining(), 3);
        assert!(state.can_retry());
    }

    #[test]
    fn test_retry_state_progression() {
        let mut state = RetryState::new(RetryConfig::with_attempts(3));

        assert!(state.can_retry());
        let delay1 = state.next_delay();
        assert_eq!(state.attempt(), 1);
        assert!(delay1 >= Duration::from_millis(0));

        assert!(state.can_retry());
        let _delay2 = state.next_delay();
        assert_eq!(state.attempt(), 2);

        assert!(state.can_retry());
        let _delay3 = state.next_delay();
        assert_eq!(state.attempt(), 3);
        assert!(!state.can_retry());
    }

    #[test]
    fn test_retry_state_remaining() {
        let mut state = RetryState::new(RetryConfig::with_attempts(5));

        assert_eq!(state.remaining(), 5);
        state.next_delay();
        assert_eq!(state.remaining(), 4);
        state.next_delay();
        assert_eq!(state.remaining(), 3);
    }

    #[test]
    fn test_retry_delay_increase() {
        let cfg = RetryConfig {
            base_delay_ms: 100,
            use_jitter: false,
            ..Default::default()
        };

        let mut state = RetryState::new(cfg);
        let delay1 = state.next_delay();

        let delay2 = state.next_delay();

        // Delay should increase with exponential backoff
        assert!(delay2.as_millis() >= delay1.as_millis());
    }

    #[test]
    fn test_retry_delay_max_cap() {
        let cfg = RetryConfig {
            max_attempts: 10,
            base_delay_ms: 100,
            max_delay_ms: 5_000,
            use_jitter: false,
        };

        let mut state = RetryState::new(cfg);

        // Skip to attempt 7 (delay would be 100 * 2^7 = 12800ms without cap)
        for _ in 0..7 {
            state.next_delay();
        }

        let delayed = state.next_delay();
        assert!(delayed.as_millis() <= 5_000);
    }
}
