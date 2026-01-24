use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// Helper for managing circuit breaker state transitions
pub struct StateTransitionHelper {
    state: AtomicU32,
    failures: AtomicU32,
    successes: AtomicU32,
    opened_at: AtomicU64,
}

impl Default for StateTransitionHelper {
    fn default() -> Self {
        Self::new()
    }
}

impl StateTransitionHelper {
    /// Create new transition helper
    pub fn new() -> Self {
        Self {
            state: AtomicU32::new(0), // Closed
            failures: AtomicU32::new(0),
            successes: AtomicU32::new(0),
            opened_at: AtomicU64::new(0),
        }
    }

    /// Transition to open state
    pub fn transition_to_open(&self) {
        self.state.store(1, Ordering::SeqCst);
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        self.opened_at.store(now, Ordering::SeqCst);
    }

    /// Transition to half-open state
    pub fn transition_to_half_open(&self) {
        self.state.store(2, Ordering::SeqCst);
        self.successes.store(0, Ordering::SeqCst);
    }

    /// Transition to closed state
    pub fn transition_to_closed(&self) {
        self.state.store(0, Ordering::SeqCst);
        self.failures.store(0, Ordering::SeqCst);
        self.successes.store(0, Ordering::SeqCst);
    }

    /// Get time since circuit was opened
    pub fn time_since_open(&self) -> Result<Duration, std::time::SystemTimeError> {
        let opened_at = self.opened_at.load(Ordering::SeqCst);
        let now = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();
        Ok(Duration::from_secs(now.saturating_sub(opened_at)))
    }

    /// Get current raw state value
    pub fn get_state(&self) -> u32 {
        self.state.load(Ordering::SeqCst)
    }

    /// Increment failures
    pub fn increment_failures(&self) -> u32 {
        self.failures.fetch_add(1, Ordering::SeqCst) + 1
    }

    /// Reset failures to zero
    pub fn reset_failures(&self) {
        self.failures.store(0, Ordering::SeqCst);
    }

    /// Increment successes
    pub fn increment_successes(&self) -> u32 {
        self.successes.fetch_add(1, Ordering::SeqCst) + 1
    }

    /// Get current failure count
    pub fn get_failures(&self) -> u32 {
        self.failures.load(Ordering::SeqCst)
    }

    /// Get current success count
    pub fn get_successes(&self) -> u32 {
        self.successes.load(Ordering::SeqCst)
    }

    /// Reset all state
    pub fn reset_all(&self) {
        self.failures.store(0, Ordering::SeqCst);
        self.successes.store(0, Ordering::SeqCst);
        self.opened_at.store(0, Ordering::SeqCst);
        self.state.store(0, Ordering::SeqCst);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_initial_state() {
        let helper = StateTransitionHelper::new();
        assert_eq!(helper.get_state(), 0);
    }

    #[test]
    fn test_transition_to_open() {
        let helper = StateTransitionHelper::new();
        helper.transition_to_open();
        assert_eq!(helper.get_state(), 1);
    }

    #[test]
    fn test_transition_to_half_open() {
        let helper = StateTransitionHelper::new();
        helper.transition_to_half_open();
        assert_eq!(helper.get_state(), 2);
    }

    #[test]
    fn test_transition_to_closed() {
        let helper = StateTransitionHelper::new();
        helper.transition_to_open();
        helper.transition_to_closed();
        assert_eq!(helper.get_state(), 0);
    }

    #[test]
    fn test_increment_failures() {
        let helper = StateTransitionHelper::new();
        assert_eq!(helper.increment_failures(), 1);
        assert_eq!(helper.increment_failures(), 2);
    }

    #[test]
    fn test_reset_all() {
        let helper = StateTransitionHelper::new();
        helper.increment_failures();
        helper.transition_to_open();
        helper.reset_all();
        assert_eq!(helper.get_state(), 0);
        assert_eq!(helper.get_failures(), 0);
    }
}
