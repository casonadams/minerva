use std::time::{Duration, Instant};

/// Timeout context for an operation
#[derive(Debug, Clone)]
pub struct TimeoutContext {
    /// When this context was created
    start_time: Instant,
    /// Total time allowed for entire operation
    total_deadline: Duration,
    /// Time allowed for current operation phase
    operation_timeout: Duration,
}

impl TimeoutContext {
    /// Create new timeout context
    pub fn new(total_deadline: Duration, operation_timeout: Duration) -> Self {
        Self {
            start_time: Instant::now(),
            total_deadline,
            operation_timeout,
        }
    }

    /// Get time elapsed since start
    pub fn elapsed(&self) -> Duration {
        self.start_time.elapsed()
    }

    /// Get remaining time until total deadline
    pub fn remaining(&self) -> Duration {
        self.total_deadline.saturating_sub(self.elapsed())
    }

    /// Has total deadline been exceeded?
    pub fn is_deadline_exceeded(&self) -> bool {
        self.elapsed() >= self.total_deadline
    }

    /// Get operation timeout (capped by remaining time)
    pub fn operation_timeout(&self) -> Duration {
        let remaining = self.remaining();
        self.operation_timeout.min(remaining)
    }

    /// Has operation timeout been exceeded?
    pub fn is_operation_timeout(&self) -> bool {
        self.elapsed() >= self.operation_timeout
    }

    /// Percentage of total deadline consumed
    pub fn deadline_percent(&self) -> f64 {
        let elapsed_ms = self.elapsed().as_millis() as f64;
        let total_ms = self.total_deadline.as_millis() as f64;
        if total_ms > 0.0 {
            (elapsed_ms / total_ms * 100.0).min(100.0)
        } else {
            100.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn test_timeout_context_creation() {
        let ctx = TimeoutContext::new(Duration::from_secs(10), Duration::from_secs(5));
        assert!(ctx.elapsed() < Duration::from_secs(1));
        assert!(ctx.remaining() > Duration::from_secs(9));
    }

    #[test]
    fn test_timeout_context_elapsed() {
        let ctx = TimeoutContext::new(Duration::from_secs(10), Duration::from_secs(5));
        thread::sleep(Duration::from_millis(100));
        assert!(ctx.elapsed() >= Duration::from_millis(100));
    }

    #[test]
    fn test_timeout_context_remaining() {
        let ctx = TimeoutContext::new(Duration::from_secs(1), Duration::from_secs(1));
        let remaining = ctx.remaining();
        assert!(remaining <= Duration::from_secs(1));
        assert!(remaining > Duration::from_millis(900));
    }

    #[test]
    fn test_timeout_context_operation_timeout() {
        let ctx = TimeoutContext::new(Duration::from_secs(10), Duration::from_secs(2));
        let op_timeout = ctx.operation_timeout();
        assert!(op_timeout <= Duration::from_secs(2));
        assert!(op_timeout > Duration::from_millis(1900));
    }

    #[test]
    fn test_timeout_context_deadline_exceeded() {
        let ctx = TimeoutContext::new(Duration::from_millis(1), Duration::from_secs(10));
        thread::sleep(Duration::from_millis(10));
        assert!(ctx.is_deadline_exceeded());
    }

    #[test]
    fn test_timeout_context_operation_timeout_exceeded() {
        let ctx = TimeoutContext::new(Duration::from_secs(10), Duration::from_millis(1));
        thread::sleep(Duration::from_millis(10));
        assert!(ctx.is_operation_timeout());
    }

    #[test]
    fn test_timeout_context_deadline_percent() {
        let ctx = TimeoutContext::new(Duration::from_secs(10), Duration::from_secs(5));
        let pct = ctx.deadline_percent();
        assert!(pct >= 0.0);
        assert!(pct <= 5.0);
    }

    #[test]
    fn test_timeout_context_capped_by_remaining() {
        let ctx = TimeoutContext::new(Duration::from_millis(100), Duration::from_secs(10));
        thread::sleep(Duration::from_millis(50));
        let op_timeout = ctx.operation_timeout();
        assert!(op_timeout < Duration::from_secs(10));
        assert!(op_timeout <= Duration::from_millis(50));
    }
}
