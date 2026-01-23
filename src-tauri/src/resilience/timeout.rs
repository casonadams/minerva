/// Timeout Management and Deadline Propagation
///
/// Handles operation timeouts with:
/// - Per-operation timeouts
/// - Total request deadlines
/// - Timeout recovery strategies
/// - Deadline tracking across async boundaries
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

/// Timeout manager for tracking multiple operations
pub struct TimeoutManager {
    max_total: Duration,
    op_timeout: Duration,
    contexts: Vec<TimeoutContext>,
}

impl TimeoutManager {
    /// Create new manager
    pub fn new(max_total: Duration, op_timeout: Duration) -> Self {
        Self {
            max_total,
            op_timeout,
            contexts: Vec::new(),
        }
    }

    /// Create new context for an operation
    pub fn create_context(&mut self) -> TimeoutContext {
        let ctx = TimeoutContext::new(self.max_total, self.op_timeout);
        self.contexts.push(ctx.clone());
        ctx
    }

    /// Check if any operation has exceeded its deadline
    pub fn any_exceeded(&self) -> bool {
        self.contexts.iter().any(|c| c.is_deadline_exceeded())
    }

    /// Get count of operations that have timed out
    pub fn timed_out_count(&self) -> usize {
        self.contexts
            .iter()
            .filter(|c| c.is_deadline_exceeded())
            .count()
    }

    /// Reset manager
    pub fn reset(&mut self) {
        self.contexts.clear();
    }

    /// Get statistics
    pub fn stats(&self) -> TimeoutStats {
        let total = self.contexts.len();
        let timed_out = self.timed_out_count();
        let avg_elapsed = if total > 0 {
            let sum: Duration = self.contexts.iter().map(|c| c.elapsed()).sum();
            sum / total as u32
        } else {
            Duration::ZERO
        };

        TimeoutStats {
            total_contexts: total,
            timed_out_count: timed_out,
            avg_elapsed,
            max_total: self.max_total,
        }
    }
}

/// Timeout statistics
#[derive(Debug, Clone)]
pub struct TimeoutStats {
    /// Total number of contexts created
    pub total_contexts: usize,
    /// Number that timed out
    pub timed_out_count: usize,
    /// Average elapsed time per context
    pub avg_elapsed: Duration,
    /// Maximum total deadline
    pub max_total: Duration,
}

impl TimeoutStats {
    /// Timeout rate as percentage
    pub fn timeout_rate(&self) -> f64 {
        if self.total_contexts > 0 {
            (self.timed_out_count as f64 / self.total_contexts as f64) * 100.0
        } else {
            0.0
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
        assert!(pct <= 5.0); // Should be less than 1% immediately
    }

    #[test]
    fn test_timeout_context_capped_by_remaining() {
        let ctx = TimeoutContext::new(Duration::from_millis(100), Duration::from_secs(10));
        thread::sleep(Duration::from_millis(50));
        let op_timeout = ctx.operation_timeout();
        assert!(op_timeout < Duration::from_secs(10));
        assert!(op_timeout <= Duration::from_millis(50));
    }

    #[test]
    fn test_timeout_manager_creation() {
        let mgr = TimeoutManager::new(Duration::from_secs(10), Duration::from_secs(5));
        assert!(!mgr.any_exceeded());
        assert_eq!(mgr.timed_out_count(), 0);
    }

    #[test]
    fn test_timeout_manager_create_context() {
        let mut mgr = TimeoutManager::new(Duration::from_secs(10), Duration::from_secs(5));
        let ctx = mgr.create_context();
        assert!(!ctx.is_deadline_exceeded());
    }

    #[test]
    fn test_timeout_manager_multiple_contexts() {
        let mut mgr = TimeoutManager::new(Duration::from_secs(10), Duration::from_secs(5));
        mgr.create_context();
        mgr.create_context();
        mgr.create_context();
        assert_eq!(mgr.contexts.len(), 3);
    }

    #[test]
    fn test_timeout_manager_stats() {
        let mut mgr = TimeoutManager::new(Duration::from_secs(10), Duration::from_secs(5));
        mgr.create_context();
        mgr.create_context();

        let stats = mgr.stats();
        assert_eq!(stats.total_contexts, 2);
        assert_eq!(stats.timed_out_count, 0);
    }

    #[test]
    fn test_timeout_stats_timeout_rate() {
        let stats = TimeoutStats {
            total_contexts: 10,
            timed_out_count: 2,
            avg_elapsed: Duration::from_millis(100),
            max_total: Duration::from_secs(1),
        };
        assert_eq!(stats.timeout_rate(), 20.0);
    }

    #[test]
    fn test_timeout_stats_zero_rate() {
        let stats = TimeoutStats {
            total_contexts: 0,
            timed_out_count: 0,
            avg_elapsed: Duration::ZERO,
            max_total: Duration::from_secs(1),
        };
        assert_eq!(stats.timeout_rate(), 0.0);
    }

    #[test]
    fn test_timeout_manager_reset() {
        let mut mgr = TimeoutManager::new(Duration::from_secs(10), Duration::from_secs(5));
        mgr.create_context();
        mgr.create_context();
        assert_eq!(mgr.contexts.len(), 2);

        mgr.reset();
        assert_eq!(mgr.contexts.len(), 0);
    }
}
