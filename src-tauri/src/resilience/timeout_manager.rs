use super::timeout_context::TimeoutContext;
use super::timeout_stats::TimeoutStats;
use std::time::Duration;

/// Timeout manager for tracking multiple operations
pub struct TimeoutManager {
    max_total: Duration,
    op_timeout: Duration,
    pub contexts: Vec<TimeoutContext>,
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

#[cfg(test)]
mod tests {
    use super::*;

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
    fn test_timeout_manager_reset() {
        let mut mgr = TimeoutManager::new(Duration::from_secs(10), Duration::from_secs(5));
        mgr.create_context();
        mgr.create_context();
        assert_eq!(mgr.contexts.len(), 2);

        mgr.reset();
        assert_eq!(mgr.contexts.len(), 0);
    }
}
