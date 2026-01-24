use std::time::Duration;

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
}
