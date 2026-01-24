use super::metrics::MetricsSnapshot;
use super::metrics_recorder::MetricsRecorder;
use super::metrics_snapshot_builder::{SnapshotBuilder, SnapshotParams};
use std::sync::Arc;
use std::time::Duration;

/// Metrics collector for request tracking
pub struct MetricsCollector {
    recorder: Arc<MetricsRecorder>,
    start_time: std::time::Instant,
}

impl MetricsCollector {
    /// Create new metrics collector
    pub fn new() -> Self {
        Self {
            recorder: Arc::new(MetricsRecorder::new()),
            start_time: std::time::Instant::now(),
        }
    }

    /// Record a successful request with response time
    pub fn record_success(&self, response_time: Duration) {
        self.recorder.record_success(response_time);
    }

    /// Record a failed request with response time
    pub fn record_failure(&self, response_time: Duration) {
        self.recorder.record_failure(response_time);
    }

    /// Record cache hit
    pub fn record_cache_hit(&self) {
        self.recorder.record_cache_hit();
    }

    /// Record cache miss
    pub fn record_cache_miss(&self) {
        self.recorder.record_cache_miss();
    }

    /// Get current metrics snapshot
    pub fn snapshot(&self) -> MetricsSnapshot {
        let total = self.recorder.total_requests();
        let success = self.recorder.successful_requests();
        let failed = self.recorder.failed_requests();
        let hits = self.recorder.cache_hits();
        let misses = self.recorder.cache_misses();

        let times = self.recorder.response_times();
        let uptime_secs = self.start_time.elapsed().as_secs();

        SnapshotBuilder::build(SnapshotParams {
            total,
            success,
            failed,
            hits,
            misses,
            times,
            uptime_secs,
        })
    }

    /// Reset all metrics
    pub fn reset(&self) {
        self.recorder.reset();
    }
}

impl Clone for MetricsCollector {
    fn clone(&self) -> Self {
        Self {
            recorder: Arc::clone(&self.recorder),
            start_time: self.start_time,
        }
    }
}

impl Default for MetricsCollector {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_collector_creation() {
        let c = MetricsCollector::new();
        let s = c.snapshot();
        assert_eq!(s.total_requests, 0);
    }

    #[test]
    fn test_record_success() {
        let c = MetricsCollector::new();
        c.record_success(Duration::from_millis(100));
        let s = c.snapshot();
        assert_eq!(s.total_requests, 1);
        assert_eq!(s.successful_requests, 1);
        assert_eq!(s.failed_requests, 0);
    }

    #[test]
    fn test_record_failure() {
        let c = MetricsCollector::new();
        c.record_failure(Duration::from_millis(100));
        let s = c.snapshot();
        assert_eq!(s.total_requests, 1);
        assert_eq!(s.successful_requests, 0);
        assert_eq!(s.failed_requests, 1);
    }

    #[test]
    fn test_cache_tracking() {
        let c = MetricsCollector::new();
        for _ in 0..8 {
            c.record_cache_hit();
        }
        for _ in 0..2 {
            c.record_cache_miss();
        }

        let s = c.snapshot();
        assert_eq!(s.cache_hits, 8);
        assert_eq!(s.cache_misses, 2);
        assert_eq!(s.cache_hit_rate_percent, 80.0);
    }

    #[test]
    fn test_cloneable() {
        let c1 = MetricsCollector::new();
        let c2 = c1.clone();

        c1.record_success(Duration::from_millis(100));
        // Both should see the same state
        assert_eq!(c2.snapshot().total_requests, 1);
    }
}
