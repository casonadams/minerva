use super::metrics::MetricsSnapshot;
use super::metrics_analyzer::MetricsAnalyzer;
use std::time::Duration;

/// Builds metrics snapshots from raw metrics state
pub struct SnapshotBuilder;

impl SnapshotBuilder {
    /// Build snapshot from current metrics state
    pub fn build(
        total: u64,
        success: u64,
        failed: u64,
        hits: u64,
        misses: u64,
        times: &[Duration],
        uptime_secs: u64,
    ) -> MetricsSnapshot {
        let (avg, min, max, p50, p95, p99) = MetricsAnalyzer::analyze_times(times);

        let rps = if uptime_secs > 0 {
            total as f64 / uptime_secs as f64
        } else {
            0.0
        };

        let error_rate = if total > 0 {
            (failed as f64 / total as f64) * 100.0
        } else {
            0.0
        };

        let hit_rate = if (hits + misses) > 0 {
            (hits as f64 / (hits + misses) as f64) * 100.0
        } else {
            0.0
        };

        MetricsSnapshot {
            total_requests: total,
            successful_requests: success,
            failed_requests: failed,
            avg_response_time_ms: avg,
            min_response_time_ms: min,
            max_response_time_ms: max,
            p50_response_time_ms: p50,
            p95_response_time_ms: p95,
            p99_response_time_ms: p99,
            rps,
            error_rate_percent: error_rate,
            cache_hits: hits,
            cache_misses: misses,
            cache_hit_rate_percent: hit_rate,
            uptime_seconds: uptime_secs,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_snapshot_builder_with_zero_metrics() {
        let snapshot = SnapshotBuilder::build(0, 0, 0, 0, 0, &[], 0);
        assert_eq!(snapshot.total_requests, 0);
        assert_eq!(snapshot.error_rate_percent, 0.0);
        assert_eq!(snapshot.cache_hit_rate_percent, 0.0);
    }

    #[test]
    fn test_snapshot_builder_rps_calculation() {
        let snapshot = SnapshotBuilder::build(100, 90, 10, 50, 50, &[], 10);
        assert_eq!(snapshot.rps, 10.0); // 100 / 10 = 10.0
    }

    #[test]
    fn test_snapshot_builder_error_rate() {
        let snapshot = SnapshotBuilder::build(100, 80, 20, 0, 0, &[], 5);
        assert_eq!(snapshot.error_rate_percent, 20.0);
    }

    #[test]
    fn test_snapshot_builder_hit_rate() {
        let snapshot = SnapshotBuilder::build(0, 0, 0, 80, 20, &[], 0);
        assert_eq!(snapshot.cache_hit_rate_percent, 80.0);
    }
}
