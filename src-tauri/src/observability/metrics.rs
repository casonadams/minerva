pub use super::metrics_collector::MetricsCollector;

/// Metrics snapshot
#[derive(Debug, Clone)]
pub struct MetricsSnapshot {
    pub total_requests: u64,
    pub successful_requests: u64,
    pub failed_requests: u64,
    pub avg_response_time_ms: f64,
    pub min_response_time_ms: f64,
    pub max_response_time_ms: f64,
    pub p50_response_time_ms: f64,
    pub p95_response_time_ms: f64,
    pub p99_response_time_ms: f64,
    pub rps: f64,
    pub error_rate_percent: f64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub cache_hit_rate_percent: f64,
    pub uptime_seconds: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_metrics_snapshot_creation() {
        let snapshot = MetricsSnapshot {
            total_requests: 100,
            successful_requests: 95,
            failed_requests: 5,
            avg_response_time_ms: 50.5,
            min_response_time_ms: 10.0,
            max_response_time_ms: 200.0,
            p50_response_time_ms: 45.0,
            p95_response_time_ms: 150.0,
            p99_response_time_ms: 195.0,
            rps: 10.0,
            error_rate_percent: 5.0,
            cache_hits: 80,
            cache_misses: 20,
            cache_hit_rate_percent: 80.0,
            uptime_seconds: 3600,
        };
        assert_eq!(snapshot.total_requests, 100);
        assert_eq!(snapshot.error_rate_percent, 5.0);
    }

    #[test]
    fn test_collector_error_rate() {
        let c = MetricsCollector::new();
        for _ in 0..8 {
            c.record_success(Duration::from_millis(100));
        }
        for _ in 0..2 {
            c.record_failure(Duration::from_millis(100));
        }

        let s = c.snapshot();
        assert_eq!(s.total_requests, 10);
        assert_eq!(s.failed_requests, 2);
        assert_eq!(s.error_rate_percent, 20.0);
    }

    #[test]
    fn test_collector_response_time_percentiles() {
        let c = MetricsCollector::new();
        for i in 1..=100 {
            c.record_success(Duration::from_millis(i));
        }

        let s = c.snapshot();
        assert!(s.avg_response_time_ms > 0.0);
        assert!(s.p95_response_time_ms >= s.p50_response_time_ms);
        assert!(s.p99_response_time_ms >= s.p95_response_time_ms);
    }

    #[test]
    fn test_collector_rps_calculation() {
        let c = MetricsCollector::new();
        c.record_success(Duration::from_millis(100));
        c.record_success(Duration::from_millis(100));

        let s = c.snapshot();
        assert!(s.rps >= 0.0);
    }
}
