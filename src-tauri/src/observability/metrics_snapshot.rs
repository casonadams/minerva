use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Server metrics snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsSnapshot {
    /// Total requests processed
    pub total_requests: u64,
    /// Successful requests
    pub successful_requests: u64,
    /// Failed requests
    pub failed_requests: u64,
    /// Average response time in ms
    pub avg_response_time_ms: f64,
    /// P95 response time in ms
    pub p95_response_time_ms: f64,
    /// P99 response time in ms
    pub p99_response_time_ms: f64,
    /// Requests per second
    pub rps: f64,
    /// Error rate percentage
    pub error_rate_percent: f64,
}

impl Default for MetricsSnapshot {
    fn default() -> Self {
        Self {
            total_requests: 0,
            successful_requests: 0,
            failed_requests: 0,
            avg_response_time_ms: 0.0,
            p95_response_time_ms: 0.0,
            p99_response_time_ms: 0.0,
            rps: 0.0,
            error_rate_percent: 0.0,
        }
    }
}

impl MetricsSnapshot {
    /// Calculate metrics from raw data
    pub fn calculate(total: u64, success: u64, response_times: &[Duration]) -> Self {
        let failed = total.saturating_sub(success);
        let error_rate = if total > 0 {
            (failed as f64 / total as f64) * 100.0
        } else {
            0.0
        };

        let avg_ms = if !response_times.is_empty() {
            let sum: u128 = response_times.iter().map(|d| d.as_millis()).sum();
            sum as f64 / response_times.len() as f64
        } else {
            0.0
        };

        let (p95_ms, p99_ms) = Self::percentiles(response_times);

        Self {
            total_requests: total,
            successful_requests: success,
            failed_requests: failed,
            avg_response_time_ms: avg_ms,
            p95_response_time_ms: p95_ms,
            p99_response_time_ms: p99_ms,
            rps: 0.0,
            error_rate_percent: error_rate,
        }
    }

    /// Calculate percentiles
    fn percentiles(response_times: &[Duration]) -> (f64, f64) {
        if response_times.is_empty() {
            return (0.0, 0.0);
        }

        let mut times: Vec<u128> = response_times.iter().map(|d| d.as_millis()).collect();
        times.sort_unstable();

        let p95_idx = ((times.len() as f64 * 0.95) as usize).min(times.len() - 1);
        let p99_idx = ((times.len() as f64 * 0.99) as usize).min(times.len() - 1);

        (times[p95_idx] as f64, times[p99_idx] as f64)
    }

    /// Is service healthy based on metrics?
    pub fn is_healthy(&self) -> bool {
        self.error_rate_percent < 5.0 && self.p99_response_time_ms < 5000.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metrics_default() {
        let m = MetricsSnapshot::default();
        assert_eq!(m.total_requests, 0);
        assert_eq!(m.successful_requests, 0);
    }

    #[test]
    fn test_metrics_calculate() {
        let times = vec![
            Duration::from_millis(100),
            Duration::from_millis(150),
            Duration::from_millis(200),
        ];
        let m = MetricsSnapshot::calculate(3, 3, &times);
        assert_eq!(m.total_requests, 3);
        assert_eq!(m.failed_requests, 0);
        assert_eq!(m.error_rate_percent, 0.0);
    }

    #[test]
    fn test_metrics_error_rate() {
        let times = vec![Duration::from_millis(100); 10];
        let m = MetricsSnapshot::calculate(10, 8, &times);
        assert_eq!(m.failed_requests, 2);
        assert_eq!(m.error_rate_percent, 20.0);
    }

    #[test]
    fn test_metrics_percentiles() {
        let times: Vec<Duration> = (1..=100).map(Duration::from_millis).collect();
        let m = MetricsSnapshot::calculate(100, 100, &times);

        assert!(m.p95_response_time_ms >= 90.0);
        assert!(m.p99_response_time_ms >= 95.0);
    }

    #[test]
    fn test_metrics_is_healthy() {
        let m = MetricsSnapshot {
            error_rate_percent: 2.0,
            p99_response_time_ms: 1000.0,
            ..Default::default()
        };
        assert!(m.is_healthy());
    }

    #[test]
    fn test_metrics_unhealthy_error_rate() {
        let m = MetricsSnapshot {
            error_rate_percent: 10.0,
            p99_response_time_ms: 1000.0,
            ..Default::default()
        };
        assert!(!m.is_healthy());
    }

    #[test]
    fn test_metrics_unhealthy_latency() {
        let m = MetricsSnapshot {
            error_rate_percent: 1.0,
            p99_response_time_ms: 10000.0,
            ..Default::default()
        };
        assert!(!m.is_healthy());
    }
}
