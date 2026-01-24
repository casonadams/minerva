use super::metrics_snapshot::MetricsSnapshot;
use std::time::Duration;

/// Metrics calculation utilities
pub struct MetricsCalculator;

impl MetricsCalculator {
    /// Calculate metrics from raw data
    pub fn calculate(total: u64, success: u64, response_times: &[Duration]) -> MetricsSnapshot {
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

        MetricsSnapshot {
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
    pub fn percentiles(response_times: &[Duration]) -> (f64, f64) {
        if response_times.is_empty() {
            return (0.0, 0.0);
        }

        let mut times: Vec<u128> = response_times.iter().map(|d| d.as_millis()).collect();
        times.sort_unstable();

        let p95_idx = ((times.len() as f64 * 0.95) as usize).min(times.len() - 1);
        let p99_idx = ((times.len() as f64 * 0.99) as usize).min(times.len() - 1);

        (times[p95_idx] as f64, times[p99_idx] as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calculate() {
        let times = vec![
            Duration::from_millis(100),
            Duration::from_millis(150),
            Duration::from_millis(200),
        ];
        let m = MetricsCalculator::calculate(3, 3, &times);
        assert_eq!(m.total_requests, 3);
        assert_eq!(m.failed_requests, 0);
        assert_eq!(m.error_rate_percent, 0.0);
    }

    #[test]
    fn test_error_rate() {
        let times = vec![Duration::from_millis(100); 10];
        let m = MetricsCalculator::calculate(10, 8, &times);
        assert_eq!(m.failed_requests, 2);
        assert_eq!(m.error_rate_percent, 20.0);
    }

    #[test]
    fn test_percentiles() {
        let times: Vec<Duration> = (1..=100).map(Duration::from_millis).collect();
        let m = MetricsCalculator::calculate(100, 100, &times);

        assert!(m.p95_response_time_ms >= 90.0);
        assert!(m.p99_response_time_ms >= 95.0);
    }
}
