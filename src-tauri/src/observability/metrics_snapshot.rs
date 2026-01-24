use serde::{Deserialize, Serialize};

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
    /// Is service healthy based on metrics?
    pub fn is_healthy(&self) -> bool {
        self.error_rate_percent < 5.0 && self.p99_response_time_ms < 5000.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default() {
        let m = MetricsSnapshot::default();
        assert_eq!(m.total_requests, 0);
        assert_eq!(m.successful_requests, 0);
    }

    #[test]
    fn test_is_healthy() {
        let m = MetricsSnapshot {
            error_rate_percent: 2.0,
            p99_response_time_ms: 1000.0,
            ..Default::default()
        };
        assert!(m.is_healthy());
    }

    #[test]
    fn test_unhealthy_error_rate() {
        let m = MetricsSnapshot {
            error_rate_percent: 10.0,
            p99_response_time_ms: 1000.0,
            ..Default::default()
        };
        assert!(!m.is_healthy());
    }

    #[test]
    fn test_unhealthy_latency() {
        let m = MetricsSnapshot {
            error_rate_percent: 1.0,
            p99_response_time_ms: 10000.0,
            ..Default::default()
        };
        assert!(!m.is_healthy());
    }
}
