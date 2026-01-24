use serde::{Deserialize, Serialize};

/// Metrics response for /metrics endpoint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsResponse {
    /// Timestamp of metrics collection
    pub timestamp: String,
    /// Server uptime seconds
    pub uptime_seconds: u64,
    /// Request metrics
    pub requests: RequestMetrics,
    /// Response time metrics in milliseconds
    pub response_times: ResponseTimeMetrics,
    /// Error metrics
    pub errors: ErrorMetrics,
    /// Cache metrics
    pub cache: CacheMetrics,
}

/// Request statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestMetrics {
    /// Total requests processed
    pub total: u64,
    /// Successful requests
    pub successful: u64,
    /// Failed requests
    pub failed: u64,
    /// Requests per second
    pub rps: f64,
}

/// Response time statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseTimeMetrics {
    /// Average response time in ms
    pub avg_ms: f64,
    /// Minimum response time in ms
    pub min_ms: f64,
    /// Maximum response time in ms
    pub max_ms: f64,
    /// P50 (median) in ms
    pub p50_ms: f64,
    /// P95 percentile in ms
    pub p95_ms: f64,
    /// P99 percentile in ms
    pub p99_ms: f64,
}

/// Error statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorMetrics {
    /// Total errors
    pub total: u64,
    /// Error rate percentage
    pub rate_percent: f64,
    /// Most common error type
    pub top_error: Option<String>,
}

/// Cache statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheMetrics {
    /// Cache hits
    pub hits: u64,
    /// Cache misses
    pub misses: u64,
    /// Hit rate percentage
    pub hit_rate_percent: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metrics_response_default() {
        let m = MetricsResponse {
            timestamp: chrono::Local::now().to_rfc3339(),
            uptime_seconds: 60,
            requests: RequestMetrics {
                total: 100,
                successful: 95,
                failed: 5,
                rps: 1.67,
            },
            response_times: ResponseTimeMetrics {
                avg_ms: 150.0,
                min_ms: 50.0,
                max_ms: 500.0,
                p50_ms: 120.0,
                p95_ms: 300.0,
                p99_ms: 450.0,
            },
            errors: ErrorMetrics {
                total: 5,
                rate_percent: 5.0,
                top_error: Some("timeout".to_string()),
            },
            cache: CacheMetrics {
                hits: 80,
                misses: 20,
                hit_rate_percent: 80.0,
            },
        };

        assert_eq!(m.requests.total, 100);
        assert_eq!(m.requests.failed, 5);
        assert!(m.response_times.avg_ms > 0.0);
    }

    #[test]
    fn test_metrics_serialization() {
        let m = MetricsResponse {
            timestamp: chrono::Local::now().to_rfc3339(),
            uptime_seconds: 60,
            requests: RequestMetrics {
                total: 100,
                successful: 95,
                failed: 5,
                rps: 1.67,
            },
            response_times: ResponseTimeMetrics {
                avg_ms: 150.0,
                min_ms: 50.0,
                max_ms: 500.0,
                p50_ms: 120.0,
                p95_ms: 300.0,
                p99_ms: 450.0,
            },
            errors: ErrorMetrics {
                total: 5,
                rate_percent: 5.0,
                top_error: None,
            },
            cache: CacheMetrics {
                hits: 80,
                misses: 20,
                hit_rate_percent: 80.0,
            },
        };

        let json = serde_json::to_string(&m).unwrap();
        assert!(json.contains("total"));
        assert!(json.contains("uptime_seconds"));
    }
}
