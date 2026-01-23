/// Observable Server Endpoints
///
/// HTTP endpoints for health, readiness, and metrics:
/// - GET /health - Component health status
/// - GET /ready - Readiness probe
/// - GET /metrics - Performance metrics
use serde::{Deserialize, Serialize};

/// Health response for /health endpoint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthEndpointResponse {
    /// Overall health status (healthy/degraded/unhealthy)
    pub status: String,
    /// Timestamp RFC3339
    pub timestamp: String,
    /// Component statuses
    pub components: ComponentStatuses,
    /// Uptime in seconds
    pub uptime_seconds: u64,
    /// Version info
    pub version: String,
}

/// Individual component statuses
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentStatuses {
    /// GPU availability and status
    pub gpu: ComponentInfo,
    /// CPU availability and status
    pub cpu: ComponentInfo,
    /// Memory availability and status
    pub memory: ComponentInfo,
    /// Model loading system status
    pub models: ComponentInfo,
    /// Inference pipeline status
    pub inference: ComponentInfo,
}

/// Component status details
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentInfo {
    /// Is this component operational
    pub operational: bool,
    /// Human-readable message
    pub message: String,
    /// Optional metrics (usage %, latency, etc)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub details: Option<std::collections::HashMap<String, String>>,
}

impl ComponentInfo {
    /// Create operational component
    pub fn operational(message: &str) -> Self {
        Self {
            operational: true,
            message: message.to_string(),
            details: None,
        }
    }

    /// Create degraded/failed component
    pub fn degraded(message: &str) -> Self {
        Self {
            operational: false,
            message: message.to_string(),
            details: None,
        }
    }

    /// Add details
    pub fn with_details(mut self, details: std::collections::HashMap<String, String>) -> Self {
        self.details = Some(details);
        self
    }
}

impl Default for ComponentInfo {
    fn default() -> Self {
        Self::operational("OK")
    }
}

/// Readiness response for /ready endpoint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReadinessResponse {
    /// Is service ready to accept traffic
    pub ready: bool,
    /// Reason if not ready
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
    /// Dependencies that are blocking readiness
    #[serde(skip_serializing_if = "Option::is_none")]
    pub blocking_components: Option<Vec<String>>,
}

impl ReadinessResponse {
    /// Create ready response
    pub fn ready() -> Self {
        Self {
            ready: true,
            reason: None,
            blocking_components: None,
        }
    }

    /// Create not-ready response
    pub fn not_ready(reason: &str) -> Self {
        Self {
            ready: false,
            reason: Some(reason.to_string()),
            blocking_components: None,
        }
    }

    /// Add blocking components
    pub fn with_blocking(mut self, components: Vec<String>) -> Self {
        self.blocking_components = Some(components);
        self
    }
}

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

impl Default for HealthEndpointResponse {
    fn default() -> Self {
        Self {
            status: "healthy".to_string(),
            timestamp: chrono::Local::now().to_rfc3339(),
            components: ComponentStatuses {
                gpu: ComponentInfo::operational("Available"),
                cpu: ComponentInfo::operational("Available"),
                memory: ComponentInfo::operational("Healthy"),
                models: ComponentInfo::operational("Ready"),
                inference: ComponentInfo::operational("Ready"),
            },
            uptime_seconds: 0,
            version: "0.1.0".to_string(),
        }
    }
}

impl HealthEndpointResponse {
    /// Determine overall status from components
    pub fn calculate_status(&mut self) {
        let all_operational = self.components.gpu.operational
            && self.components.cpu.operational
            && self.components.memory.operational
            && self.components.models.operational
            && self.components.inference.operational;

        self.status = if all_operational {
            "healthy".to_string()
        } else if self.components.cpu.operational && self.components.models.operational {
            "degraded".to_string()
        } else {
            "unhealthy".to_string()
        };
    }

    /// Is service healthy?
    pub fn is_healthy(&self) -> bool {
        self.status == "healthy"
    }

    /// Can accept requests?
    pub fn can_accept_requests(&self) -> bool {
        self.components.cpu.operational && self.components.models.operational
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_component_info_operational() {
        let c = ComponentInfo::operational("test");
        assert!(c.operational);
    }

    #[test]
    fn test_component_info_degraded() {
        let c = ComponentInfo::degraded("test");
        assert!(!c.operational);
    }

    #[test]
    fn test_health_response_default() {
        let resp = HealthEndpointResponse::default();
        assert!(resp.is_healthy());
    }

    #[test]
    fn test_health_response_calculate_status() {
        let mut resp = HealthEndpointResponse::default();
        resp.components.gpu.operational = false;
        resp.calculate_status();
        assert_eq!(resp.status, "degraded");
    }

    #[test]
    fn test_health_response_unhealthy() {
        let mut resp = HealthEndpointResponse::default();
        resp.components.cpu.operational = false;
        resp.calculate_status();
        assert_eq!(resp.status, "unhealthy");
        assert!(!resp.is_healthy());
    }

    #[test]
    fn test_health_response_can_accept_requests() {
        let resp = HealthEndpointResponse::default();
        assert!(resp.can_accept_requests());

        let mut resp = HealthEndpointResponse::default();
        resp.components.cpu.operational = false;
        assert!(!resp.can_accept_requests());
    }

    #[test]
    fn test_readiness_ready() {
        let r = ReadinessResponse::ready();
        assert!(r.ready);
        assert!(r.reason.is_none());
    }

    #[test]
    fn test_readiness_not_ready() {
        let r = ReadinessResponse::not_ready("initializing");
        assert!(!r.ready);
        assert!(r.reason.is_some());
    }

    #[test]
    fn test_readiness_with_blocking() {
        let r = ReadinessResponse::not_ready("blocked").with_blocking(vec!["gpu".to_string()]);
        assert!(r.blocking_components.is_some());
        assert_eq!(r.blocking_components.unwrap().len(), 1);
    }

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
