pub use super::component_health::ComponentHealth;
pub use super::health_status::ComponentStatus;
/// Health Check & Readiness Probes
///
/// Provides health status for:
/// - Service readiness
/// - Dependency availability
/// - Resource constraints
/// - Operational metrics
use serde::{Deserialize, Serialize};

/// Health check response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthResponse {
    /// Overall service status
    pub status: String,
    /// Timestamp of check
    pub timestamp: String,
    /// Component health details
    pub components: ComponentHealth,
    /// Optional metadata
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<std::collections::HashMap<String, String>>,
}

/// Ready check response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReadyResponse {
    /// Is service ready to accept traffic
    pub ready: bool,
    /// Reason if not ready
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
}

impl Default for HealthResponse {
    fn default() -> Self {
        Self {
            status: "healthy".to_string(),
            timestamp: chrono::Local::now().to_rfc3339(),
            components: ComponentHealth::all_healthy(),
            metadata: None,
        }
    }
}

impl HealthResponse {
    /// Create from component statuses
    pub fn from_components(components: ComponentHealth) -> Self {
        let overall_healthy = components.gpu.healthy
            && components.cpu.healthy
            && components.memory.healthy
            && components.model_cache.healthy;

        let status = if overall_healthy {
            "healthy".to_string()
        } else {
            "degraded".to_string()
        };

        Self {
            status,
            timestamp: chrono::Local::now().to_rfc3339(),
            components,
            metadata: None,
        }
    }

    /// Is service healthy?
    pub fn is_healthy(&self) -> bool {
        self.status == "healthy"
    }

    /// Can accept requests?
    pub fn can_serve(&self) -> bool {
        self.components.cpu.healthy
            && !self.components.memory.healthy
            && self.components.model_cache.healthy
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_health_response_default() {
        let resp = HealthResponse::default();
        assert_eq!(resp.status, "healthy");
        assert!(resp.is_healthy());
    }

    #[test]
    fn test_health_response_from_components() {
        let components = ComponentHealth::all_healthy();
        let resp = HealthResponse::from_components(components);
        assert_eq!(resp.status, "healthy");
        assert!(resp.is_healthy());
    }

    #[test]
    fn test_health_response_degraded() {
        let mut components = ComponentHealth::all_healthy();
        components.gpu = ComponentStatus::unhealthy("gpu", "Not available");

        let resp = HealthResponse::from_components(components);
        assert_eq!(resp.status, "degraded");
        assert!(!resp.is_healthy());
    }

    #[test]
    fn test_ready_response_ready() {
        let resp = ReadyResponse {
            ready: true,
            reason: None,
        };
        assert!(resp.ready);
    }

    #[test]
    fn test_ready_response_not_ready() {
        let resp = ReadyResponse {
            ready: false,
            reason: Some("Initializing".to_string()),
        };
        assert!(!resp.ready);
        assert!(resp.reason.is_some());
    }

    #[test]
    fn test_health_response_serialization() {
        let resp = HealthResponse::default();
        let json = serde_json::to_string(&resp).unwrap();
        assert!(json.contains("healthy"));
        assert!(json.contains("components"));
    }

    #[test]
    fn test_can_serve() {
        let mut components = ComponentHealth::all_healthy();
        components.memory = ComponentStatus::unhealthy("memory", "Pressure");

        let resp = HealthResponse::from_components(components);
        assert_eq!(resp.status, "degraded");
    }
}
