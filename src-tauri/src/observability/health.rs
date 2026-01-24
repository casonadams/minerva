use serde::{Deserialize, Serialize};
use std::collections::HashMap;

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
    pub details: Option<HashMap<String, String>>,
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
    pub fn with_details(mut self, details: HashMap<String, String>) -> Self {
        self.details = Some(details);
        self
    }
}

impl Default for ComponentInfo {
    fn default() -> Self {
        Self::operational("OK")
    }
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
}
