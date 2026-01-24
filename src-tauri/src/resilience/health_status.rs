use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Individual component status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentStatus {
    /// Component name
    pub name: String,
    /// Is healthy (true/false)
    pub healthy: bool,
    /// Status message
    pub message: String,
    /// Optional metrics
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metrics: Option<HashMap<String, String>>,
}

impl ComponentStatus {
    /// Create healthy status
    pub fn healthy(name: &str, message: &str) -> Self {
        Self {
            name: name.to_string(),
            healthy: true,
            message: message.to_string(),
            metrics: None,
        }
    }

    /// Create unhealthy status
    pub fn unhealthy(name: &str, message: &str) -> Self {
        Self {
            name: name.to_string(),
            healthy: false,
            message: message.to_string(),
            metrics: None,
        }
    }

    /// Add metrics
    pub fn with_metrics(mut self, metrics: HashMap<String, String>) -> Self {
        self.metrics = Some(metrics);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_healthy() {
        let status = ComponentStatus::healthy("test", "All good");
        assert!(status.healthy);
        assert_eq!(status.message, "All good");
    }

    #[test]
    fn test_unhealthy() {
        let status = ComponentStatus::unhealthy("test", "Failed");
        assert!(!status.healthy);
        assert_eq!(status.message, "Failed");
    }

    #[test]
    fn test_with_metrics() {
        let mut metrics = HashMap::new();
        metrics.insert("usage".to_string(), "75%".to_string());

        let status = ComponentStatus::healthy("test", "OK").with_metrics(metrics);
        assert!(status.metrics.is_some());
        assert_eq!(
            status.metrics.as_ref().unwrap().get("usage").unwrap(),
            "75%"
        );
    }
}
