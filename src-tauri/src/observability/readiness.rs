use serde::{Deserialize, Serialize};

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

#[cfg(test)]
mod tests {
    use super::*;

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
}
