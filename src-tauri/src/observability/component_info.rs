use serde::{Deserialize, Serialize};
use std::collections::HashMap;

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
}
