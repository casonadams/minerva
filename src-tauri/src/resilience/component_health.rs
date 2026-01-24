use super::health_status::ComponentStatus;
use serde::{Deserialize, Serialize};

/// Individual component health
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentHealth {
    /// GPU status
    pub gpu: ComponentStatus,
    /// CPU status
    pub cpu: ComponentStatus,
    /// Memory status
    pub memory: ComponentStatus,
    /// Model cache status
    pub model_cache: ComponentStatus,
}

impl ComponentHealth {
    /// Create all healthy
    pub fn all_healthy() -> Self {
        Self {
            gpu: ComponentStatus::healthy("gpu", "GPU available"),
            cpu: ComponentStatus::healthy("cpu", "CPU available"),
            memory: ComponentStatus::healthy("memory", "Memory healthy"),
            model_cache: ComponentStatus::healthy("model_cache", "Cache operational"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_all_healthy() {
        let health = ComponentHealth::all_healthy();
        assert!(health.gpu.healthy);
        assert!(health.cpu.healthy);
        assert!(health.memory.healthy);
        assert!(health.model_cache.healthy);
    }
}
