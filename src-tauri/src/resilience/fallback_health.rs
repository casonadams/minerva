/// Health check status
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HealthStatus {
    /// All systems operational
    Healthy,
    /// Degraded but operational
    Degraded,
    /// Not operational
    Unhealthy,
}

/// Memory health status
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryStatus {
    /// Plenty of memory available
    Healthy,
    /// Some memory pressure
    Moderate,
    /// Critical memory pressure
    Critical,
}

/// System health indicators
#[derive(Debug, Clone)]
pub struct HealthCheck {
    /// GPU availability
    pub gpu_available: bool,
    /// CPU available
    pub cpu_available: bool,
    /// Memory status
    pub memory_status: MemoryStatus,
    /// Overall status
    pub overall: HealthStatus,
}

impl Default for HealthCheck {
    fn default() -> Self {
        Self {
            gpu_available: true,
            cpu_available: true,
            memory_status: MemoryStatus::Healthy,
            overall: HealthStatus::Healthy,
        }
    }
}

impl HealthCheck {
    /// Check overall health
    pub fn health_status(&self) -> HealthStatus {
        if !self.cpu_available || matches!(self.memory_status, MemoryStatus::Critical) {
            HealthStatus::Unhealthy
        } else if !self.gpu_available || matches!(self.memory_status, MemoryStatus::Moderate) {
            HealthStatus::Degraded
        } else {
            HealthStatus::Healthy
        }
    }

    /// Can process requests?
    pub fn can_process(&self) -> bool {
        !matches!(self.health_status(), HealthStatus::Unhealthy)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_health_check_default() {
        let hc = HealthCheck::default();
        assert!(hc.gpu_available);
        assert!(hc.cpu_available);
        assert_eq!(hc.memory_status, MemoryStatus::Healthy);
        assert_eq!(hc.health_status(), HealthStatus::Healthy);
    }

    #[test]
    fn test_health_check_degraded() {
        let hc = HealthCheck {
            gpu_available: false,
            ..Default::default()
        };
        assert_eq!(hc.health_status(), HealthStatus::Degraded);
        assert!(hc.can_process());
    }

    #[test]
    fn test_health_check_unhealthy_no_cpu() {
        let hc = HealthCheck {
            cpu_available: false,
            ..Default::default()
        };
        assert_eq!(hc.health_status(), HealthStatus::Unhealthy);
        assert!(!hc.can_process());
    }

    #[test]
    fn test_health_check_unhealthy_critical_memory() {
        let hc = HealthCheck {
            memory_status: MemoryStatus::Critical,
            ..Default::default()
        };
        assert_eq!(hc.health_status(), HealthStatus::Unhealthy);
        assert!(!hc.can_process());
    }

    #[test]
    fn test_health_check_degraded_moderate_memory() {
        let hc = HealthCheck {
            memory_status: MemoryStatus::Moderate,
            ..Default::default()
        };
        assert_eq!(hc.health_status(), HealthStatus::Degraded);
        assert!(hc.can_process());
    }
}
