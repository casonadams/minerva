/// Fallback Mechanisms for Graceful Degradation
///
/// Provides fallback strategies when primary methods fail:
/// - GPU → CPU fallback for inference
/// - Primary model → fallback model
/// - Streaming → batch fallback
/// - Resource constraints handling
use crate::error::MinervaError;

/// Fallback strategy options
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FallbackStrategy {
    /// Use CPU instead of GPU
    GpuToCpu,
    /// Use smaller/faster model
    UseAltModel,
    /// Switch from streaming to batch
    StreamingToBatch,
    /// Reduce batch size and retry
    ReduceBatchSize,
    /// Use cached result if available
    UseCache,
    /// No fallback available
    None,
}

/// Fallback decision for errors
pub struct FallbackDecision;

impl FallbackDecision {
    /// Determine fallback strategy for an error
    pub fn strategy_for(error: &MinervaError) -> FallbackStrategy {
        match error {
            // GPU failures → try CPU
            MinervaError::GpuOutOfMemory(_) | MinervaError::GpuContextLost(_) => {
                FallbackStrategy::GpuToCpu
            }

            // Resource exhaustion → reduce batch or use cache
            MinervaError::OutOfMemory(_) => FallbackStrategy::ReduceBatchSize,

            // Model corruption → try alternate model
            MinervaError::ModelCorrupted(_) => FallbackStrategy::UseAltModel,

            // Streaming errors → try batch
            MinervaError::StreamingError(_) => FallbackStrategy::StreamingToBatch,

            // Other errors → no fallback
            _ => FallbackStrategy::None,
        }
    }

    /// Get human-readable fallback message
    pub fn message(strategy: FallbackStrategy) -> &'static str {
        match strategy {
            FallbackStrategy::GpuToCpu => "GPU unavailable, switching to CPU inference...",
            FallbackStrategy::UseAltModel => "Primary model unavailable, trying alternate model...",
            FallbackStrategy::StreamingToBatch => {
                "Streaming failed, switching to batch inference..."
            }
            FallbackStrategy::ReduceBatchSize => "Reducing batch size due to memory constraints...",
            FallbackStrategy::UseCache => "Using cached result...",
            FallbackStrategy::None => "No fallback strategy available",
        }
    }

    /// Is fallback available?
    pub fn is_available(strategy: FallbackStrategy) -> bool {
        !matches!(strategy, FallbackStrategy::None)
    }
}

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
    fn test_fallback_gpu_error() {
        let err = MinervaError::GpuOutOfMemory("16GB".to_string());
        let strategy = FallbackDecision::strategy_for(&err);
        assert_eq!(strategy, FallbackStrategy::GpuToCpu);
    }

    #[test]
    fn test_fallback_streaming_error() {
        let err = MinervaError::StreamingError("connection".to_string());
        let strategy = FallbackDecision::strategy_for(&err);
        assert_eq!(strategy, FallbackStrategy::StreamingToBatch);
    }

    #[test]
    fn test_fallback_memory_error() {
        let err = MinervaError::OutOfMemory("heap".to_string());
        let strategy = FallbackDecision::strategy_for(&err);
        assert_eq!(strategy, FallbackStrategy::ReduceBatchSize);
    }

    #[test]
    fn test_fallback_model_corrupted() {
        let err = MinervaError::ModelCorrupted("header".to_string());
        let strategy = FallbackDecision::strategy_for(&err);
        assert_eq!(strategy, FallbackStrategy::UseAltModel);
    }

    #[test]
    fn test_fallback_no_strategy() {
        let err = MinervaError::InvalidRequest("bad".to_string());
        let strategy = FallbackDecision::strategy_for(&err);
        assert_eq!(strategy, FallbackStrategy::None);
    }

    #[test]
    fn test_fallback_is_available() {
        assert!(FallbackDecision::is_available(FallbackStrategy::GpuToCpu));
        assert!(!FallbackDecision::is_available(FallbackStrategy::None));
    }

    #[test]
    fn test_fallback_messages() {
        let msg = FallbackDecision::message(FallbackStrategy::GpuToCpu);
        assert!(msg.contains("CPU"));

        let msg = FallbackDecision::message(FallbackStrategy::StreamingToBatch);
        assert!(msg.contains("batch"));
    }

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
