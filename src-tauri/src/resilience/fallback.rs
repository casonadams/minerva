pub use crate::resilience::fallback_health::{HealthCheck, HealthStatus, MemoryStatus};
/// Fallback Mechanisms for Graceful Degradation
///
/// Provides fallback strategies when primary methods fail:
/// - GPU → CPU fallback for inference
/// - Primary model → fallback model
/// - Streaming → batch fallback
/// - Resource constraints handling
pub use crate::resilience::fallback_strategy::{FallbackDecision, FallbackStrategy};
