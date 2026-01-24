/// Health status responses for the service
///
/// Tracks:
/// - Overall health status (healthy/degraded/unhealthy)
/// - Component availability (GPU, CPU, memory, models, inference)
/// - Request acceptance capability
pub use super::component_info::ComponentInfo;
pub use super::health_types::{ComponentStatuses, HealthEndpointResponse};
