/// Server Observability & Metrics
///
/// Provides comprehensive observability for production deployments:
/// - Health check endpoints with component status
/// - Readiness probes for orchestration
/// - Performance metrics collection
/// - Request tracing and logging
pub mod endpoints;
pub mod health;
pub mod metrics;
pub mod metrics_response;
pub mod metrics_snapshot;
pub mod readiness;
pub mod request_trace;
pub mod trace_id_generator;
pub mod tracing_middleware;

pub use metrics_snapshot::MetricsSnapshot;
