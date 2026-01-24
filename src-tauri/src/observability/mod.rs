/// Server Observability & Metrics
///
/// Provides comprehensive observability for production deployments:
/// - Health check endpoints with component status
/// - Readiness probes for orchestration
/// - Performance metrics collection
/// - Request tracing and logging
pub mod component_info;
pub mod endpoints;
pub mod health;
pub mod health_types;
pub mod metrics;
pub mod metrics_analyzer;
pub mod metrics_calculator;
pub mod metrics_recorder;
pub mod metrics_response;
pub mod metrics_snapshot;
pub mod metrics_snapshot_builder;
pub mod readiness;
pub mod request_trace;
pub mod response_time_store;
pub mod trace_id_generator;
pub mod tracing_middleware;

pub use metrics_snapshot::MetricsSnapshot;
