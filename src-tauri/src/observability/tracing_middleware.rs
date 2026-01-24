/// Request Tracing Middleware
///
/// Integrates with logging infrastructure for distributed tracing:
/// - Request ID propagation
/// - Latency tracking
/// - Error logging
/// - Metrics collection
pub use crate::observability::request_trace::RequestTrace;
pub use crate::observability::trace_id_generator::TraceIdGenerator;
