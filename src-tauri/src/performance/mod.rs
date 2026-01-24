/// Performance Optimization & Monitoring
///
/// Desktop-focused performance features:
/// - Adaptive resource usage based on system state
/// - Performance profiling and metrics
/// - Memory-aware inference configuration
/// - UI responsiveness optimization
pub mod adaptive;
pub mod execution_modes;
pub mod inference_metrics_query;
pub mod integration;
pub mod metrics_analyzer_integration;
pub mod metrics_queries;
pub mod metrics_storage;
pub mod operation_profile;
pub mod performance_metrics;
pub mod profile_analyzer;
pub mod profiler;
pub mod resource_state;
pub mod scoped_timer;
pub mod window_state;

pub use performance_metrics::PerformanceMetrics;
pub use resource_state::ResourceState;
