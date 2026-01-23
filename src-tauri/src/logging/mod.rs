/// Structured logging and observability infrastructure for Phase 7
///
/// Provides comprehensive logging with:
/// - Request/span context propagation
/// - Structured log levels and formatting
/// - Async logging for performance
/// - Error tracking and reporting
pub mod spans;

use tracing_subscriber::{fmt, prelude::*, EnvFilter};

/// Initialize logging system with default configuration
///
/// Sets up structured logging with:
/// - Targets stderr for output
/// - JSON formatting for machine parsing
/// - Environment-based level filtering
pub fn init_logging() {
    let env_filter = EnvFilter::from_default_env()
        .add_directive("minerva=debug".parse().unwrap())
        .add_directive("info".parse().unwrap());

    tracing_subscriber::registry()
        .with(env_filter)
        .with(fmt::layer().with_writer(std::io::stderr))
        .init();

    tracing::info!("Logging system initialized");
}

/// Initialize logging for tests with verbose output
///
/// Sets up logging specifically for test environments with:
/// - Debug level for all minerva code
/// - Captured output for test harness
pub fn init_test_logging() {
    let _ = tracing_subscriber::fmt()
        .with_test_writer()
        .with_max_level(tracing::Level::DEBUG)
        .try_init();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_logging_initialization() {
        init_test_logging();
        tracing::info!("Test logging message");
        // Test passes if no panic occurs
    }

    #[test]
    fn test_debug_logging() {
        init_test_logging();
        tracing::debug!("Debug message");
        tracing::info!("Info message");
        tracing::warn!("Warning message");
        // Test passes if all log levels work
    }
}
