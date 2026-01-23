/// Request and span context management for Phase 7
///
/// Manages request IDs and execution spans to track:
/// - Request flow through the system
/// - Operation boundaries
/// - Error contexts
/// - Performance metrics per span
use std::sync::atomic::{AtomicU64, Ordering};
use uuid::Uuid;

static REQUEST_ID_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Generate a unique request ID for tracing through the system
pub fn generate_request_id() -> String {
    let counter = REQUEST_ID_COUNTER.fetch_add(1, Ordering::SeqCst);
    let uuid_str = Uuid::new_v4().to_string();
    format!("{}-{}", &uuid_str[..8], counter)
}

/// Create a request span with ID and metadata
///
/// Returns span guard that will be entered for async operations
#[macro_export]
macro_rules! request_span {
    ($request_id:expr, $($key:tt = $value:tt),*) => {
        {
            tracing::info_span!(
                "request",
                request_id = %$request_id,
                $($key = $value),*
            )
        }
    };
}

/// Create an operation span for specific tasks
#[macro_export]
macro_rules! operation_span {
    ($name:expr, $($key:tt = $value:tt),*) => {
        {
            tracing::debug_span!(
                $name,
                $($key = $value),*
            )
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_request_id() {
        let id1 = generate_request_id();
        let id2 = generate_request_id();

        assert!(!id1.is_empty());
        assert!(!id2.is_empty());
        assert_ne!(id1, id2); // Should be unique
        assert!(id1.contains('-')); // Should have uuid prefix
    }

    #[test]
    fn test_request_id_format() {
        let id = generate_request_id();
        let parts: Vec<&str> = id.split('-').collect();

        // Format: uuid_prefix-counter
        assert!(parts.len() >= 2);
        assert!(!parts[0].is_empty());
    }

    #[test]
    fn test_request_id_uniqueness() {
        let mut ids = std::collections::HashSet::new();

        for _ in 0..100 {
            let id = generate_request_id();
            assert!(ids.insert(id), "Generated duplicate request ID");
        }

        assert_eq!(ids.len(), 100);
    }

    #[test]
    fn test_span_macros() {
        crate::logging::spans::tests::init_test_logging();

        let request_id = generate_request_id();
        let span = request_span!(request_id, model = "test-model");
        let _guard = span.enter();

        tracing::info!("Test message in span");
        // Test passes if span creation and entry works
    }

    fn init_test_logging() {
        let _ = tracing_subscriber::fmt()
            .with_test_writer()
            .with_max_level(tracing::Level::DEBUG)
            .try_init();
    }
}
