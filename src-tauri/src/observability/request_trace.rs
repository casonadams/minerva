use std::time::Instant;

/// Request tracing context
#[derive(Debug, Clone)]
pub struct RequestTrace {
    /// Unique request ID
    pub request_id: String,
    /// When request started
    pub start_time: Instant,
    /// Request method (GET, POST, etc)
    pub method: String,
    /// Request path
    pub path: String,
}

impl RequestTrace {
    /// Create new request trace
    pub fn new(request_id: String, method: String, path: String) -> Self {
        Self {
            request_id,
            start_time: Instant::now(),
            method,
            path,
        }
    }

    /// Get elapsed time since request started
    pub fn elapsed_ms(&self) -> u128 {
        self.start_time.elapsed().as_millis()
    }

    /// Create trace log entry
    pub fn log_entry(&self, status: u16) -> String {
        let elapsed = self.elapsed_ms();
        format!(
            "request_id={} method={} path={} status={} latency_ms={}",
            self.request_id, self.method, self.path, status, elapsed
        )
    }

    /// Create error log entry
    pub fn log_error(&self, error: &str) -> String {
        let elapsed = self.elapsed_ms();
        format!(
            "request_id={} method={} path={} error=\"{}\" latency_ms={}",
            self.request_id, self.method, self.path, error, elapsed
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn test_request_trace_creation() {
        let trace = RequestTrace::new(
            "test-id".to_string(),
            "GET".to_string(),
            "/health".to_string(),
        );
        assert_eq!(trace.request_id, "test-id");
        assert_eq!(trace.method, "GET");
        assert_eq!(trace.path, "/health");
    }

    #[test]
    fn test_request_trace_elapsed() {
        let trace = RequestTrace::new(
            "test-id".to_string(),
            "GET".to_string(),
            "/health".to_string(),
        );
        thread::sleep(std::time::Duration::from_millis(10));
        assert!(trace.elapsed_ms() >= 10);
    }

    #[test]
    fn test_request_trace_log_entry() {
        let trace = RequestTrace::new(
            "test-id-123".to_string(),
            "POST".to_string(),
            "/api/v1/models".to_string(),
        );
        let log = trace.log_entry(200);
        assert!(log.contains("request_id=test-id-123"));
        assert!(log.contains("method=POST"));
        assert!(log.contains("path=/api/v1/models"));
        assert!(log.contains("status=200"));
        assert!(log.contains("latency_ms="));
    }

    #[test]
    fn test_request_trace_log_error() {
        let trace = RequestTrace::new(
            "test-id".to_string(),
            "GET".to_string(),
            "/health".to_string(),
        );
        let log = trace.log_error("timeout");
        assert!(log.contains("request_id=test-id"));
        assert!(log.contains("error=\"timeout\""));
        assert!(log.contains("latency_ms="));
    }
}
