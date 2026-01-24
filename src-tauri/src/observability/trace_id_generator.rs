/// Trace ID generator
pub struct TraceIdGenerator;

impl TraceIdGenerator {
    /// Generate new trace ID (UUID format)
    pub fn generate() -> String {
        uuid::Uuid::new_v4().to_string()
    }

    /// Extract or generate trace ID from header
    pub fn from_header_or_new(header_value: Option<&str>) -> String {
        header_value
            .and_then(|v| {
                if !v.is_empty() && v.len() < 200 {
                    Some(v.to_string())
                } else {
                    None
                }
            })
            .unwrap_or_else(Self::generate)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trace_id_generation() {
        let id = TraceIdGenerator::generate();
        assert!(!id.is_empty());
        assert!(id.len() >= 30);
    }

    #[test]
    fn test_trace_id_from_header_valid() {
        let id = TraceIdGenerator::from_header_or_new(Some("custom-trace-id"));
        assert_eq!(id, "custom-trace-id");
    }

    #[test]
    fn test_trace_id_from_header_empty() {
        let id = TraceIdGenerator::from_header_or_new(Some(""));
        assert!(!id.is_empty());
    }

    #[test]
    fn test_trace_id_from_header_none() {
        let id = TraceIdGenerator::from_header_or_new(None);
        assert!(!id.is_empty());
    }

    #[test]
    fn test_trace_id_from_header_too_long() {
        let long_header = "x".repeat(300);
        let id = TraceIdGenerator::from_header_or_new(Some(&long_header));
        assert!(!id.is_empty());
        assert_ne!(id, long_header);
    }
}
