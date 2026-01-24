use std::time::Instant;

/// Server operation context for tracking
#[derive(Debug, Clone)]
pub struct OperationContext {
    /// Operation start time
    pub start_time: Instant,
    /// Operation name
    pub operation: String,
    /// Model being used
    pub model: Option<String>,
}

impl OperationContext {
    /// Create new operation context
    pub fn new(operation: &str) -> Self {
        Self {
            start_time: Instant::now(),
            operation: operation.to_string(),
            model: None,
        }
    }

    /// Set model name
    pub fn with_model(mut self, model: &str) -> Self {
        self.model = Some(model.to_string());
        self
    }

    /// Get elapsed time in milliseconds
    pub fn elapsed_ms(&self) -> u64 {
        self.start_time.elapsed().as_millis() as u64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_operation_context_creation() {
        let ctx = OperationContext::new("inference");
        assert_eq!(ctx.operation, "inference");
        assert!(ctx.model.is_none());
    }

    #[test]
    fn test_operation_context_with_model() {
        let ctx = OperationContext::new("inference").with_model("mistral");
        assert_eq!(ctx.model, Some("mistral".to_string()));
    }

    #[test]
    fn test_operation_context_elapsed() {
        let ctx = OperationContext::new("test");
        std::thread::sleep(std::time::Duration::from_millis(10));
        assert!(ctx.elapsed_ms() >= 10);
    }

    #[test]
    fn test_operation_context_builder_chain() {
        let ctx = OperationContext::new("generate").with_model("llama");
        assert_eq!(ctx.operation, "generate");
        assert_eq!(ctx.model, Some("llama".to_string()));
    }
}
