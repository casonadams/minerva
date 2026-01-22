/// Performance metrics for inference operations
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct InferenceMetrics {
    pub request_id: String,
    pub model_name: String,
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
    pub total_time_ms: u128,
    pub model_load_time_ms: u128,
    pub generation_time_ms: u128,
}

impl InferenceMetrics {
    /// Create new metrics
    #[allow(dead_code)]
    pub fn new(
        request_id: String,
        model_name: String,
        prompt_tokens: usize,
        completion_tokens: usize,
        model_load_time_ms: u128,
    ) -> Self {
        let total_tokens = prompt_tokens + completion_tokens;

        Self {
            request_id,
            model_name,
            prompt_tokens,
            completion_tokens,
            total_tokens,
            total_time_ms: 0,
            model_load_time_ms,
            generation_time_ms: 0,
        }
    }

    /// Calculate tokens per second (generation speed)
    #[allow(dead_code)]
    pub fn tokens_per_second(&self) -> f64 {
        if self.generation_time_ms == 0 {
            0.0
        } else {
            self.completion_tokens as f64 / (self.generation_time_ms as f64 / 1000.0)
        }
    }

    /// Calculate prompt tokens per second (prompt processing speed)
    #[allow(dead_code)]
    pub fn prompt_tokens_per_second(&self) -> f64 {
        // For now, use total generation time as approximation
        if self.generation_time_ms == 0 {
            0.0
        } else {
            self.prompt_tokens as f64 / (self.generation_time_ms as f64 / 1000.0)
        }
    }

    /// Get summary string for logging
    #[allow(dead_code)]
    pub fn summary(&self) -> String {
        format!(
            "Inference: {} tokens/sec, {} total tokens, {}ms generation, {}ms model load",
            self.tokens_per_second() as u32,
            self.total_tokens,
            self.generation_time_ms,
            self.model_load_time_ms,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metrics_creation() {
        let metrics =
            InferenceMetrics::new("req-1".to_string(), "test-model".to_string(), 10, 20, 100);

        assert_eq!(metrics.request_id, "req-1");
        assert_eq!(metrics.model_name, "test-model");
        assert_eq!(metrics.prompt_tokens, 10);
        assert_eq!(metrics.completion_tokens, 20);
        assert_eq!(metrics.total_tokens, 30);
    }

    #[test]
    fn test_tokens_per_second_calculation() {
        let mut metrics =
            InferenceMetrics::new("req-1".to_string(), "test-model".to_string(), 10, 100, 100);
        metrics.generation_time_ms = 1000;

        let tps = metrics.tokens_per_second();
        assert!((tps - 100.0).abs() < 0.1);
    }

    #[test]
    fn test_tokens_per_second_zero_time() {
        let metrics =
            InferenceMetrics::new("req-1".to_string(), "test-model".to_string(), 10, 100, 100);
        assert_eq!(metrics.tokens_per_second(), 0.0);
    }

    #[test]
    fn test_metrics_summary() {
        let mut metrics =
            InferenceMetrics::new("req-1".to_string(), "test-model".to_string(), 10, 100, 50);
        metrics.generation_time_ms = 500;

        let summary = metrics.summary();
        assert!(summary.contains("tokens/sec"));
        assert!(summary.contains("total tokens"));
    }
}
