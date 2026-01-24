/// Inference operation metrics
#[derive(Debug, Clone)]
pub struct InferenceMetrics {
    /// Model name being used
    pub model: String,
    /// Tokens generated
    pub tokens_generated: u64,
    /// Total duration in milliseconds
    pub duration_ms: u64,
    /// Tokens per second
    pub tokens_per_second: f64,
    /// GPU used
    pub used_gpu: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_inference_metrics() {
        let metrics = InferenceMetrics {
            model: "test".to_string(),
            tokens_generated: 100,
            duration_ms: 1000,
            tokens_per_second: 100.0,
            used_gpu: true,
        };
        assert_eq!(metrics.tokens_per_second, 100.0);
        assert_eq!(metrics.tokens_generated, 100);
    }

    #[test]
    fn test_inference_metrics_creation() {
        let metrics = InferenceMetrics {
            model: "mistral".to_string(),
            tokens_generated: 256,
            duration_ms: 2560,
            tokens_per_second: 100.0,
            used_gpu: false,
        };
        assert_eq!(metrics.model, "mistral");
        assert!(!metrics.used_gpu);
    }
}
