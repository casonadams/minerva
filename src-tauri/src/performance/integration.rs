use super::metrics_analyzer_integration::MetricsAnalysisHelper;
use parking_lot::RwLock;
/// Performance Integration with Server
///
/// Connects performance metrics to HTTP server observability:
/// - Inference metrics collection
/// - Server request tracking
/// - Performance dashboard data
use std::sync::Arc;
use std::time::Instant;

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

/// Performance metrics aggregator for server
pub struct ServerMetricsAggregator {
    inference_metrics: Arc<RwLock<Vec<InferenceMetrics>>>,
    max_stored: usize,
}

impl ServerMetricsAggregator {
    /// Create new aggregator
    pub fn new() -> Self {
        Self {
            inference_metrics: Arc::new(RwLock::new(Vec::new())),
            max_stored: 1000, // Keep last 1000 inferences
        }
    }

    /// Record inference metrics
    pub fn record_inference(&self, metrics: InferenceMetrics) {
        let mut m = self.inference_metrics.write();
        m.push(metrics);

        // Keep only the last N entries
        if m.len() > self.max_stored {
            let remove_count = m.len() - self.max_stored;
            m.drain(0..remove_count);
        }
    }

    /// Get average tokens per second
    pub fn avg_tokens_per_second(&self) -> f64 {
        let metrics = self.inference_metrics.read();
        MetricsAnalysisHelper::avg_tokens_per_second(&metrics)
    }

    /// Get average inference time
    pub fn avg_inference_time_ms(&self) -> f64 {
        let metrics = self.inference_metrics.read();
        MetricsAnalysisHelper::avg_inference_time_ms(&metrics)
    }

    /// Get GPU usage percentage (how many used GPU)
    pub fn gpu_usage_percent(&self) -> f64 {
        let metrics = self.inference_metrics.read();
        MetricsAnalysisHelper::gpu_usage_percent(&metrics)
    }

    /// Get most recent metrics count
    pub fn recent_count(&self) -> usize {
        self.inference_metrics.read().len()
    }

    /// Clear metrics
    pub fn reset(&self) {
        self.inference_metrics.write().clear();
    }
}

impl Clone for ServerMetricsAggregator {
    fn clone(&self) -> Self {
        Self {
            inference_metrics: Arc::clone(&self.inference_metrics),
            max_stored: self.max_stored,
        }
    }
}

impl Default for ServerMetricsAggregator {
    fn default() -> Self {
        Self::new()
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
    fn test_inference_metrics() {
        let metrics = InferenceMetrics {
            model: "test".to_string(),
            tokens_generated: 100,
            duration_ms: 1000,
            tokens_per_second: 100.0,
            used_gpu: true,
        };
        assert_eq!(metrics.tokens_per_second, 100.0);
    }

    #[test]
    fn test_metrics_aggregator_creation() {
        let agg = ServerMetricsAggregator::new();
        assert_eq!(agg.recent_count(), 0);
    }

    #[test]
    fn test_record_inference() {
        let agg = ServerMetricsAggregator::new();
        let metrics = InferenceMetrics {
            model: "test".to_string(),
            tokens_generated: 100,
            duration_ms: 1000,
            tokens_per_second: 100.0,
            used_gpu: true,
        };
        agg.record_inference(metrics);
        assert_eq!(agg.recent_count(), 1);
    }

    #[test]
    fn test_avg_tokens_per_second() {
        let agg = ServerMetricsAggregator::new();
        agg.record_inference(InferenceMetrics {
            model: "test".to_string(),
            tokens_generated: 100,
            duration_ms: 1000,
            tokens_per_second: 100.0,
            used_gpu: true,
        });
        agg.record_inference(InferenceMetrics {
            model: "test".to_string(),
            tokens_generated: 200,
            duration_ms: 2000,
            tokens_per_second: 100.0,
            used_gpu: false,
        });
        assert_eq!(agg.avg_tokens_per_second(), 100.0);
    }

    #[test]
    fn test_avg_inference_time() {
        let agg = ServerMetricsAggregator::new();
        agg.record_inference(InferenceMetrics {
            model: "test".to_string(),
            tokens_generated: 100,
            duration_ms: 1000,
            tokens_per_second: 100.0,
            used_gpu: true,
        });
        agg.record_inference(InferenceMetrics {
            model: "test".to_string(),
            tokens_generated: 200,
            duration_ms: 2000,
            tokens_per_second: 100.0,
            used_gpu: true,
        });
        assert_eq!(agg.avg_inference_time_ms(), 1500.0);
    }

    #[test]
    fn test_gpu_usage_percent() {
        let agg = ServerMetricsAggregator::new();
        for i in 0..10 {
            agg.record_inference(InferenceMetrics {
                model: "test".to_string(),
                tokens_generated: 100,
                duration_ms: 1000,
                tokens_per_second: 100.0,
                used_gpu: i < 7, // 7 out of 10 used GPU
            });
        }
        assert_eq!(agg.gpu_usage_percent(), 70.0);
    }

    #[test]
    fn test_reset() {
        let agg = ServerMetricsAggregator::new();
        agg.record_inference(InferenceMetrics {
            model: "test".to_string(),
            tokens_generated: 100,
            duration_ms: 1000,
            tokens_per_second: 100.0,
            used_gpu: true,
        });
        assert_eq!(agg.recent_count(), 1);

        agg.reset();
        assert_eq!(agg.recent_count(), 0);
    }

    #[test]
    fn test_cloneable() {
        let agg1 = ServerMetricsAggregator::new();
        let agg2 = agg1.clone();

        agg1.record_inference(InferenceMetrics {
            model: "test".to_string(),
            tokens_generated: 100,
            duration_ms: 1000,
            tokens_per_second: 100.0,
            used_gpu: true,
        });

        assert_eq!(agg2.recent_count(), 1);
    }
}
