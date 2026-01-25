use super::inference_metrics::InferenceMetrics;
use super::inference_metrics_query::InferenceMetricsQuery;
use super::metrics_storage::MetricsStorage;

/// Performance metrics aggregator for server
pub struct ServerMetricsAggregator {
    storage: MetricsStorage,
}

impl ServerMetricsAggregator {
    /// Create new aggregator
    pub fn new() -> Self {
        Self {
            storage: MetricsStorage::new(),
        }
    }

    /// Record inference metrics
    pub fn record_inference(&self, metrics: InferenceMetrics) {
        self.storage.record(metrics);
    }

    /// Get average tokens per second
    pub fn avg_tokens_per_second(&self) -> f64 {
        let metrics = self.storage.get_all();
        InferenceMetricsQuery::avg_tokens_per_second(&metrics)
    }

    /// Get average inference time
    pub fn avg_inference_time_ms(&self) -> f64 {
        let metrics = self.storage.get_all();
        InferenceMetricsQuery::avg_inference_time_ms(&metrics)
    }

    /// Get GPU usage percentage (how many used GPU)
    pub fn gpu_usage_percent(&self) -> f64 {
        let metrics = self.storage.get_all();
        InferenceMetricsQuery::gpu_usage_percent(&metrics)
    }

    /// Get most recent metrics count
    pub fn recent_count(&self) -> usize {
        self.storage.count()
    }

    /// Clear metrics
    pub fn reset(&self) {
        self.storage.clear();
    }
}

impl Clone for ServerMetricsAggregator {
    fn clone(&self) -> Self {
        Self {
            storage: self.storage.clone(),
        }
    }
}

impl Default for ServerMetricsAggregator {
    fn default() -> Self {
        Self::new()
    }
}
