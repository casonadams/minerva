use parking_lot::RwLock;
use std::sync::Arc;

use super::integration::InferenceMetrics;

/// Manages storage of inference metrics with bounded capacity
pub struct MetricsStorage {
    metrics: Arc<RwLock<Vec<InferenceMetrics>>>,
    max_stored: usize,
}

impl MetricsStorage {
    /// Create new metrics storage with default capacity
    pub fn new() -> Self {
        Self {
            metrics: Arc::new(RwLock::new(Vec::new())),
            max_stored: 1000,
        }
    }

    /// Record an inference metric
    pub fn record(&self, metric: InferenceMetrics) {
        let mut m = self.metrics.write();
        m.push(metric);

        // Keep only the last N entries
        if m.len() > self.max_stored {
            let remove_count = m.len() - self.max_stored;
            m.drain(0..remove_count);
        }
    }

    /// Get all metrics
    pub fn get_all(&self) -> Vec<InferenceMetrics> {
        self.metrics.read().clone()
    }

    /// Get count of stored metrics
    pub fn count(&self) -> usize {
        self.metrics.read().len()
    }

    /// Clear all metrics
    pub fn clear(&self) {
        self.metrics.write().clear();
    }
}

impl Clone for MetricsStorage {
    fn clone(&self) -> Self {
        Self {
            metrics: Arc::clone(&self.metrics),
            max_stored: self.max_stored,
        }
    }
}

impl Default for MetricsStorage {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_record_metric() {
        let storage = MetricsStorage::new();
        storage.record(InferenceMetrics {
            model: "test".to_string(),
            tokens_generated: 100,
            duration_ms: 1000,
            tokens_per_second: 100.0,
            used_gpu: true,
        });
        assert_eq!(storage.count(), 1);
    }

    #[test]
    fn test_get_all_metrics() {
        let storage = MetricsStorage::new();
        storage.record(InferenceMetrics {
            model: "test".to_string(),
            tokens_generated: 100,
            duration_ms: 1000,
            tokens_per_second: 100.0,
            used_gpu: true,
        });
        let metrics = storage.get_all();
        assert_eq!(metrics.len(), 1);
    }

    #[test]
    fn test_clear_metrics() {
        let storage = MetricsStorage::new();
        storage.record(InferenceMetrics {
            model: "test".to_string(),
            tokens_generated: 100,
            duration_ms: 1000,
            tokens_per_second: 100.0,
            used_gpu: true,
        });
        assert_eq!(storage.count(), 1);
        storage.clear();
        assert_eq!(storage.count(), 0);
    }

    #[test]
    fn test_multiple_records() {
        let storage = MetricsStorage::new();
        for i in 0..5 {
            storage.record(InferenceMetrics {
                model: format!("model_{}", i),
                tokens_generated: 100,
                duration_ms: 1000,
                tokens_per_second: 100.0,
                used_gpu: true,
            });
        }
        assert_eq!(storage.count(), 5);
    }

    #[test]
    fn test_clone_shares_state() {
        let storage1 = MetricsStorage::new();
        let storage2 = storage1.clone();
        storage1.record(InferenceMetrics {
            model: "test".to_string(),
            tokens_generated: 100,
            duration_ms: 1000,
            tokens_per_second: 100.0,
            used_gpu: true,
        });
        assert_eq!(storage2.count(), 1);
    }
}
