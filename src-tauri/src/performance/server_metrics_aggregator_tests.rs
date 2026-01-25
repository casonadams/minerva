#[cfg(test)]
mod tests {
    use crate::performance::inference_metrics::InferenceMetrics;
    use crate::performance::server_metrics_aggregator::ServerMetricsAggregator;

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
                used_gpu: i < 7,
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
