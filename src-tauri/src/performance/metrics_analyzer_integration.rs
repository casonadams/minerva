use super::integration::InferenceMetrics;

/// Analyzes aggregated inference metrics
pub struct MetricsAnalysisHelper;

impl MetricsAnalysisHelper {
    /// Get average tokens per second from metrics
    pub fn avg_tokens_per_second(metrics: &[InferenceMetrics]) -> f64 {
        if metrics.is_empty() {
            return 0.0;
        }

        let sum: f64 = metrics.iter().map(|m| m.tokens_per_second).sum();
        sum / metrics.len() as f64
    }

    /// Get average inference time in milliseconds
    pub fn avg_inference_time_ms(metrics: &[InferenceMetrics]) -> f64 {
        if metrics.is_empty() {
            return 0.0;
        }

        let sum: u64 = metrics.iter().map(|m| m.duration_ms).sum();
        sum as f64 / metrics.len() as f64
    }

    /// Get GPU usage percentage
    pub fn gpu_usage_percent(metrics: &[InferenceMetrics]) -> f64 {
        if metrics.is_empty() {
            return 0.0;
        }

        let gpu_count = metrics.iter().filter(|m| m.used_gpu).count();
        (gpu_count as f64 / metrics.len() as f64) * 100.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_avg_tokens_per_second() {
        let metrics = vec![
            InferenceMetrics {
                model: "test".to_string(),
                tokens_generated: 100,
                duration_ms: 1000,
                tokens_per_second: 100.0,
                used_gpu: true,
            },
            InferenceMetrics {
                model: "test".to_string(),
                tokens_generated: 200,
                duration_ms: 2000,
                tokens_per_second: 100.0,
                used_gpu: false,
            },
        ];
        assert_eq!(
            MetricsAnalysisHelper::avg_tokens_per_second(&metrics),
            100.0
        );
    }

    #[test]
    fn test_avg_inference_time() {
        let metrics = vec![
            InferenceMetrics {
                model: "test".to_string(),
                tokens_generated: 100,
                duration_ms: 1000,
                tokens_per_second: 100.0,
                used_gpu: true,
            },
            InferenceMetrics {
                model: "test".to_string(),
                tokens_generated: 200,
                duration_ms: 2000,
                tokens_per_second: 100.0,
                used_gpu: true,
            },
        ];
        assert_eq!(
            MetricsAnalysisHelper::avg_inference_time_ms(&metrics),
            1500.0
        );
    }

    #[test]
    fn test_gpu_usage_percent() {
        let metrics: Vec<InferenceMetrics> = (0..10)
            .map(|i| InferenceMetrics {
                model: "test".to_string(),
                tokens_generated: 100,
                duration_ms: 1000,
                tokens_per_second: 100.0,
                used_gpu: i < 7,
            })
            .collect();
        assert_eq!(MetricsAnalysisHelper::gpu_usage_percent(&metrics), 70.0);
    }

    #[test]
    fn test_empty_metrics() {
        let metrics = vec![];
        assert_eq!(MetricsAnalysisHelper::avg_tokens_per_second(&metrics), 0.0);
        assert_eq!(MetricsAnalysisHelper::avg_inference_time_ms(&metrics), 0.0);
        assert_eq!(MetricsAnalysisHelper::gpu_usage_percent(&metrics), 0.0);
    }
}
