use super::adaptive_config::AdaptiveConfig;

/// Handles dynamic adjustment of adaptive configuration
pub struct AdaptiveAdjuster;

impl AdaptiveAdjuster {
    /// Adjust GPU usage based on temperature/load
    pub fn adjust_gpu_usage(config: &mut AdaptiveConfig, gpu_hot: bool, cpu_busy: bool) {
        if gpu_hot {
            config.use_gpu = false;
        } else if !cpu_busy && !config.use_gpu {
            // GPU is cool and CPU is not busy, consider using GPU
            config.use_gpu = true;
        }
    }

    /// Adjust batch size based on memory pressure
    pub fn adjust_batch_size(config: &mut AdaptiveConfig, memory_percent_used: f64) {
        if memory_percent_used > 80.0 {
            config.batch_size = (config.batch_size as f64 * 0.7) as u32;
            config.batch_size = config.batch_size.max(1);
        } else if memory_percent_used < 40.0 && config.batch_size < 64 {
            config.batch_size = (config.batch_size as f64 * 1.2) as u32;
            config.batch_size = config.batch_size.min(64);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adjust_gpu_usage_when_hot() {
        let mut config = AdaptiveConfig {
            use_gpu: true,
            ..Default::default()
        };
        AdaptiveAdjuster::adjust_gpu_usage(&mut config, true, false);
        assert!(!config.use_gpu);
    }

    #[test]
    fn test_adjust_gpu_usage_when_cool() {
        let mut config = AdaptiveConfig {
            use_gpu: false,
            ..Default::default()
        };
        AdaptiveAdjuster::adjust_gpu_usage(&mut config, false, false);
        assert!(config.use_gpu);
    }

    #[test]
    fn test_adjust_batch_size_high_memory() {
        let mut config = AdaptiveConfig::default();
        let initial = config.batch_size;
        AdaptiveAdjuster::adjust_batch_size(&mut config, 85.0);
        assert!(config.batch_size < initial);
    }

    #[test]
    fn test_adjust_batch_size_low_memory() {
        let mut config = AdaptiveConfig::default();
        let initial = config.batch_size;
        AdaptiveAdjuster::adjust_batch_size(&mut config, 30.0);
        assert!(config.batch_size >= initial);
    }

    #[test]
    fn test_adjust_batch_size_min_boundary() {
        let mut config = AdaptiveConfig {
            batch_size: 2,
            ..Default::default()
        };
        AdaptiveAdjuster::adjust_batch_size(&mut config, 85.0);
        assert_eq!(config.batch_size, 1);
    }

    #[test]
    fn test_adjust_batch_size_max_boundary() {
        let mut config = AdaptiveConfig {
            batch_size: 60,
            ..Default::default()
        };
        AdaptiveAdjuster::adjust_batch_size(&mut config, 30.0);
        assert_eq!(config.batch_size, 64);
    }
}
