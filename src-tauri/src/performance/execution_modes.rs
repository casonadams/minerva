use super::adaptive::AdaptiveConfig;

/// Execution mode preference
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionMode {
    /// Prioritize accuracy and quality
    HighQuality,
    /// Balance quality and speed
    Balanced,
    /// Prioritize speed and responsiveness
    HighPerformance,
    /// Minimal resource usage (mobile/low-end)
    PowerSaver,
}

impl ExecutionMode {
    /// Get config for execution mode
    pub fn to_config(self) -> AdaptiveConfig {
        match self {
            Self::HighQuality => AdaptiveConfig {
                use_gpu: true,
                batch_size: 32,
                use_quantized: false,
                max_concurrent: 4,
                enable_prefetch: true,
            },
            Self::Balanced => AdaptiveConfig {
                use_gpu: true,
                batch_size: 16,
                use_quantized: false,
                max_concurrent: 4,
                enable_prefetch: true,
            },
            Self::HighPerformance => AdaptiveConfig {
                use_gpu: true,
                batch_size: 8,
                use_quantized: true,
                max_concurrent: 2,
                enable_prefetch: false,
            },
            Self::PowerSaver => AdaptiveConfig {
                use_gpu: false,
                batch_size: 4,
                use_quantized: true,
                max_concurrent: 1,
                enable_prefetch: false,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_execution_modes() {
        let high_quality = ExecutionMode::HighQuality.to_config();
        assert!(high_quality.use_gpu);
        assert!(!high_quality.use_quantized);

        let power_saver = ExecutionMode::PowerSaver.to_config();
        assert!(!power_saver.use_gpu);
        assert!(power_saver.use_quantized);
    }

    #[test]
    fn test_balanced_config() {
        let cfg = ExecutionMode::Balanced.to_config();
        assert!(cfg.use_gpu);
        assert_eq!(cfg.batch_size, 16);
    }

    #[test]
    fn test_high_performance_config() {
        let cfg = ExecutionMode::HighPerformance.to_config();
        assert!(cfg.use_quantized);
        assert!(!cfg.enable_prefetch);
    }
}
