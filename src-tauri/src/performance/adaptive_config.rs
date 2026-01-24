use super::execution_modes::ExecutionMode;
use super::window_state::WindowState;

/// Adaptive performance configuration
#[derive(Debug, Clone, Copy)]
pub struct AdaptiveConfig {
    /// Use GPU for inference
    pub use_gpu: bool,
    /// Batch size for inference
    pub batch_size: u32,
    /// Use quantized models (INT8 instead of FP32)
    pub use_quantized: bool,
    /// Maximum concurrent inferences
    pub max_concurrent: u32,
    /// Enable prefetching of next batch
    pub enable_prefetch: bool,
}

impl Default for AdaptiveConfig {
    fn default() -> Self {
        Self {
            use_gpu: true,
            batch_size: 32,
            use_quantized: false,
            max_concurrent: 4,
            enable_prefetch: true,
        }
    }
}

impl AdaptiveConfig {
    /// Get config for execution mode
    pub fn for_mode(mode: ExecutionMode) -> Self {
        mode.to_config()
    }

    /// Adjust for window state
    pub fn for_window_state(self, state: WindowState) -> Self {
        state.adjust_config(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adaptive_config_default() {
        let config = AdaptiveConfig::default();
        assert!(config.use_gpu);
        assert_eq!(config.batch_size, 32);
        assert!(!config.use_quantized);
    }

    #[test]
    fn test_execution_modes() {
        let high_quality = AdaptiveConfig::for_mode(ExecutionMode::HighQuality);
        assert!(high_quality.use_gpu);
        assert!(!high_quality.use_quantized);

        let power_saver = AdaptiveConfig::for_mode(ExecutionMode::PowerSaver);
        assert!(!power_saver.use_gpu);
        assert!(power_saver.use_quantized);
    }

    #[test]
    fn test_config_for_window_state() {
        let base = AdaptiveConfig::default();

        let fg = base.for_window_state(WindowState::Foreground);
        assert_eq!(fg.max_concurrent, base.max_concurrent);

        let bg = base.for_window_state(WindowState::Background);
        assert!(bg.max_concurrent <= base.max_concurrent);

        let min = base.for_window_state(WindowState::Minimized);
        assert!(!min.use_gpu);
        assert_eq!(min.batch_size, 1);
    }
}
