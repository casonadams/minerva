use parking_lot::RwLock;
/// Adaptive Performance Configuration
///
/// Adjusts inference settings based on system state and user preferences:
/// - CPU vs GPU selection based on load
/// - Batch size optimization
/// - Model precision selection (FP32 vs INT8 quantization)
/// - Background/Foreground optimization
use std::sync::Arc;

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

/// Window focus state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WindowState {
    /// Window is in foreground, user actively using app
    Foreground,
    /// Window is in background
    Background,
    /// Window is minimized
    Minimized,
}

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
        match mode {
            ExecutionMode::HighQuality => Self {
                use_gpu: true,
                batch_size: 32,
                use_quantized: false,
                max_concurrent: 4,
                enable_prefetch: true,
            },
            ExecutionMode::Balanced => Self {
                use_gpu: true,
                batch_size: 16,
                use_quantized: false,
                max_concurrent: 4,
                enable_prefetch: true,
            },
            ExecutionMode::HighPerformance => Self {
                use_gpu: true,
                batch_size: 8,
                use_quantized: true,
                max_concurrent: 2,
                enable_prefetch: false,
            },
            ExecutionMode::PowerSaver => Self {
                use_gpu: false,
                batch_size: 4,
                use_quantized: true,
                max_concurrent: 1,
                enable_prefetch: false,
            },
        }
    }

    /// Adjust for window state
    pub fn for_window_state(self, state: WindowState) -> Self {
        match state {
            WindowState::Foreground => self,
            WindowState::Background => Self {
                max_concurrent: self.max_concurrent.max(1) - 1,
                enable_prefetch: false,
                ..self
            },
            WindowState::Minimized => Self {
                use_gpu: false,
                batch_size: 1,
                max_concurrent: 1,
                enable_prefetch: false,
                ..self
            },
        }
    }
}

/// Adaptive configuration manager
pub struct AdaptiveConfigManager {
    current: Arc<RwLock<AdaptiveConfig>>,
    mode: Arc<RwLock<ExecutionMode>>,
    window_state: Arc<RwLock<WindowState>>,
}

impl AdaptiveConfigManager {
    /// Create new manager
    pub fn new() -> Self {
        Self {
            current: Arc::new(RwLock::new(AdaptiveConfig::default())),
            mode: Arc::new(RwLock::new(ExecutionMode::Balanced)),
            window_state: Arc::new(RwLock::new(WindowState::Foreground)),
        }
    }

    /// Set execution mode
    pub fn set_mode(&self, mode: ExecutionMode) {
        let window = *self.window_state.read();
        let config = AdaptiveConfig::for_mode(mode).for_window_state(window);
        *self.current.write() = config;
        *self.mode.write() = mode;
    }

    /// Set window state
    pub fn set_window_state(&self, state: WindowState) {
        let mode = *self.mode.read();
        let base = AdaptiveConfig::for_mode(mode);
        let config = base.for_window_state(state);
        *self.current.write() = config;
        *self.window_state.write() = state;
    }

    /// Get current configuration
    pub fn snapshot(&self) -> AdaptiveConfig {
        *self.current.read()
    }

    /// Get current execution mode
    pub fn mode(&self) -> ExecutionMode {
        *self.mode.read()
    }

    /// Get current window state
    pub fn window_state(&self) -> WindowState {
        *self.window_state.read()
    }

    /// Adjust GPU usage based on temperature/load
    pub fn adjust_gpu_usage(&self, gpu_hot: bool, cpu_busy: bool) {
        let mut config = self.current.write();
        if gpu_hot {
            config.use_gpu = false;
        } else if !cpu_busy && !config.use_gpu {
            // GPU is cool and CPU is not busy, consider using GPU
            config.use_gpu = true;
        }
    }

    /// Adjust batch size based on memory pressure
    pub fn adjust_batch_size(&self, memory_percent_used: f64) {
        let mut config = self.current.write();
        if memory_percent_used > 80.0 {
            config.batch_size = (config.batch_size as f64 * 0.7) as u32;
            config.batch_size = config.batch_size.max(1);
        } else if memory_percent_used < 40.0 && config.batch_size < 64 {
            config.batch_size = (config.batch_size as f64 * 1.2) as u32;
            config.batch_size = config.batch_size.min(64);
        }
    }
}

impl Clone for AdaptiveConfigManager {
    fn clone(&self) -> Self {
        Self {
            current: Arc::clone(&self.current),
            mode: Arc::clone(&self.mode),
            window_state: Arc::clone(&self.window_state),
        }
    }
}

impl Default for AdaptiveConfigManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

    #[test]
    fn test_manager_creation() {
        let mgr = AdaptiveConfigManager::new();
        let cfg = mgr.snapshot();
        assert!(cfg.use_gpu);
        assert_eq!(mgr.mode(), ExecutionMode::Balanced);
    }

    #[test]
    fn test_set_mode() {
        let mgr = AdaptiveConfigManager::new();
        mgr.set_mode(ExecutionMode::PowerSaver);
        let cfg = mgr.snapshot();
        assert!(!cfg.use_gpu);
        assert_eq!(mgr.mode(), ExecutionMode::PowerSaver);
    }

    #[test]
    fn test_set_window_state() {
        let mgr = AdaptiveConfigManager::new();
        mgr.set_window_state(WindowState::Minimized);
        let cfg = mgr.snapshot();
        assert!(!cfg.use_gpu);
        assert_eq!(mgr.window_state(), WindowState::Minimized);
    }

    #[test]
    fn test_adjust_gpu_usage() {
        let mgr = AdaptiveConfigManager::new();
        mgr.adjust_gpu_usage(true, false); // GPU hot
        let cfg = mgr.snapshot();
        assert!(!cfg.use_gpu);
    }

    #[test]
    fn test_adjust_batch_size_high_memory() {
        let mgr = AdaptiveConfigManager::new();
        let initial = mgr.snapshot().batch_size;
        mgr.adjust_batch_size(85.0);
        let adjusted = mgr.snapshot().batch_size;
        assert!(adjusted < initial);
    }

    #[test]
    fn test_adjust_batch_size_low_memory() {
        let mgr = AdaptiveConfigManager::new();
        let initial = mgr.snapshot().batch_size;
        mgr.adjust_batch_size(30.0);
        let adjusted = mgr.snapshot().batch_size;
        assert!(adjusted >= initial);
    }

    #[test]
    fn test_cloneable() {
        let mgr1 = AdaptiveConfigManager::new();
        let mgr2 = mgr1.clone();
        mgr1.set_mode(ExecutionMode::PowerSaver);
        assert_eq!(mgr2.mode(), ExecutionMode::PowerSaver);
    }
}
