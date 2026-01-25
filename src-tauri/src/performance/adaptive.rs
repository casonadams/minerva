pub use super::adaptive_config::AdaptiveConfig;

use super::adaptive_adjuster::AdaptiveAdjuster;
use super::execution_modes::ExecutionMode;
use super::window_state::WindowState;
use parking_lot::RwLock;
use std::sync::Arc;

/// Adaptive configuration manager
///
/// Adjusts inference settings based on system state and user preferences:
/// - CPU vs GPU selection based on load
/// - Batch size optimization
/// - Model precision selection (FP32 vs INT8 quantization)
/// - Background/Foreground optimization
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
        AdaptiveAdjuster::adjust_gpu_usage(&mut config, gpu_hot, cpu_busy);
    }

    /// Adjust batch size based on memory pressure
    pub fn adjust_batch_size(&self, memory_percent_used: f64) {
        let mut config = self.current.write();
        AdaptiveAdjuster::adjust_batch_size(&mut config, memory_percent_used);
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
