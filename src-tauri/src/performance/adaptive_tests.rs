#[cfg(test)]
mod tests {
    use crate::performance::adaptive::AdaptiveConfigManager;
    use crate::performance::execution_modes::ExecutionMode;
    use crate::performance::window_state::WindowState;

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
