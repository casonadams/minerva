use super::adaptive::AdaptiveConfig;

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

impl WindowState {
    /// Adjust config for window state
    pub fn adjust_config(self, config: AdaptiveConfig) -> AdaptiveConfig {
        match self {
            Self::Foreground => config,
            Self::Background => AdaptiveConfig {
                max_concurrent: config.max_concurrent.max(1) - 1,
                enable_prefetch: false,
                ..config
            },
            Self::Minimized => AdaptiveConfig {
                use_gpu: false,
                batch_size: 1,
                max_concurrent: 1,
                enable_prefetch: false,
                ..config
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_window_state_foreground() {
        let base = AdaptiveConfig::default();
        let fg = WindowState::Foreground.adjust_config(base);
        assert_eq!(fg.max_concurrent, base.max_concurrent);
    }

    #[test]
    fn test_window_state_background() {
        let base = AdaptiveConfig::default();
        let bg = WindowState::Background.adjust_config(base);
        assert!(bg.max_concurrent <= base.max_concurrent);
        assert!(!bg.enable_prefetch);
    }

    #[test]
    fn test_window_state_minimized() {
        let base = AdaptiveConfig::default();
        let min = WindowState::Minimized.adjust_config(base);
        assert!(!min.use_gpu);
        assert_eq!(min.batch_size, 1);
    }
}
