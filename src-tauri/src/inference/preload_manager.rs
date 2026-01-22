use super::model_cache::ModelCache;
use super::model_registry::ModelRegistry;
use crate::error::MinervaResult;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::path::PathBuf;
use std::time::Instant;

/// Preload configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(dead_code)]
pub struct PreloadConfig {
    pub enabled: bool,
    pub strategy: PreloadStrategy,
    pub batch_size: usize,
    pub delay_ms: u64,
}

impl Default for PreloadConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            strategy: PreloadStrategy::Sequential,
            batch_size: 1,
            delay_ms: 100,
        }
    }
}

/// Preload strategy
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[allow(dead_code)]
pub enum PreloadStrategy {
    /// Load one model at a time sequentially
    Sequential,
    /// Load most frequently used models first
    Frequency,
    /// Load recently accessed models first
    Recency,
    /// Load smallest models first (faster)
    Size,
}

/// Preload task in queue
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct PreloadTask {
    model_id: String,
    model_path: PathBuf,
    priority: u32,
    created_at: Instant,
}

/// Statistics for preload operations
#[derive(Debug, Clone, Default)]
#[allow(dead_code)]
pub struct PreloadStats {
    pub total_preloaded: u64,
    pub successful: u64,
    pub failed: u64,
    pub skipped: u64,
    pub total_time_ms: u128,
}

impl PreloadStats {
    /// Get success rate as percentage
    #[allow(dead_code)]
    pub fn success_rate(&self) -> f32 {
        if self.total_preloaded == 0 {
            0.0
        } else {
            (self.successful as f32 / self.total_preloaded as f32) * 100.0
        }
    }

    /// Get average time per preload
    #[allow(dead_code)]
    pub fn avg_time_ms(&self) -> f32 {
        if self.successful == 0 {
            0.0
        } else {
            self.total_time_ms as f32 / self.successful as f32
        }
    }
}

/// Manages preloading of models into cache
#[derive(Debug)]
#[allow(dead_code)]
pub struct PreloadManager {
    queue: VecDeque<PreloadTask>,
    registry: ModelRegistry,
    config: PreloadConfig,
    stats: PreloadStats,
    last_preload: Option<Instant>,
}

impl PreloadManager {
    /// Create new preload manager
    #[allow(dead_code)]
    pub fn new(registry: ModelRegistry) -> Self {
        Self {
            queue: VecDeque::new(),
            registry,
            config: PreloadConfig::default(),
            stats: PreloadStats::default(),
            last_preload: None,
        }
    }

    /// Create with custom configuration
    #[allow(dead_code)]
    pub fn with_config(registry: ModelRegistry, config: PreloadConfig) -> Self {
        Self {
            queue: VecDeque::new(),
            registry,
            config,
            stats: PreloadStats::default(),
            last_preload: None,
        }
    }

    /// Queue model for preloading
    #[allow(dead_code)]
    pub fn queue(&mut self, model_id: &str, model_path: PathBuf) -> MinervaResult<()> {
        self.registry.register(model_id, model_path.clone())?;

        let priority = self.calculate_priority(model_id);
        let task = PreloadTask {
            model_id: model_id.to_string(),
            model_path,
            priority,
            created_at: Instant::now(),
        };

        self.queue.push_back(task);
        tracing::debug!("Model queued for preload: {}", model_id);
        Ok(())
    }

    /// Process preload queue up to batch size
    #[allow(dead_code)]
    pub fn process_batch(&mut self, cache: &mut ModelCache) -> MinervaResult<usize> {
        if !self.config.enabled || self.queue.is_empty() {
            return Ok(0);
        }

        // Check rate limiting
        if let Some(last) = self.last_preload {
            if last.elapsed().as_millis() < self.config.delay_ms as u128 {
                return Ok(0);
            }
        }

        let mut processed = 0;
        for _ in 0..self.config.batch_size {
            if let Some(task) = self.queue.pop_front() {
                let start = Instant::now();
                match cache.preload(&task.model_id, task.model_path.clone()) {
                    Ok(()) => {
                        self.stats.successful += 1;
                        self.stats.total_time_ms += start.elapsed().as_millis();
                        processed += 1;
                        tracing::info!("Model preloaded: {}", task.model_id);
                    }
                    Err(e) => {
                        self.stats.failed += 1;
                        tracing::warn!("Failed to preload {}: {}", task.model_id, e);
                    }
                }
                self.stats.total_preloaded += 1;
            } else {
                break;
            }
        }

        self.last_preload = Some(Instant::now());
        Ok(processed)
    }

    /// Get number of queued tasks
    #[allow(dead_code)]
    pub fn queue_size(&self) -> usize {
        self.queue.len()
    }

    /// Clear the queue
    #[allow(dead_code)]
    pub fn clear_queue(&mut self) {
        self.queue.clear();
        tracing::info!("Preload queue cleared");
    }

    /// Get preload statistics
    #[allow(dead_code)]
    pub fn stats(&self) -> PreloadStats {
        self.stats.clone()
    }

    /// Reset statistics
    #[allow(dead_code)]
    pub fn reset_stats(&mut self) {
        self.stats = PreloadStats::default();
    }

    /// Calculate priority for a model
    fn calculate_priority(&self, model_id: &str) -> u32 {
        match self.config.strategy {
            PreloadStrategy::Frequency => self
                .registry
                .get(model_id)
                .map(|m| (m.access_count as u32).saturating_mul(10))
                .unwrap_or(0),
            PreloadStrategy::Recency => self
                .registry
                .get(model_id)
                .and_then(|m| m.age_seconds())
                .map(|age| 1000_u64.saturating_sub(age) as u32)
                .unwrap_or(500),
            PreloadStrategy::Size => self
                .registry
                .get(model_id)
                .map(|m| (5000_u64.saturating_sub(m.size_mb)) as u32)
                .unwrap_or(0),
            PreloadStrategy::Sequential => self.queue.len() as u32,
        }
    }

    /// Get configuration
    #[allow(dead_code)]
    pub fn config(&self) -> &PreloadConfig {
        &self.config
    }

    /// Update configuration
    #[allow(dead_code)]
    pub fn set_config(&mut self, config: PreloadConfig) {
        self.config = config;
    }

    /// Enable/disable preloading
    #[allow(dead_code)]
    pub fn set_enabled(&mut self, enabled: bool) {
        self.config.enabled = enabled;
    }

    /// Get queue as list (for inspection)
    #[allow(dead_code)]
    pub fn queue_list(&self) -> Vec<String> {
        self.queue.iter().map(|t| t.model_id.clone()).collect()
    }
}

impl Default for PreloadManager {
    fn default() -> Self {
        Self::new(ModelRegistry::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_preload_config_default() {
        let config = PreloadConfig::default();
        assert!(config.enabled);
        assert_eq!(config.batch_size, 1);
        assert_eq!(config.delay_ms, 100);
    }

    #[test]
    fn test_preload_stats_default() {
        let stats = PreloadStats::default();
        assert_eq!(stats.total_preloaded, 0);
        assert_eq!(stats.successful, 0);
        assert_eq!(stats.success_rate(), 0.0);
    }

    #[test]
    fn test_preload_stats_success_rate() {
        let stats = PreloadStats {
            total_preloaded: 10,
            successful: 8,
            failed: 2,
            skipped: 0,
            total_time_ms: 1000,
        };
        assert_eq!(stats.success_rate(), 80.0);
    }

    #[test]
    fn test_preload_stats_avg_time() {
        let stats = PreloadStats {
            total_preloaded: 4,
            successful: 4,
            failed: 0,
            skipped: 0,
            total_time_ms: 400,
        };
        assert_eq!(stats.avg_time_ms(), 100.0);
    }

    #[test]
    fn test_manager_creation() {
        let manager = PreloadManager::default();
        assert_eq!(manager.queue_size(), 0);
        assert!(manager.config.enabled);
    }

    #[test]
    fn test_manager_queue_size() {
        let manager = PreloadManager::new(ModelRegistry::default());
        assert_eq!(manager.queue_size(), 0);
    }

    #[test]
    fn test_manager_clear_queue() {
        let mut manager = PreloadManager::new(ModelRegistry::default());
        manager.clear_queue();
        assert_eq!(manager.queue_size(), 0);
    }

    #[test]
    fn test_manager_reset_stats() {
        let mut manager = PreloadManager::new(ModelRegistry::default());
        manager.stats.successful = 5;
        manager.reset_stats();
        assert_eq!(manager.stats.successful, 0);
    }

    #[test]
    fn test_manager_set_enabled() {
        let mut manager = PreloadManager::new(ModelRegistry::default());
        manager.set_enabled(false);
        assert!(!manager.config.enabled);
    }

    #[test]
    fn test_preload_strategy_sequential() {
        let strategy = PreloadStrategy::Sequential;
        assert!(matches!(strategy, PreloadStrategy::Sequential));
    }

    #[test]
    fn test_preload_strategy_frequency() {
        let strategy = PreloadStrategy::Frequency;
        assert!(matches!(strategy, PreloadStrategy::Frequency));
    }

    #[test]
    fn test_preload_strategy_recency() {
        let strategy = PreloadStrategy::Recency;
        assert!(matches!(strategy, PreloadStrategy::Recency));
    }

    #[test]
    fn test_preload_strategy_size() {
        let strategy = PreloadStrategy::Size;
        assert!(matches!(strategy, PreloadStrategy::Size));
    }

    #[test]
    fn test_manager_queue_list() {
        let manager = PreloadManager::new(ModelRegistry::default());
        let list = manager.queue_list();
        assert!(list.is_empty());
    }

    #[test]
    fn test_manager_with_config() {
        let config = PreloadConfig {
            enabled: false,
            strategy: PreloadStrategy::Frequency,
            batch_size: 2,
            delay_ms: 50,
        };
        let manager = PreloadManager::with_config(ModelRegistry::default(), config);
        assert!(!manager.config.enabled);
        assert_eq!(manager.config.batch_size, 2);
    }

    #[test]
    fn test_manager_set_config() {
        let mut manager = PreloadManager::new(ModelRegistry::default());
        let config = PreloadConfig {
            enabled: false,
            batch_size: 3,
            ..Default::default()
        };
        manager.set_config(config);
        assert!(!manager.config.enabled);
        assert_eq!(manager.config.batch_size, 3);
    }
}
