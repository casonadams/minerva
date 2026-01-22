use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};

/// Garbage collection policy
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default)]
#[allow(dead_code)]
pub enum GCPolicy {
    /// Mark and sweep collection
    #[default]
    MarkAndSweep,
    /// Generational collection
    Generational,
    /// Reference counting
    ReferenceCount,
}

/// Garbage collection statistics
#[derive(Debug, Clone, Default)]
#[allow(dead_code)]
pub struct GCStats {
    pub total_collections: u64,
    pub total_freed_mb: u64,
    pub last_collection: Option<Instant>,
    pub avg_collection_time_ms: u128,
    pub models_collected: u64,
}

impl GCStats {
    /// Get collection frequency (collections per hour)
    #[allow(dead_code)]
    pub fn collection_frequency(&self) -> f32 {
        if self.last_collection.is_none() {
            0.0
        } else {
            (self.total_collections as f32 / 3600.0).max(0.0)
        }
    }

    /// Get average freed MB per collection
    #[allow(dead_code)]
    pub fn avg_freed_per_collection(&self) -> f32 {
        if self.total_collections == 0 {
            0.0
        } else {
            self.total_freed_mb as f32 / self.total_collections as f32
        }
    }
}

/// Garbage collection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(dead_code)]
pub struct GCConfig {
    pub policy: GCPolicy,
    pub collection_interval_ms: u64,
    pub min_free_mb: u64,
    pub auto_collect: bool,
    pub aggressive_mode: bool,
}

impl Default for GCConfig {
    fn default() -> Self {
        Self {
            policy: GCPolicy::MarkAndSweep,
            collection_interval_ms: 60000, // 1 minute
            min_free_mb: 500,
            auto_collect: true,
            aggressive_mode: false,
        }
    }
}

/// Garbage collector for cache management
#[derive(Debug)]
#[allow(dead_code)]
pub struct GarbageCollector {
    config: GCConfig,
    stats: GCStats,
    next_collection: Instant,
}

impl GarbageCollector {
    /// Create new garbage collector
    #[allow(dead_code)]
    pub fn new() -> Self {
        Self {
            config: GCConfig::default(),
            stats: GCStats::default(),
            next_collection: Instant::now(),
        }
    }

    /// Create with custom configuration
    #[allow(dead_code)]
    pub fn with_config(config: GCConfig) -> Self {
        Self {
            config,
            stats: GCStats::default(),
            next_collection: Instant::now(),
        }
    }

    /// Check if collection is needed
    #[allow(dead_code)]
    pub fn should_collect(&self) -> bool {
        if !self.config.auto_collect {
            return false;
        }

        self.next_collection.elapsed().as_millis() >= self.config.collection_interval_ms as u128
    }

    /// Perform garbage collection
    #[allow(dead_code)]
    pub fn collect(&mut self, freed_mb: u64, models_collected: u64) {
        let start = Instant::now();

        self.stats.total_collections += 1;
        self.stats.total_freed_mb += freed_mb;
        self.stats.models_collected += models_collected;
        self.stats.last_collection = Some(Instant::now());

        let collection_time = start.elapsed().as_millis();
        self.stats.avg_collection_time_ms =
            (self.stats.avg_collection_time_ms + collection_time) / 2;

        self.next_collection =
            Instant::now() + Duration::from_millis(self.config.collection_interval_ms);

        tracing::info!(
            "Garbage collection complete: {} MB freed, {} models collected, {} ms elapsed",
            freed_mb,
            models_collected,
            collection_time
        );
    }

    /// Get time until next collection
    #[allow(dead_code)]
    pub fn time_until_next_collection(&self) -> Duration {
        self.next_collection
            .duration_since(Instant::now())
            .max(Duration::from_secs(0))
    }

    /// Get statistics
    #[allow(dead_code)]
    pub fn stats(&self) -> GCStats {
        self.stats.clone()
    }

    /// Get configuration
    #[allow(dead_code)]
    pub fn config(&self) -> &GCConfig {
        &self.config
    }

    /// Update configuration
    #[allow(dead_code)]
    pub fn set_config(&mut self, config: GCConfig) {
        self.config = config;
    }

    /// Enable/disable auto collection
    #[allow(dead_code)]
    pub fn set_auto_collect(&mut self, enabled: bool) {
        self.config.auto_collect = enabled;
    }

    /// Set collection policy
    #[allow(dead_code)]
    pub fn set_policy(&mut self, policy: GCPolicy) {
        self.config.policy = policy;
    }

    /// Reset statistics
    #[allow(dead_code)]
    pub fn reset_stats(&mut self) {
        self.stats = GCStats::default();
    }
}

impl Default for GarbageCollector {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gc_stats_default() {
        let stats = GCStats::default();
        assert_eq!(stats.total_collections, 0);
        assert_eq!(stats.total_freed_mb, 0);
    }

    #[test]
    fn test_gc_stats_collection_frequency() {
        let stats = GCStats::default();
        assert_eq!(stats.collection_frequency(), 0.0);
    }

    #[test]
    fn test_gc_stats_avg_freed() {
        let stats = GCStats {
            total_collections: 10,
            total_freed_mb: 500,
            ..Default::default()
        };
        assert_eq!(stats.avg_freed_per_collection(), 50.0);
    }

    #[test]
    fn test_gc_config_default() {
        let config = GCConfig::default();
        assert!(config.auto_collect);
        assert_eq!(config.collection_interval_ms, 60000);
    }

    #[test]
    fn test_gc_policy_default() {
        let policy = GCPolicy::default();
        assert!(matches!(policy, GCPolicy::MarkAndSweep));
    }

    #[test]
    fn test_garbage_collector_creation() {
        let collector = GarbageCollector::new();
        assert!(collector.config.auto_collect);
    }

    #[test]
    fn test_garbage_collector_with_config() {
        let config = GCConfig {
            auto_collect: false,
            ..Default::default()
        };
        let collector = GarbageCollector::with_config(config);
        assert!(!collector.config.auto_collect);
    }

    #[test]
    fn test_garbage_collector_should_collect() {
        let mut collector = GarbageCollector::new();
        collector.config.collection_interval_ms = 0;
        assert!(collector.should_collect());
    }

    #[test]
    fn test_garbage_collector_collect() {
        let mut collector = GarbageCollector::new();
        collector.collect(100, 5);
        assert_eq!(collector.stats.total_collections, 1);
        assert_eq!(collector.stats.total_freed_mb, 100);
        assert_eq!(collector.stats.models_collected, 5);
    }

    #[test]
    fn test_garbage_collector_stats() {
        let collector = GarbageCollector::new();
        let stats = collector.stats();
        assert_eq!(stats.total_collections, 0);
    }

    #[test]
    fn test_garbage_collector_set_auto_collect() {
        let mut collector = GarbageCollector::new();
        collector.set_auto_collect(false);
        assert!(!collector.config.auto_collect);
    }

    #[test]
    fn test_garbage_collector_set_policy() {
        let mut collector = GarbageCollector::new();
        collector.set_policy(GCPolicy::Generational);
        assert!(matches!(collector.config.policy, GCPolicy::Generational));
    }

    #[test]
    fn test_garbage_collector_reset_stats() {
        let mut collector = GarbageCollector::new();
        collector.collect(100, 5);
        collector.reset_stats();
        assert_eq!(collector.stats.total_collections, 0);
    }

    #[test]
    fn test_gc_policy_all_variants() {
        let policies = [
            GCPolicy::MarkAndSweep,
            GCPolicy::Generational,
            GCPolicy::ReferenceCount,
        ];

        for policy in policies {
            assert!(matches!(
                policy,
                GCPolicy::MarkAndSweep | GCPolicy::Generational | GCPolicy::ReferenceCount
            ));
        }
    }

    #[test]
    fn test_gc_time_until_next_collection() {
        let collector = GarbageCollector::new();
        let time = collector.time_until_next_collection();
        assert!(time.as_millis() < 100);
    }
}
