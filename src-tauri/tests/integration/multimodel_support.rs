// Multi-Model Support (Phase 4 Steps 1-2) Integration Tests

use minerva_lib::inference::context_manager::ContextManager;
use minerva_lib::inference::model_cache::{EvictionPolicy, ModelCache};
use minerva_lib::inference::model_registry::ModelRegistry;
use minerva_lib::inference::preload_manager::PreloadManager;

// Phase 4 Step 1: Multi-Model Support

#[test]
fn test_context_manager_cache_stats() {
    let manager = ContextManager::new(2);
    let stats = manager.cache_stats();

    assert_eq!(stats.hits, 0);
    assert_eq!(stats.misses, 0);
    assert_eq!(stats.evictions, 0);
    assert_eq!(stats.preloads, 0);
}

#[test]
fn test_context_manager_memory_tracking() {
    let mut manager = ContextManager::new(2);
    assert_eq!(manager.estimated_memory_mb(), 0);

    manager.update_memory_estimate();
    assert_eq!(manager.estimated_memory_mb(), 0);
}

#[test]
fn test_context_manager_memory_pressure() {
    let manager = ContextManager::new(3);
    assert!(!manager.has_memory_pressure());
}

#[test]
fn test_model_cache_creation_lru() {
    let cache = ModelCache::new(3, EvictionPolicy::Lru);
    assert_eq!(cache.size(), 0);
    assert_eq!(cache.capacity(), 3);
    assert!(cache.list().is_empty());
}

#[test]
fn test_model_cache_creation_lfu() {
    let cache = ModelCache::new(2, EvictionPolicy::Lfu);
    assert_eq!(cache.capacity(), 2);
}

#[test]
fn test_model_cache_creation_fifo() {
    let cache = ModelCache::new(4, EvictionPolicy::Fifo);
    assert_eq!(cache.capacity(), 4);
}

#[test]
fn test_model_cache_default() {
    let cache = ModelCache::default();
    assert_eq!(cache.capacity(), 3);
    assert_eq!(cache.size(), 0);
}

#[test]
fn test_model_cache_contains() {
    let cache = ModelCache::new(2, EvictionPolicy::Lru);
    assert!(!cache.contains("test"));
}

#[test]
fn test_model_cache_stats_hit_rate() {
    use minerva_lib::inference::model_cache::CacheStats;

    let stats = CacheStats {
        hits: 75,
        misses: 25,
        ..Default::default()
    };
    assert_eq!(stats.hit_rate(), 75.0);
}

#[test]
fn test_context_manager_with_policy() {
    let manager = ContextManager::with_policy(4, EvictionPolicy::Lfu);
    assert_eq!(manager.max_models(), 4);
    assert_eq!(manager.loaded_count(), 0);
}

#[test]
fn test_model_cache_get_nonexistent() {
    let mut cache = ModelCache::new(2, EvictionPolicy::Lru);
    let result = cache.get_mut("nonexistent");
    assert!(result.is_err());
}

#[test]
fn test_model_cache_list_operations() {
    let cache = ModelCache::new(3, EvictionPolicy::Lru);
    let list = cache.list();
    assert!(list.is_empty());
    assert_eq!(list.len(), 0);
}

// Phase 4 Step 2: Model Caching & Preloading

#[test]
fn test_model_registry_creation() {
    let registry = ModelRegistry::new();
    assert!(registry.list().is_empty());
    assert_eq!(registry.cached_size_mb(), 0);
}

#[test]
fn test_model_registry_default() {
    let registry = ModelRegistry::default();
    assert_eq!(registry.list().len(), 0);
    assert_eq!(registry.list_cached().len(), 0);
}

#[test]
fn test_model_registry_cache_usage() {
    let registry = ModelRegistry::new();
    assert_eq!(registry.cache_usage_percent(), 0.0);
}

#[test]
fn test_model_registry_max_cache_size() {
    let mut registry = ModelRegistry::new();
    registry.set_max_cache_size(10000);

    assert!(!registry.would_exceed_limit(5000));
    assert!(registry.would_exceed_limit(10001));
    assert!(!registry.would_exceed_limit(9999));
}

#[test]
fn test_model_registry_oldest_cached() {
    let registry = ModelRegistry::new();
    assert!(registry.oldest_cached().is_empty());
}

#[test]
fn test_model_registry_least_used() {
    let registry = ModelRegistry::new();
    assert!(registry.least_used_cached().is_empty());
}

#[test]
fn test_model_registry_remove() {
    let mut registry = ModelRegistry::new();
    assert!(registry.remove("nonexistent").is_none());
}

#[test]
fn test_model_registry_clear() {
    let mut registry = ModelRegistry::new();
    registry.clear();
    assert!(registry.list().is_empty());
}

#[test]
fn test_preload_manager_creation() {
    let manager = PreloadManager::default();
    assert_eq!(manager.queue_size(), 0);
}

#[test]
fn test_preload_manager_queue() {
    let manager = PreloadManager::default();
    assert!(manager.queue_list().is_empty());
}

#[test]
fn test_preload_manager_config() {
    let manager =
        PreloadManager::new(minerva_lib::inference::model_registry::ModelRegistry::default());
    let config = manager.config();
    assert!(config.enabled);
}

#[test]
fn test_preload_manager_clear_queue() {
    let mut manager = PreloadManager::default();
    manager.clear_queue();
    assert_eq!(manager.queue_size(), 0);
}

#[test]
fn test_preload_manager_stats() {
    let manager = PreloadManager::default();
    let stats = manager.stats();
    assert_eq!(stats.total_preloaded, 0);
    assert_eq!(stats.success_rate(), 0.0);
}

#[test]
fn test_preload_manager_reset_stats() {
    let mut manager = PreloadManager::default();
    manager.reset_stats();
    assert_eq!(manager.stats().total_preloaded, 0);
}

#[test]
fn test_preload_config_default() {
    use minerva_lib::inference::preload_manager::PreloadConfig;

    let config = PreloadConfig::default();
    assert!(config.enabled);
    assert_eq!(config.batch_size, 1);
}

#[test]
fn test_preload_strategy_sequential() {
    use minerva_lib::inference::preload_manager::PreloadStrategy;

    let strategy = PreloadStrategy::Sequential;
    assert!(matches!(strategy, PreloadStrategy::Sequential));
}

#[test]
fn test_preload_strategy_frequency() {
    use minerva_lib::inference::preload_manager::PreloadStrategy;

    let strategy = PreloadStrategy::Frequency;
    assert!(matches!(strategy, PreloadStrategy::Frequency));
}

#[test]
fn test_preload_strategy_recency() {
    use minerva_lib::inference::preload_manager::PreloadStrategy;

    let strategy = PreloadStrategy::Recency;
    assert!(matches!(strategy, PreloadStrategy::Recency));
}

#[test]
fn test_preload_strategy_size() {
    use minerva_lib::inference::preload_manager::PreloadStrategy;

    let strategy = PreloadStrategy::Size;
    assert!(matches!(strategy, PreloadStrategy::Size));
}

#[test]
fn test_preload_stats_calculation() {
    use minerva_lib::inference::preload_manager::PreloadStats;

    let stats = PreloadStats {
        total_preloaded: 10,
        successful: 9,
        failed: 1,
        skipped: 0,
        total_time_ms: 900,
    };

    assert_eq!(stats.success_rate(), 90.0);
    assert_eq!(stats.avg_time_ms(), 100.0);
}

#[test]
fn test_model_registry_get_nonexistent() {
    let registry = ModelRegistry::new();
    assert!(registry.get("nonexistent").is_none());
}

#[test]
fn test_preload_manager_set_enabled() {
    let mut manager = PreloadManager::default();
    manager.set_enabled(false);
    assert!(!manager.config().enabled);
}
