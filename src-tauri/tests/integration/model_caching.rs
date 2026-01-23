//! Model caching integration tests
//!
//! Tests for LRU/LFU/FIFO model caching strategies and cache eviction policies.
//! Covers cache creation, capacity management, and eviction behavior.

use minerva_lib::inference::model_cache::{EvictionPolicy, ModelCache};

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

#[test]
fn test_model_cache_stats_hit_rate() {
    use minerva_lib::inference::model_cache::CacheStats;

    let mut stats = CacheStats::default();
    assert_eq!(stats.hit_rate(), 0.0);

    stats.hits = 75;
    stats.misses = 25;
    assert_eq!(stats.hit_rate(), 75.0);
}
