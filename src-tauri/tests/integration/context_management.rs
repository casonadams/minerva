//! Context management integration tests
//!
//! Tests for multi-model context management, memory tracking, and cache statistics.
//! Covers context lifecycle, memory pressure, and policy application.

use minerva_lib::inference::context_manager::ContextManager;
use minerva_lib::inference::model_cache::EvictionPolicy;

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

    let mut full_manager = ContextManager::new(1);
    full_manager.update_memory_estimate();
    let memory_after = full_manager.estimated_memory_mb();
    assert_eq!(memory_after, 0);
}

#[test]
fn test_context_manager_with_policy() {
    let manager = ContextManager::with_policy(4, EvictionPolicy::Lfu);
    assert_eq!(manager.max_models(), 4);
    assert_eq!(manager.loaded_count(), 0);
}
