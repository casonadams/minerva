//! Model preloading integration tests
//!
//! Tests for preload manager, preload strategies, and preloading statistics.
//! Covers queue management, configuration, and preload strategy selection.

use minerva_lib::inference::model_registry::ModelRegistry;
use minerva_lib::inference::preload_manager::{
    PreloadConfig, PreloadManager, PreloadStats, PreloadStrategy,
};

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
    let manager = PreloadManager::new(ModelRegistry::default());
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
fn test_preload_manager_set_enabled() {
    let mut manager = PreloadManager::default();
    manager.set_enabled(false);
    assert!(!manager.config().enabled);
}

#[test]
fn test_preload_config_default() {
    let config = PreloadConfig::default();
    assert!(config.enabled);
    assert_eq!(config.batch_size, 1);
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
fn test_preload_stats_calculation() {
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
