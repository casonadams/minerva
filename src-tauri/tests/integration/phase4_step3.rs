// Phase 4 Step 3: Advanced Parameter Tuning and Optimization

use minerva_lib::inference::cache_optimizer::{
    CacheOptimizer, OptimizationConfig, OptimizationStrategy, SystemMemory,
};
use minerva_lib::inference::garbage_collector::{GCStats, GarbageCollector};
use minerva_lib::inference::pattern_detector::PatternDetector;

#[test]
fn test_system_memory_available_percent() {
    let mem = SystemMemory {
        total_mb: 16000,
        available_mb: 8000,
        used_mb: 8000,
    };
    assert_eq!(mem.available_percent(), 50.0);
}

#[test]
fn test_system_memory_under_pressure() {
    let mem = SystemMemory {
        total_mb: 10000,
        available_mb: 1000,
        used_mb: 9000,
    };
    assert!(mem.under_pressure());
}

#[test]
fn test_optimization_strategy_default() {
    let strategy = OptimizationStrategy::default();
    assert!(matches!(strategy, OptimizationStrategy::Balanced));
}

#[test]
fn test_cache_optimizer_creation() {
    let optimizer = CacheOptimizer::new();
    assert_eq!(optimizer.current_size(), 3000);
}

#[test]
fn test_cache_optimizer_calculate_optimal_size() {
    let optimizer = CacheOptimizer::new();
    let mem = SystemMemory {
        total_mb: 16000,
        available_mb: 8000,
        used_mb: 8000,
    };

    let optimal = optimizer.calculate_optimal_size(&mem);
    assert!(optimal > 0);
}

#[test]
fn test_cache_optimizer_should_optimize() {
    let optimizer = CacheOptimizer::new();
    assert!(optimizer.should_optimize());
}

#[test]
fn test_cache_optimizer_conservative_strategy() {
    let config = OptimizationConfig {
        strategy: OptimizationStrategy::Conservative,
        ..Default::default()
    };
    let optimizer = CacheOptimizer::with_config(config);

    let mem = SystemMemory {
        total_mb: 16000,
        available_mb: 8000,
        used_mb: 8000,
    };

    let optimal = optimizer.calculate_optimal_size(&mem);
    assert!(optimal > 0);
}

#[test]
fn test_cache_optimizer_aggressive_strategy() {
    let config = OptimizationConfig {
        strategy: OptimizationStrategy::Aggressive,
        ..Default::default()
    };
    let optimizer = CacheOptimizer::with_config(config);

    let mem = SystemMemory {
        total_mb: 16000,
        available_mb: 8000,
        used_mb: 8000,
    };

    let optimal = optimizer.calculate_optimal_size(&mem);
    assert!(optimal > 0);
}

#[test]
fn test_cache_optimizer_set_size() {
    let mut optimizer = CacheOptimizer::new();
    optimizer.set_size(5000);
    assert_eq!(optimizer.current_size(), 5000);
}

#[test]
fn test_cache_optimizer_set_size_respects_bounds() {
    let mut optimizer = CacheOptimizer::new();
    optimizer.set_size(100000);
    assert!(optimizer.current_size() <= optimizer.config().max_cache_mb);
}

#[test]
fn test_pattern_detector_creation() {
    let detector = PatternDetector::new(10);
    assert_eq!(detector.total_models(), 0);
}

#[test]
fn test_pattern_detector_record_access() {
    let mut detector = PatternDetector::new(10);
    detector.record_access("model-1");
    detector.record_access("model-1");
    assert_eq!(detector.get_access_count("model-1"), 2);
}

#[test]
fn test_pattern_detector_hot_models() {
    let mut detector = PatternDetector::new(5);
    for _ in 0..6 {
        detector.record_access("model-1");
    }
    detector.record_access("model-2");

    let hot = detector.get_hot_models();
    assert_eq!(hot.len(), 1);
}

#[test]
fn test_pattern_detector_cold_models() {
    let mut detector = PatternDetector::new(5);
    detector.record_access("model-1");
    detector.record_access("model-2");

    let cold = detector.get_cold_models();
    assert_eq!(cold.len(), 2);
}

#[test]
fn test_pattern_detector_analyze() {
    let mut detector = PatternDetector::new(3);
    for _ in 0..4 {
        detector.record_access("model-1");
    }

    let results = detector.analyze();
    assert!(!results.is_empty());
}

#[test]
fn test_pattern_detector_clear() {
    let mut detector = PatternDetector::new(5);
    detector.record_access("model-1");
    detector.clear();
    assert_eq!(detector.total_models(), 0);
}

#[test]
fn test_pattern_detector_set_threshold() {
    let mut detector = PatternDetector::new(10);
    detector.set_hot_threshold(5);
    for _ in 0..5 {
        detector.record_access("model-1");
    }
    assert!(detector.get_pattern("model-1").unwrap().is_hot(5));
}

#[test]
fn test_garbage_collector_creation() {
    let collector = GarbageCollector::new();
    assert!(collector.config().auto_collect);
}

#[test]
fn test_garbage_collector_should_collect() {
    let collector = GarbageCollector::new();
    // New collectors should be ready to collect
    assert!(collector.stats().total_collections == 0);
}

#[test]
fn test_garbage_collector_collect() {
    let mut collector = GarbageCollector::new();
    collector.collect(100, 5);
    assert_eq!(collector.stats().total_collections, 1);
    assert_eq!(collector.stats().total_freed_mb, 100);
}

#[test]
fn test_garbage_collector_stats() {
    let collector = GarbageCollector::new();
    let stats = collector.stats();
    assert_eq!(stats.total_collections, 0);
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
fn test_optimization_config_default() {
    let config = OptimizationConfig::default();
    assert!(config.auto_optimize);
}

#[test]
fn test_garbage_collector_set_auto_collect() {
    let mut collector = GarbageCollector::new();
    collector.set_auto_collect(false);
    assert!(!collector.config().auto_collect);
}

#[test]
fn test_pattern_detector_should_analyze() {
    let detector = PatternDetector::new(5);
    assert!(detector.should_analyze());
}

#[test]
fn test_cache_optimizer_config() {
    let optimizer = CacheOptimizer::new();
    assert_eq!(optimizer.config().min_cache_mb, 1000);
    assert_eq!(optimizer.config().max_cache_mb, 50000);
}
