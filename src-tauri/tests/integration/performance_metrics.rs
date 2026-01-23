//! Performance metrics integration tests
//!
//! Tests for performance tracking, benchmarking, and GPU vs CPU performance comparison.
//! Covers metrics collection, calculation, and analysis.

use minerva_lib::inference::benchmarks::{
    PerformanceAccumulator, PerformanceMetrics, PerformanceMetricsInput,
};
use std::time::Duration;

#[test]
fn test_performance_metrics_tracking() {
    let input = PerformanceMetricsInput {
        duration: Duration::from_secs(1),
        token_count: 256,
        memory_bytes: 2_000_000,
        gpu_used: true,
    };
    let metrics = PerformanceMetrics::new(input);

    assert_eq!(metrics.token_count, 256);
    assert_eq!(metrics.tokens_per_sec, 256.0);
    assert!(metrics.summary().contains("GPU"));
    assert!(metrics.summary().contains("256"));
}

#[test]
fn test_gpu_vs_cpu_performance_comparison() {
    let mut accumulator = PerformanceAccumulator::new();

    // Simulate GPU measurements (faster)
    accumulator.add_gpu_measurement(Duration::from_millis(50));
    accumulator.add_gpu_measurement(Duration::from_millis(55));

    // Simulate CPU measurements (slower)
    accumulator.add_cpu_measurement(Duration::from_millis(500));
    accumulator.add_cpu_measurement(Duration::from_millis(550));

    let speedup = accumulator.speedup_factor().unwrap();
    assert!(speedup > 5.0); // GPU should be ~10x faster
    assert_eq!(accumulator.gpu_count(), 2);
    assert_eq!(accumulator.cpu_count(), 2);
}
