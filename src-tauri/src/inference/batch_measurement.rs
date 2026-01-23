/// Performance measurement utilities for batch processing operations
///
/// This module provides timing and performance measurement functions
/// for profiling batch operations without relying on external benchmarking tools.
use std::time::Instant;

/// Measure the execution time of a closure and return the result + duration
pub fn measure_time<F, R>(f: F) -> (R, u128)
where
    F: FnOnce() -> R,
{
    let start = Instant::now();
    let result = f();
    let duration = start.elapsed().as_micros();
    (result, duration)
}

/// Perform a simple timing measurement
pub fn time_operation<F>(name: &str, iterations: usize, f: F) -> f64
where
    F: Fn(),
{
    let start = Instant::now();
    for _ in 0..iterations {
        f();
    }
    let duration = start.elapsed().as_micros() as f64;
    let avg_us = duration / iterations as f64;
    let avg_ms = avg_us / 1000.0;

    println!(
        "{}: {:.3}ms (avg over {} iterations)",
        name, avg_ms, iterations
    );
    avg_ms
}

/// Perform timing measurement and return raw statistics
pub fn measure_operation<F>(name: &str, iterations: usize, f: F) -> OperationStats
where
    F: Fn(),
{
    let mut durations = Vec::new();

    for _ in 0..iterations {
        let start = Instant::now();
        f();
        let duration_us = start.elapsed().as_micros();
        durations.push(duration_us);
    }

    let total: u128 = durations.iter().sum();
    let avg = total / iterations as u128;
    let min = *durations.iter().min().unwrap_or(&0);
    let max = *durations.iter().max().unwrap_or(&0);

    let stats = OperationStats {
        name: name.to_string(),
        iterations,
        total_us: total,
        avg_us: avg,
        min_us: min,
        max_us: max,
    };

    println!("{}", stats);
    stats
}

/// Statistics for a timed operation
#[derive(Debug, Clone)]
pub struct OperationStats {
    pub name: String,
    pub iterations: usize,
    pub total_us: u128,
    pub avg_us: u128,
    pub min_us: u128,
    pub max_us: u128,
}

impl OperationStats {
    pub fn avg_ms(&self) -> f64 {
        self.avg_us as f64 / 1000.0
    }

    pub fn total_ms(&self) -> f64 {
        self.total_us as f64 / 1000.0
    }

    pub fn throughput_per_second(&self) -> f64 {
        (self.iterations as f64 * 1_000_000.0) / self.total_us as f64
    }
}

impl std::fmt::Display for OperationStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}: avg={:.3}ms, min={:.3}ms, max={:.3}ms, total={:.3}ms, throughput={:.0}/sec",
            self.name,
            self.avg_ms(),
            self.min_us as f64 / 1000.0,
            self.max_us as f64 / 1000.0,
            self.total_ms(),
            self.throughput_per_second()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_measure_time() {
        let (result, duration) = measure_time(|| {
            std::thread::sleep(std::time::Duration::from_millis(1));
            42
        });

        assert_eq!(result, 42);
        assert!(duration >= 1000); // At least 1000 microseconds (1ms)
    }

    #[test]
    fn test_operation_stats_calculations() {
        let stats = OperationStats {
            name: "test".to_string(),
            iterations: 10,
            total_us: 10_000, // 10ms total
            avg_us: 1_000,    // 1ms average
            min_us: 900,
            max_us: 1_200,
        };

        assert!((stats.avg_ms() - 1.0).abs() < 0.01);
        assert!((stats.total_ms() - 10.0).abs() < 0.01);
        assert!(stats.throughput_per_second() > 900.0); // ~1000/sec
    }
}
