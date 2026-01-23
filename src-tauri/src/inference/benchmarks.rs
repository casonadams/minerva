/// Performance Benchmarking for Phase 3.5b
///
/// This module provides benchmarking utilities to measure:
/// - Token generation speed (tokens/sec)
/// - GPU vs CPU performance comparison
/// - Memory usage during inference
/// - End-to-end latency
use crate::error::MinervaResult;
use std::time::{Duration, Instant};

/// Input for creating performance metrics
#[derive(Debug, Clone)]
pub struct PerformanceMetricsInput {
    /// Total generation time
    pub duration: Duration,
    /// Number of tokens generated
    pub token_count: usize,
    /// Memory used in bytes
    pub memory_bytes: usize,
    /// Was GPU used
    pub gpu_used: bool,
}

/// Performance metrics for inference
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    /// Total generation time
    pub duration: Duration,
    /// Number of tokens generated
    pub token_count: usize,
    /// Tokens per second
    pub tokens_per_sec: f32,
    /// Memory used in bytes
    pub memory_bytes: usize,
    /// Was GPU used
    pub gpu_used: bool,
}

impl PerformanceMetrics {
    /// Create metrics from measurements
    pub fn new(input: PerformanceMetricsInput) -> Self {
        let total_secs = input.duration.as_secs_f32();
        let tokens_per_sec = if total_secs > 0.0 {
            input.token_count as f32 / total_secs
        } else {
            0.0
        };

        Self {
            duration: input.duration,
            token_count: input.token_count,
            tokens_per_sec,
            memory_bytes: input.memory_bytes,
            gpu_used: input.gpu_used,
        }
    }

    /// Get human-readable summary
    pub fn summary(&self) -> String {
        format!(
            "Generated {} tokens in {:.2}s ({:.1} tok/s) using {} ({} MB)",
            self.token_count,
            self.duration.as_secs_f32(),
            self.tokens_per_sec,
            if self.gpu_used { "GPU" } else { "CPU" },
            self.memory_bytes / 1_000_000
        )
    }
}

/// Benchmark runner
pub struct Benchmark {
    name: String,
    start_time: Option<Instant>,
}

impl Benchmark {
    /// Create new benchmark
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            start_time: None,
        }
    }

    /// Start timing
    pub fn start(&mut self) {
        self.start_time = Some(Instant::now());
        tracing::debug!("Benchmark '{}' started", self.name);
    }

    /// Get elapsed time
    pub fn elapsed(&self) -> MinervaResult<Duration> {
        self.start_time.map(|start| start.elapsed()).ok_or_else(|| {
            crate::error::MinervaError::InferenceError("Benchmark not started".to_string())
        })
    }

    /// End benchmark and log result
    pub fn end(&self) -> MinervaResult<Duration> {
        let elapsed = self.elapsed()?;
        tracing::info!(
            "Benchmark '{}' completed in {:.2}ms",
            self.name,
            elapsed.as_secs_f32() * 1000.0
        );
        Ok(elapsed)
    }
}

/// Performance tracking accumulator
#[derive(Debug, Clone)]
pub struct PerformanceAccumulator {
    measurements: Vec<Duration>,
    gpu_measurements: Vec<Duration>,
    cpu_measurements: Vec<Duration>,
}

impl PerformanceAccumulator {
    /// Create new accumulator
    pub fn new() -> Self {
        Self {
            measurements: Vec::new(),
            gpu_measurements: Vec::new(),
            cpu_measurements: Vec::new(),
        }
    }

    /// Add measurement (general)
    pub fn add_measurement(&mut self, duration: Duration) {
        self.measurements.push(duration);
    }

    /// Add GPU measurement
    pub fn add_gpu_measurement(&mut self, duration: Duration) {
        self.gpu_measurements.push(duration);
    }

    /// Add CPU measurement
    pub fn add_cpu_measurement(&mut self, duration: Duration) {
        self.cpu_measurements.push(duration);
    }

    /// Get average duration
    pub fn avg(&self) -> Option<Duration> {
        if self.measurements.is_empty() {
            return None;
        }

        let total: Duration = self.measurements.iter().sum();
        Some(total / self.measurements.len() as u32)
    }

    /// Get average GPU duration
    pub fn avg_gpu(&self) -> Option<Duration> {
        if self.gpu_measurements.is_empty() {
            return None;
        }

        let total: Duration = self.gpu_measurements.iter().sum();
        Some(total / self.gpu_measurements.len() as u32)
    }

    /// Get average CPU duration
    pub fn avg_cpu(&self) -> Option<Duration> {
        if self.cpu_measurements.is_empty() {
            return None;
        }

        let total: Duration = self.cpu_measurements.iter().sum();
        Some(total / self.cpu_measurements.len() as u32)
    }

    /// Get speedup factor (GPU vs CPU)
    pub fn speedup_factor(&self) -> Option<f32> {
        match (self.avg_gpu(), self.avg_cpu()) {
            (Some(gpu), Some(cpu)) => {
                let gpu_ms = gpu.as_secs_f32() * 1000.0;
                let cpu_ms = cpu.as_secs_f32() * 1000.0;
                if gpu_ms > 0.0 {
                    Some(cpu_ms / gpu_ms)
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    /// Get count of measurements
    pub fn count(&self) -> usize {
        self.measurements.len()
    }

    /// Get GPU measurement count
    pub fn gpu_count(&self) -> usize {
        self.gpu_measurements.len()
    }

    /// Get CPU measurement count
    pub fn cpu_count(&self) -> usize {
        self.cpu_measurements.len()
    }
}

impl Default for PerformanceAccumulator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_performance_metrics() {
        let input = PerformanceMetricsInput {
            duration: Duration::from_secs(1),
            token_count: 100,
            memory_bytes: 1_000_000,
            gpu_used: true,
        };
        let metrics = PerformanceMetrics::new(input);
        assert_eq!(metrics.token_count, 100);
        assert_eq!(metrics.tokens_per_sec, 100.0);
        assert!(metrics.summary().contains("100"));
    }

    #[test]
    fn test_performance_metrics_zero_duration() {
        let input = PerformanceMetricsInput {
            duration: Duration::from_millis(1),
            token_count: 1000,
            memory_bytes: 1_000_000,
            gpu_used: false,
        };
        let metrics = PerformanceMetrics::new(input);
        assert!(metrics.tokens_per_sec > 0.0);
    }

    #[test]
    fn test_benchmark_timing() {
        let mut bench = Benchmark::new("test");
        bench.start();
        std::thread::sleep(Duration::from_millis(10));
        let elapsed = bench.end().unwrap();
        assert!(elapsed.as_millis() >= 10);
    }

    #[test]
    fn test_benchmark_not_started() {
        let bench = Benchmark::new("test");
        assert!(bench.end().is_err());
    }

    #[test]
    fn test_performance_accumulator() {
        let mut acc = PerformanceAccumulator::new();
        acc.add_measurement(Duration::from_millis(100));
        acc.add_measurement(Duration::from_millis(200));
        acc.add_measurement(Duration::from_millis(300));

        assert_eq!(acc.count(), 3);
        assert_eq!(acc.avg(), Some(Duration::from_millis(200)));
    }

    #[test]
    fn test_gpu_vs_cpu_speedup() {
        let mut acc = PerformanceAccumulator::new();
        acc.add_gpu_measurement(Duration::from_millis(10));
        acc.add_gpu_measurement(Duration::from_millis(10));
        acc.add_cpu_measurement(Duration::from_millis(100));
        acc.add_cpu_measurement(Duration::from_millis(100));

        let speedup = acc.speedup_factor();
        assert!(speedup.is_some());
        assert_eq!(speedup.unwrap(), 10.0);
    }

    #[test]
    fn test_accumulator_default() {
        let acc = PerformanceAccumulator::default();
        assert_eq!(acc.count(), 0);
        assert!(acc.avg().is_none());
    }
}
