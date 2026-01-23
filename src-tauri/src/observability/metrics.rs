use parking_lot::RwLock;
/// Metrics Collection and Aggregation
///
/// Tracks server-wide metrics:
/// - Request counts and rates
/// - Response times (average, percentiles)
/// - Error tracking and rates
/// - Cache hit rates
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

/// Metrics collector
pub struct MetricsCollector {
    state: Arc<MetricsState>,
}

struct MetricsState {
    total_requests: AtomicU64,
    successful_requests: AtomicU64,
    failed_requests: AtomicU64,
    response_times: RwLock<Vec<Duration>>,
    cache_hits: AtomicU64,
    cache_misses: AtomicU64,
    start_time: std::time::Instant,
}

impl MetricsCollector {
    /// Create new metrics collector
    pub fn new() -> Self {
        Self {
            state: Arc::new(MetricsState {
                total_requests: AtomicU64::new(0),
                successful_requests: AtomicU64::new(0),
                failed_requests: AtomicU64::new(0),
                response_times: RwLock::new(Vec::new()),
                cache_hits: AtomicU64::new(0),
                cache_misses: AtomicU64::new(0),
                start_time: std::time::Instant::now(),
            }),
        }
    }

    /// Record a successful request with response time
    pub fn record_success(&self, response_time: Duration) {
        self.state
            .successful_requests
            .fetch_add(1, Ordering::Relaxed);
        self.state.total_requests.fetch_add(1, Ordering::Relaxed);

        let mut times = self.state.response_times.write();
        times.push(response_time);
        // Keep only last 10000 measurements to avoid unbounded memory
        if times.len() > 10000 {
            times.drain(0..5000);
        }
    }

    /// Record a failed request with response time
    pub fn record_failure(&self, response_time: Duration) {
        self.state.failed_requests.fetch_add(1, Ordering::Relaxed);
        self.state.total_requests.fetch_add(1, Ordering::Relaxed);

        let mut times = self.state.response_times.write();
        times.push(response_time);
        if times.len() > 10000 {
            times.drain(0..5000);
        }
    }

    /// Record cache hit
    pub fn record_cache_hit(&self) {
        self.state.cache_hits.fetch_add(1, Ordering::Relaxed);
    }

    /// Record cache miss
    pub fn record_cache_miss(&self) {
        self.state.cache_misses.fetch_add(1, Ordering::Relaxed);
    }

    /// Get current metrics snapshot
    pub fn snapshot(&self) -> MetricsSnapshot {
        let total = self.state.total_requests.load(Ordering::Relaxed);
        let success = self.state.successful_requests.load(Ordering::Relaxed);
        let failed = self.state.failed_requests.load(Ordering::Relaxed);
        let hits = self.state.cache_hits.load(Ordering::Relaxed);
        let misses = self.state.cache_misses.load(Ordering::Relaxed);

        let times = self.state.response_times.read();
        let (avg, min, max, p50, p95, p99) = Self::analyze_times(&times);

        let uptime_secs = self.state.start_time.elapsed().as_secs();
        let rps = if uptime_secs > 0 {
            total as f64 / uptime_secs as f64
        } else {
            0.0
        };

        let error_rate = if total > 0 {
            (failed as f64 / total as f64) * 100.0
        } else {
            0.0
        };

        let hit_rate = if (hits + misses) > 0 {
            (hits as f64 / (hits + misses) as f64) * 100.0
        } else {
            0.0
        };

        MetricsSnapshot {
            total_requests: total,
            successful_requests: success,
            failed_requests: failed,
            avg_response_time_ms: avg,
            min_response_time_ms: min,
            max_response_time_ms: max,
            p50_response_time_ms: p50,
            p95_response_time_ms: p95,
            p99_response_time_ms: p99,
            rps,
            error_rate_percent: error_rate,
            cache_hits: hits,
            cache_misses: misses,
            cache_hit_rate_percent: hit_rate,
            uptime_seconds: uptime_secs,
        }
    }

    /// Reset all metrics
    pub fn reset(&self) {
        self.state.total_requests.store(0, Ordering::Relaxed);
        self.state.successful_requests.store(0, Ordering::Relaxed);
        self.state.failed_requests.store(0, Ordering::Relaxed);
        self.state.cache_hits.store(0, Ordering::Relaxed);
        self.state.cache_misses.store(0, Ordering::Relaxed);
        self.state.response_times.write().clear();
    }

    fn analyze_times(times: &[Duration]) -> (f64, f64, f64, f64, f64, f64) {
        if times.is_empty() {
            return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        }

        let mut vals: Vec<u128> = times.iter().map(|d| d.as_millis()).collect();
        vals.sort_unstable();

        let sum: u128 = vals.iter().sum();
        let avg = sum as f64 / vals.len() as f64;
        let min = vals[0] as f64;
        let max = vals[vals.len() - 1] as f64;

        let p50_idx = (vals.len() / 2).saturating_sub(1).min(vals.len() - 1);
        let p95_idx = ((vals.len() as f64 * 0.95) as usize)
            .saturating_sub(1)
            .min(vals.len() - 1);
        let p99_idx = ((vals.len() as f64 * 0.99) as usize)
            .saturating_sub(1)
            .min(vals.len() - 1);

        (
            avg,
            min,
            max,
            vals[p50_idx] as f64,
            vals[p95_idx] as f64,
            vals[p99_idx] as f64,
        )
    }
}

impl Clone for MetricsCollector {
    fn clone(&self) -> Self {
        Self {
            state: Arc::clone(&self.state),
        }
    }
}

impl Default for MetricsCollector {
    fn default() -> Self {
        Self::new()
    }
}

/// Metrics snapshot
#[derive(Debug, Clone)]
pub struct MetricsSnapshot {
    pub total_requests: u64,
    pub successful_requests: u64,
    pub failed_requests: u64,
    pub avg_response_time_ms: f64,
    pub min_response_time_ms: f64,
    pub max_response_time_ms: f64,
    pub p50_response_time_ms: f64,
    pub p95_response_time_ms: f64,
    pub p99_response_time_ms: f64,
    pub rps: f64,
    pub error_rate_percent: f64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub cache_hit_rate_percent: f64,
    pub uptime_seconds: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_collector_creation() {
        let c = MetricsCollector::new();
        let s = c.snapshot();
        assert_eq!(s.total_requests, 0);
    }

    #[test]
    fn test_record_success() {
        let c = MetricsCollector::new();
        c.record_success(Duration::from_millis(100));
        let s = c.snapshot();
        assert_eq!(s.total_requests, 1);
        assert_eq!(s.successful_requests, 1);
        assert_eq!(s.failed_requests, 0);
    }

    #[test]
    fn test_record_failure() {
        let c = MetricsCollector::new();
        c.record_failure(Duration::from_millis(100));
        let s = c.snapshot();
        assert_eq!(s.total_requests, 1);
        assert_eq!(s.successful_requests, 0);
        assert_eq!(s.failed_requests, 1);
    }

    #[test]
    fn test_error_rate() {
        let c = MetricsCollector::new();
        for _ in 0..8 {
            c.record_success(Duration::from_millis(100));
        }
        for _ in 0..2 {
            c.record_failure(Duration::from_millis(100));
        }

        let s = c.snapshot();
        assert_eq!(s.total_requests, 10);
        assert_eq!(s.failed_requests, 2);
        assert_eq!(s.error_rate_percent, 20.0);
    }

    #[test]
    fn test_cache_tracking() {
        let c = MetricsCollector::new();
        for _ in 0..8 {
            c.record_cache_hit();
        }
        for _ in 0..2 {
            c.record_cache_miss();
        }

        let s = c.snapshot();
        assert_eq!(s.cache_hits, 8);
        assert_eq!(s.cache_misses, 2);
        assert_eq!(s.cache_hit_rate_percent, 80.0);
    }

    #[test]
    fn test_response_time_percentiles() {
        let c = MetricsCollector::new();
        for i in 1..=100 {
            c.record_success(Duration::from_millis(i));
        }

        let s = c.snapshot();
        assert!(s.avg_response_time_ms > 0.0);
        assert!(s.p95_response_time_ms >= s.p50_response_time_ms);
        assert!(s.p99_response_time_ms >= s.p95_response_time_ms);
    }

    #[test]
    fn test_rps_calculation() {
        let c = MetricsCollector::new();
        c.record_success(Duration::from_millis(100));
        c.record_success(Duration::from_millis(100));

        let s = c.snapshot();
        assert!(s.rps >= 0.0);
    }

    #[test]
    fn test_reset() {
        let c = MetricsCollector::new();
        c.record_success(Duration::from_millis(100));
        assert_eq!(c.snapshot().total_requests, 1);

        c.reset();
        assert_eq!(c.snapshot().total_requests, 0);
    }

    #[test]
    fn test_cloneable() {
        let c1 = MetricsCollector::new();
        let c2 = c1.clone();

        c1.record_success(Duration::from_millis(100));
        // Both should see the same state
        assert_eq!(c2.snapshot().total_requests, 1);
    }
}
