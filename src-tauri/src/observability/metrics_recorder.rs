use super::response_time_store::ResponseTimeStore;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

/// Records metrics for requests and cache operations
pub struct MetricsRecorder {
    total_requests: AtomicU64,
    successful_requests: AtomicU64,
    failed_requests: AtomicU64,
    response_times: ResponseTimeStore,
    cache_hits: AtomicU64,
    cache_misses: AtomicU64,
}

impl MetricsRecorder {
    /// Create new metrics recorder
    pub fn new() -> Self {
        Self {
            total_requests: AtomicU64::new(0),
            successful_requests: AtomicU64::new(0),
            failed_requests: AtomicU64::new(0),
            response_times: ResponseTimeStore::new(),
            cache_hits: AtomicU64::new(0),
            cache_misses: AtomicU64::new(0),
        }
    }

    /// Record a successful request with response time
    pub fn record_success(&self, response_time: Duration) {
        self.successful_requests.fetch_add(1, Ordering::Relaxed);
        self.total_requests.fetch_add(1, Ordering::Relaxed);
        self.response_times.store(response_time);
    }

    /// Record a failed request with response time
    pub fn record_failure(&self, response_time: Duration) {
        self.failed_requests.fetch_add(1, Ordering::Relaxed);
        self.total_requests.fetch_add(1, Ordering::Relaxed);
        self.response_times.store(response_time);
    }

    /// Record cache hit
    pub fn record_cache_hit(&self) {
        self.cache_hits.fetch_add(1, Ordering::Relaxed);
    }

    /// Record cache miss
    pub fn record_cache_miss(&self) {
        self.cache_misses.fetch_add(1, Ordering::Relaxed);
    }

    /// Get total requests
    pub fn total_requests(&self) -> u64 {
        self.total_requests.load(Ordering::Relaxed)
    }

    /// Get successful requests
    pub fn successful_requests(&self) -> u64 {
        self.successful_requests.load(Ordering::Relaxed)
    }

    /// Get failed requests
    pub fn failed_requests(&self) -> u64 {
        self.failed_requests.load(Ordering::Relaxed)
    }

    /// Get cache hits
    pub fn cache_hits(&self) -> u64 {
        self.cache_hits.load(Ordering::Relaxed)
    }

    /// Get cache misses
    pub fn cache_misses(&self) -> u64 {
        self.cache_misses.load(Ordering::Relaxed)
    }

    /// Get all response times
    pub fn response_times(&self) -> Vec<Duration> {
        self.response_times.get_times()
    }

    /// Reset all metrics
    pub fn reset(&self) {
        self.total_requests.store(0, Ordering::Relaxed);
        self.successful_requests.store(0, Ordering::Relaxed);
        self.failed_requests.store(0, Ordering::Relaxed);
        self.cache_hits.store(0, Ordering::Relaxed);
        self.cache_misses.store(0, Ordering::Relaxed);
        self.response_times.clear();
    }
}

impl Clone for MetricsRecorder {
    fn clone(&self) -> Self {
        Self {
            total_requests: AtomicU64::new(self.total_requests.load(Ordering::Relaxed)),
            successful_requests: AtomicU64::new(self.successful_requests.load(Ordering::Relaxed)),
            failed_requests: AtomicU64::new(self.failed_requests.load(Ordering::Relaxed)),
            response_times: self.response_times.clone(),
            cache_hits: AtomicU64::new(self.cache_hits.load(Ordering::Relaxed)),
            cache_misses: AtomicU64::new(self.cache_misses.load(Ordering::Relaxed)),
        }
    }
}

impl Default for MetricsRecorder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_record_success() {
        let recorder = MetricsRecorder::new();
        recorder.record_success(Duration::from_millis(100));
        assert_eq!(recorder.total_requests(), 1);
        assert_eq!(recorder.successful_requests(), 1);
        assert_eq!(recorder.failed_requests(), 0);
    }

    #[test]
    fn test_record_failure() {
        let recorder = MetricsRecorder::new();
        recorder.record_failure(Duration::from_millis(100));
        assert_eq!(recorder.total_requests(), 1);
        assert_eq!(recorder.successful_requests(), 0);
        assert_eq!(recorder.failed_requests(), 1);
    }

    #[test]
    fn test_cache_operations() {
        let recorder = MetricsRecorder::new();
        for _ in 0..8 {
            recorder.record_cache_hit();
        }
        for _ in 0..2 {
            recorder.record_cache_miss();
        }
        assert_eq!(recorder.cache_hits(), 8);
        assert_eq!(recorder.cache_misses(), 2);
    }

    #[test]
    fn test_reset() {
        let recorder = MetricsRecorder::new();
        recorder.record_success(Duration::from_millis(100));
        assert_eq!(recorder.total_requests(), 1);
        recorder.reset();
        assert_eq!(recorder.total_requests(), 0);
    }
}
