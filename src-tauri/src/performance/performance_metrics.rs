use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

/// Performance metrics tracker
pub struct PerformanceMetrics {
    pub(crate) total_inferences: Arc<AtomicU64>,
    pub(crate) successful_inferences: Arc<AtomicU64>,
    pub(crate) total_inference_time_ms: Arc<AtomicU64>,
    pub(crate) peak_memory_mb: Arc<AtomicU64>,
    pub(crate) current_memory_mb: Arc<AtomicU64>,
}

impl PerformanceMetrics {
    /// Create new metrics tracker
    pub fn new() -> Self {
        Self {
            total_inferences: Arc::new(AtomicU64::new(0)),
            successful_inferences: Arc::new(AtomicU64::new(0)),
            total_inference_time_ms: Arc::new(AtomicU64::new(0)),
            peak_memory_mb: Arc::new(AtomicU64::new(0)),
            current_memory_mb: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Record inference completion
    pub fn record_inference(&self, duration_ms: u64, success: bool) {
        self.total_inferences.fetch_add(1, Ordering::Relaxed);
        if success {
            self.successful_inferences.fetch_add(1, Ordering::Relaxed);
        }
        self.total_inference_time_ms
            .fetch_add(duration_ms, Ordering::Relaxed);
    }

    /// Update memory usage
    pub fn update_memory(&self, current_mb: u64) {
        self.current_memory_mb.store(current_mb, Ordering::Relaxed);
        let peak = self.peak_memory_mb.load(Ordering::Relaxed);
        if current_mb > peak {
            self.peak_memory_mb.store(current_mb, Ordering::Relaxed);
        }
    }

    /// Reset all metrics
    pub fn reset(&self) {
        self.total_inferences.store(0, Ordering::Relaxed);
        self.successful_inferences.store(0, Ordering::Relaxed);
        self.total_inference_time_ms.store(0, Ordering::Relaxed);
        self.peak_memory_mb.store(0, Ordering::Relaxed);
    }
}

impl Clone for PerformanceMetrics {
    fn clone(&self) -> Self {
        Self {
            total_inferences: Arc::clone(&self.total_inferences),
            successful_inferences: Arc::clone(&self.successful_inferences),
            total_inference_time_ms: Arc::clone(&self.total_inference_time_ms),
            peak_memory_mb: Arc::clone(&self.peak_memory_mb),
            current_memory_mb: Arc::clone(&self.current_memory_mb),
        }
    }
}

impl Default for PerformanceMetrics {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_creation() {
        let m = PerformanceMetrics::new();
        assert_eq!(m.avg_inference_time_ms(), 0.0);
    }

    #[test]
    fn test_memory_tracking() {
        let m = PerformanceMetrics::new();
        m.update_memory(512);
        m.update_memory(256);
        m.update_memory(768);
        assert_eq!(m.current_memory_mb(), 768);
        assert_eq!(m.peak_memory_mb(), 768);
    }

    #[test]
    fn test_reset() {
        let m = PerformanceMetrics::new();
        m.record_inference(100, true);
        m.update_memory(512);
        m.reset();
        assert_eq!(m.current_memory_mb(), 512);
        assert_eq!(m.avg_inference_time_ms(), 0.0);
    }

    #[test]
    fn test_cloneable() {
        let m1 = PerformanceMetrics::new();
        let m2 = m1.clone();
        m1.record_inference(100, true);
        assert_eq!(m2.avg_inference_time_ms(), 100.0);
    }
}
