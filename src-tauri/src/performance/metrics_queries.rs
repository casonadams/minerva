use super::performance_metrics::PerformanceMetrics;

impl PerformanceMetrics {
    /// Get average inference time in ms
    pub fn avg_inference_time_ms(&self) -> f64 {
        let total = self.total_inferences_count();
        if total == 0 {
            0.0
        } else {
            let time = self.total_time_ms();
            time as f64 / total as f64
        }
    }

    /// Get success rate percentage
    pub fn success_rate_percent(&self) -> f64 {
        let total = self.total_inferences_count();
        if total == 0 {
            0.0
        } else {
            let success = self.successful_count();
            (success as f64 / total as f64) * 100.0
        }
    }

    /// Get current memory usage in MB
    pub fn current_memory_mb(&self) -> u64 {
        self.current_memory_mb
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Get peak memory usage in MB
    pub fn peak_memory_mb(&self) -> u64 {
        self.peak_memory_mb
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    // Private helpers
    fn total_inferences_count(&self) -> u64 {
        self.total_inferences
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    fn successful_count(&self) -> u64 {
        self.successful_inferences
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    fn total_time_ms(&self) -> u64 {
        self.total_inference_time_ms
            .load(std::sync::atomic::Ordering::Relaxed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_avg_inference_time() {
        let m = PerformanceMetrics::new();
        m.record_inference(100, true);
        m.record_inference(150, true);
        assert_eq!(m.avg_inference_time_ms(), 125.0);
    }

    #[test]
    fn test_success_rate() {
        let m = PerformanceMetrics::new();
        m.record_inference(100, true);
        m.record_inference(100, true);
        m.record_inference(100, false);
        let rate = m.success_rate_percent();
        assert!((rate - 66.66).abs() < 0.01);
    }

    #[test]
    fn test_current_memory() {
        let m = PerformanceMetrics::new();
        m.update_memory(512);
        assert_eq!(m.current_memory_mb(), 512);
    }

    #[test]
    fn test_peak_memory() {
        let m = PerformanceMetrics::new();
        m.update_memory(512);
        m.update_memory(256);
        m.update_memory(768);
        assert_eq!(m.peak_memory_mb(), 768);
    }
}
