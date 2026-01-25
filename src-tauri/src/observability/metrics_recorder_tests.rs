#[cfg(test)]
mod tests {
    use crate::observability::metrics_recorder::MetricsRecorder;
    use std::time::Duration;

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
