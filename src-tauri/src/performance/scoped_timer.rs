use super::profiler::Profiler;
use std::time::Instant;

/// Scoped timer that records duration on drop
pub struct ScopedTimer {
    name: String,
    start: Instant,
    profiler: Profiler,
}

impl ScopedTimer {
    /// Create new scoped timer
    pub fn new(name: String, profiler: Profiler) -> Self {
        Self {
            name,
            start: Instant::now(),
            profiler,
        }
    }
}

impl Drop for ScopedTimer {
    fn drop(&mut self) {
        let duration_ms = self.start.elapsed().as_millis() as u64;
        self.profiler.record(&self.name, duration_ms);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_scoped_timer() {
        let prof = Profiler::new();
        {
            let _timer = ScopedTimer::new("test_op".to_string(), prof.clone());
            std::thread::sleep(Duration::from_millis(10));
        }
        let op = prof.get("test_op").unwrap();
        assert_eq!(op.call_count, 1);
        assert!(op.total_duration_ms >= 10);
    }

    #[test]
    fn test_timer_is_recorded() {
        let prof = Profiler::new();
        {
            let _timer = ScopedTimer::new("op1".to_string(), prof.clone());
        }
        assert!(prof.get("op1").is_some());
    }
}
