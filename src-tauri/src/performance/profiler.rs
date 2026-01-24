pub use super::operation_profile::OperationProfile;
use super::profile_analyzer::ProfileAnalyzer;
use super::scoped_timer::ScopedTimer;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;

/// Performance profiler
///
/// Lightweight profiling for desktop app optimization:
/// - Operation timing
/// - Bottleneck detection
/// - Performance reports
pub struct Profiler {
    profiles: Arc<RwLock<HashMap<String, OperationProfile>>>,
}

impl Profiler {
    /// Create new profiler
    pub fn new() -> Self {
        Self {
            profiles: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Start timing an operation
    pub fn start(&self, name: &str) -> ScopedTimer {
        ScopedTimer::new(name.to_string(), self.clone())
    }

    /// Record operation timing
    pub fn record(&self, name: &str, duration_ms: u64) {
        let mut profiles = self.profiles.write();
        profiles
            .entry(name.to_string())
            .or_insert_with(|| OperationProfile::new(name))
            .add(duration_ms);
    }

    /// Get profile for operation
    pub fn get(&self, name: &str) -> Option<OperationProfile> {
        self.profiles.read().get(name).cloned()
    }

    /// Get all profiles
    pub fn all(&self) -> Vec<OperationProfile> {
        self.profiles.read().values().cloned().collect()
    }

    /// Get top N slowest operations by total time
    pub fn top_slowest(&self, n: usize) -> Vec<OperationProfile> {
        let profiles = self.all();
        ProfileAnalyzer::top_slowest(&profiles, n)
    }

    /// Get summary report
    pub fn report(&self) -> String {
        let profiles = self.all();
        ProfileAnalyzer::generate_report(&profiles)
    }

    /// Reset all profiles
    pub fn reset(&self) {
        self.profiles.write().clear();
    }
}

impl Clone for Profiler {
    fn clone(&self) -> Self {
        Self {
            profiles: Arc::clone(&self.profiles),
        }
    }
}

impl Default for Profiler {
    fn default() -> Self {
        Self::new()
    }
}

// ScopedTimer is in scoped_timer module

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_profiler_creation() {
        let prof = Profiler::new();
        assert!(prof.all().is_empty());
    }

    #[test]
    fn test_profiler_record() {
        let prof = Profiler::new();
        prof.record("op1", 100);
        prof.record("op1", 150);
        prof.record("op2", 200);

        assert_eq!(prof.all().len(), 2);
        let op1 = prof.get("op1").unwrap();
        assert_eq!(op1.call_count, 2);
        assert_eq!(op1.total_duration_ms, 250);
    }

    #[test]
    fn test_profiler_scoped_timer() {
        use std::time::Duration;
        let prof = Profiler::new();
        {
            let _timer = prof.start("test_op");
            std::thread::sleep(Duration::from_millis(10));
        }
        let op = prof.get("test_op").unwrap();
        assert_eq!(op.call_count, 1);
        assert!(op.total_duration_ms >= 10);
    }

    #[test]
    fn test_profiler_reset() {
        let prof = Profiler::new();
        prof.record("op1", 100);
        assert_eq!(prof.all().len(), 1);

        prof.reset();
        assert!(prof.all().is_empty());
    }

    #[test]
    fn test_profiler_cloneable() {
        let prof1 = Profiler::new();
        let prof2 = prof1.clone();

        prof1.record("op1", 100);
        assert_eq!(prof2.get("op1").unwrap().call_count, 1);
    }
}
