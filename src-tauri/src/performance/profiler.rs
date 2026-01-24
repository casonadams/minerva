use super::profile_analyzer::ProfileAnalyzer;
use super::scoped_timer::ScopedTimer;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;
/// Performance Profiler
///
/// Lightweight profiling for desktop app optimization:
/// - Operation timing
/// - Bottleneck detection
/// - Performance reports

/// Operation profile data
#[derive(Debug, Clone)]
pub struct OperationProfile {
    /// Operation name
    pub name: String,
    /// Total calls
    pub call_count: u64,
    /// Total duration
    pub total_duration_ms: u64,
    /// Minimum duration
    pub min_duration_ms: u64,
    /// Maximum duration
    pub max_duration_ms: u64,
}

impl OperationProfile {
    /// Create new profile
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            call_count: 0,
            total_duration_ms: 0,
            min_duration_ms: u64::MAX,
            max_duration_ms: 0,
        }
    }

    /// Add measurement
    pub fn add(&mut self, duration_ms: u64) {
        self.call_count += 1;
        self.total_duration_ms += duration_ms;
        self.min_duration_ms = self.min_duration_ms.min(duration_ms);
        self.max_duration_ms = self.max_duration_ms.max(duration_ms);
    }

    /// Get average duration
    pub fn avg_duration_ms(&self) -> f64 {
        if self.call_count == 0 {
            0.0
        } else {
            self.total_duration_ms as f64 / self.call_count as f64
        }
    }

    /// Get total time percentage (if total_operations is known)
    pub fn time_percent(&self, total_ms: u64) -> f64 {
        if total_ms == 0 {
            0.0
        } else {
            (self.total_duration_ms as f64 / total_ms as f64) * 100.0
        }
    }
}

/// Performance profiler
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
    fn test_operation_profile_creation() {
        let p = OperationProfile::new("test");
        assert_eq!(p.call_count, 0);
        assert_eq!(p.avg_duration_ms(), 0.0);
    }

    #[test]
    fn test_operation_profile_add() {
        let mut p = OperationProfile::new("test");
        p.add(100);
        p.add(200);
        p.add(150);
        assert_eq!(p.call_count, 3);
        assert_eq!(p.total_duration_ms, 450);
        assert_eq!(p.min_duration_ms, 100);
        assert_eq!(p.max_duration_ms, 200);
    }

    #[test]
    fn test_operation_profile_avg() {
        let mut p = OperationProfile::new("test");
        p.add(100);
        p.add(200);
        assert_eq!(p.avg_duration_ms(), 150.0);
    }

    #[test]
    fn test_operation_profile_time_percent() {
        let mut p = OperationProfile::new("test");
        p.add(250);
        let pct = p.time_percent(1000);
        assert_eq!(pct, 25.0);
    }

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
