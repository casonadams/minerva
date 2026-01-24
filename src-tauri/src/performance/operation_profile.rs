/// Performance profile for a single operation
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
}
