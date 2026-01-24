use super::profiler::OperationProfile;
use std::cmp::Reverse;

/// Analyzes operation profiles and generates reports
pub struct ProfileAnalyzer;

impl ProfileAnalyzer {
    /// Get top N slowest operations by total time
    pub fn top_slowest(profiles: &[OperationProfile], n: usize) -> Vec<OperationProfile> {
        let mut sorted = profiles.to_vec();
        sorted.sort_by_key(|p| Reverse(p.total_duration_ms));
        sorted.into_iter().take(n).collect()
    }

    /// Generate performance report
    pub fn generate_report(profiles: &[OperationProfile]) -> String {
        if profiles.is_empty() {
            return "No profiling data".to_string();
        }

        let total_ms: u64 = profiles.iter().map(|p| p.total_duration_ms).sum();
        let mut report = format!("Performance Profile Report (Total: {}ms)\n", total_ms);
        report.push_str(&"=".repeat(80));
        report.push('\n');

        for profile in profiles.iter() {
            let pct = profile.time_percent(total_ms);
            report.push_str(&format!(
                "{:<30} {:>8} calls {:>10.2}ms avg {:>6.2}% [{} - {}]ms\n",
                profile.name,
                profile.call_count,
                profile.avg_duration_ms(),
                pct,
                profile.min_duration_ms,
                profile.max_duration_ms
            ));
        }

        report
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_top_slowest() {
        let mut p1 = OperationProfile::new("fast");
        p1.add(10);
        let mut p2 = OperationProfile::new("slow");
        p2.add(1000);
        let mut p3 = OperationProfile::new("medium");
        p3.add(500);

        let profiles = vec![p1, p2, p3];
        let top = ProfileAnalyzer::top_slowest(&profiles, 2);
        assert_eq!(top.len(), 2);
        assert_eq!(top[0].name, "slow");
    }

    #[test]
    fn test_generate_report() {
        let mut p1 = OperationProfile::new("op1");
        p1.add(100);
        let mut p2 = OperationProfile::new("op2");
        p2.add(200);

        let profiles = vec![p1, p2];
        let report = ProfileAnalyzer::generate_report(&profiles);
        assert!(report.contains("op1"));
        assert!(report.contains("op2"));
        assert!(report.contains("Report"));
    }

    #[test]
    fn test_empty_report() {
        let profiles = vec![];
        let report = ProfileAnalyzer::generate_report(&profiles);
        assert_eq!(report, "No profiling data");
    }
}
