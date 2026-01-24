use std::time::Duration;

/// Metrics analysis and snapshot building
pub struct MetricsAnalyzer;

impl MetricsAnalyzer {
    /// Analyze response times and extract statistics
    pub fn analyze_times(times: &[Duration]) -> (f64, f64, f64, f64, f64, f64) {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_analyze_empty() {
        let (avg, min, max, p50, p95, p99) = MetricsAnalyzer::analyze_times(&[]);
        assert_eq!(avg, 0.0);
        assert_eq!(min, 0.0);
        assert_eq!(max, 0.0);
        assert_eq!(p50, 0.0);
        assert_eq!(p95, 0.0);
        assert_eq!(p99, 0.0);
    }

    #[test]
    fn test_analyze_percentiles() {
        let times: Vec<Duration> = (1..=100).map(|i| Duration::from_millis(i)).collect();
        let (avg, min, max, p50, p95, p99) = MetricsAnalyzer::analyze_times(&times);

        assert!(avg > 0.0);
        assert!(min >= 1.0);
        assert!(max <= 100.0);
        assert!(p50 >= p50);
        assert!(p95 >= p50);
        assert!(p99 >= p95);
    }
}
