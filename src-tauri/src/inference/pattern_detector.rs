use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Usage pattern for a model
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct UsagePattern {
    pub model_id: String,
    pub access_count: u64,
    pub last_access: Instant,
    pub total_duration: Duration,
    pub avg_interval: Duration,
}

impl UsagePattern {
    /// Check if pattern is "hot" (frequently accessed)
    #[allow(dead_code)]
    pub fn is_hot(&self, threshold: u64) -> bool {
        self.access_count >= threshold
    }

    /// Get trend direction (increasing/decreasing/stable)
    #[allow(dead_code)]
    pub fn get_trend(&self, recent_count: u64) -> Trend {
        if recent_count > self.access_count / 2 {
            Trend::Increasing
        } else if recent_count < self.access_count / 4 {
            Trend::Decreasing
        } else {
            Trend::Stable
        }
    }
}

/// Access trend direction
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[allow(dead_code)]
pub enum Trend {
    Increasing,
    Stable,
    Decreasing,
}

/// Pattern detection result
#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(dead_code)]
pub struct PatternResult {
    pub model_id: String,
    pub should_preload: bool,
    pub priority: u32,
    pub reason: String,
}

/// Detects usage patterns for preloading decisions
#[derive(Debug)]
#[allow(dead_code)]
pub struct PatternDetector {
    patterns: HashMap<String, UsagePattern>,
    hot_threshold: u64,
    analysis_window: Duration,
    last_analysis: Option<Instant>,
}

impl PatternDetector {
    /// Create new pattern detector
    #[allow(dead_code)]
    pub fn new(hot_threshold: u64) -> Self {
        Self {
            patterns: HashMap::new(),
            hot_threshold,
            analysis_window: Duration::from_secs(300), // 5 minutes
            last_analysis: None,
        }
    }

    /// Record access to a model
    #[allow(dead_code)]
    pub fn record_access(&mut self, model_id: &str) {
        let now = Instant::now();
        self.patterns
            .entry(model_id.to_string())
            .and_modify(|p| {
                let elapsed = now.duration_since(p.last_access);
                p.access_count += 1;
                p.last_access = now;
                if p.access_count > 1 {
                    p.avg_interval = (p.avg_interval + elapsed) / 2;
                } else {
                    p.avg_interval = elapsed;
                }
            })
            .or_insert(UsagePattern {
                model_id: model_id.to_string(),
                access_count: 1,
                last_access: now,
                total_duration: Duration::from_secs(0),
                avg_interval: Duration::from_secs(0),
            });
    }

    /// Analyze patterns and get preload recommendations
    #[allow(dead_code)]
    pub fn analyze(&mut self) -> Vec<PatternResult> {
        let mut results = Vec::new();

        for (model_id, pattern) in &self.patterns {
            let should_preload = pattern.is_hot(self.hot_threshold);

            let priority = pattern.access_count.min(100) as u32;

            let reason = if should_preload {
                format!("Hot model: {} accesses", pattern.access_count)
            } else {
                format!("Cold model: {} accesses", pattern.access_count)
            };

            results.push(PatternResult {
                model_id: model_id.clone(),
                should_preload,
                priority,
                reason,
            });
        }

        results.sort_by(|a, b| b.priority.cmp(&a.priority));
        self.last_analysis = Some(Instant::now());

        tracing::info!(
            "Pattern analysis complete: {} models analyzed",
            results.len()
        );

        results
    }

    /// Get hot models (candidates for preloading)
    #[allow(dead_code)]
    pub fn get_hot_models(&self) -> Vec<&UsagePattern> {
        self.patterns
            .values()
            .filter(|p| p.is_hot(self.hot_threshold))
            .collect()
    }

    /// Get cold models (candidates for eviction)
    #[allow(dead_code)]
    pub fn get_cold_models(&self) -> Vec<&UsagePattern> {
        self.patterns
            .values()
            .filter(|p| !p.is_hot(self.hot_threshold))
            .collect()
    }

    /// Get model access count
    #[allow(dead_code)]
    pub fn get_access_count(&self, model_id: &str) -> u64 {
        self.patterns
            .get(model_id)
            .map(|p| p.access_count)
            .unwrap_or(0)
    }

    /// Clear patterns
    #[allow(dead_code)]
    pub fn clear(&mut self) {
        self.patterns.clear();
    }

    /// Get total unique models
    #[allow(dead_code)]
    pub fn total_models(&self) -> usize {
        self.patterns.len()
    }

    /// Check if analysis is needed
    #[allow(dead_code)]
    pub fn should_analyze(&self) -> bool {
        match self.last_analysis {
            None => true,
            Some(last) => last.elapsed() >= self.analysis_window,
        }
    }

    /// Get time since last analysis
    #[allow(dead_code)]
    pub fn time_since_analysis(&self) -> Option<Duration> {
        self.last_analysis.map(|t| t.elapsed())
    }

    /// Get pattern for a model
    #[allow(dead_code)]
    pub fn get_pattern(&self, model_id: &str) -> Option<&UsagePattern> {
        self.patterns.get(model_id)
    }

    /// Set hot threshold
    #[allow(dead_code)]
    pub fn set_hot_threshold(&mut self, threshold: u64) {
        self.hot_threshold = threshold;
    }
}

impl Default for PatternDetector {
    fn default() -> Self {
        Self::new(10)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_usage_pattern_is_hot() {
        let pattern = UsagePattern {
            model_id: "test".to_string(),
            access_count: 20,
            last_access: Instant::now(),
            total_duration: Duration::from_secs(0),
            avg_interval: Duration::from_secs(0),
        };

        assert!(pattern.is_hot(10));
        assert!(!pattern.is_hot(30));
    }

    #[test]
    fn test_usage_pattern_trend() {
        let pattern = UsagePattern {
            model_id: "test".to_string(),
            access_count: 100,
            last_access: Instant::now(),
            total_duration: Duration::from_secs(0),
            avg_interval: Duration::from_secs(0),
        };

        let trend = pattern.get_trend(60);
        assert!(matches!(trend, Trend::Increasing));

        let trend = pattern.get_trend(20);
        assert!(matches!(trend, Trend::Decreasing));
    }

    #[test]
    fn test_pattern_detector_creation() {
        let detector = PatternDetector::new(10);
        assert_eq!(detector.total_models(), 0);
    }

    #[test]
    fn test_pattern_detector_record_access() {
        let mut detector = PatternDetector::new(10);
        detector.record_access("model-1");
        detector.record_access("model-1");
        assert_eq!(detector.get_access_count("model-1"), 2);
    }

    #[test]
    fn test_pattern_detector_multiple_models() {
        let mut detector = PatternDetector::new(5);
        detector.record_access("model-1");
        detector.record_access("model-2");
        detector.record_access("model-1");
        assert_eq!(detector.total_models(), 2);
    }

    #[test]
    fn test_pattern_detector_hot_models() {
        let mut detector = PatternDetector::new(5);
        detector.record_access("model-1");
        detector.record_access("model-1");
        detector.record_access("model-1");
        detector.record_access("model-1");
        detector.record_access("model-1");
        detector.record_access("model-1");
        detector.record_access("model-2");

        let hot = detector.get_hot_models();
        assert_eq!(hot.len(), 1);
    }

    #[test]
    fn test_pattern_detector_cold_models() {
        let mut detector = PatternDetector::new(5);
        detector.record_access("model-1");
        detector.record_access("model-2");

        let cold = detector.get_cold_models();
        assert_eq!(cold.len(), 2);
    }

    #[test]
    fn test_pattern_detector_analyze() {
        let mut detector = PatternDetector::new(3);
        detector.record_access("model-1");
        detector.record_access("model-1");
        detector.record_access("model-1");
        detector.record_access("model-1");

        let results = detector.analyze();
        assert!(!results.is_empty());
    }

    #[test]
    fn test_pattern_detector_clear() {
        let mut detector = PatternDetector::new(5);
        detector.record_access("model-1");
        detector.clear();
        assert_eq!(detector.total_models(), 0);
    }

    #[test]
    fn test_pattern_detector_should_analyze() {
        let detector = PatternDetector::new(5);
        assert!(detector.should_analyze());
    }

    #[test]
    fn test_pattern_detector_set_threshold() {
        let mut detector = PatternDetector::new(10);
        detector.set_hot_threshold(5);
        detector.record_access("model-1");
        detector.record_access("model-1");
        detector.record_access("model-1");
        detector.record_access("model-1");
        detector.record_access("model-1");
        assert!(detector.get_pattern("model-1").unwrap().is_hot(5));
    }

    #[test]
    fn test_trend_enum_variants() {
        let trends = [Trend::Increasing, Trend::Stable, Trend::Decreasing];
        for trend in trends {
            assert!(matches!(
                trend,
                Trend::Increasing | Trend::Stable | Trend::Decreasing
            ));
        }
    }

    #[test]
    fn test_pattern_result_creation() {
        let result = PatternResult {
            model_id: "test".to_string(),
            should_preload: true,
            priority: 50,
            reason: "Hot model".to_string(),
        };

        assert!(result.should_preload);
        assert_eq!(result.priority, 50);
    }
}
