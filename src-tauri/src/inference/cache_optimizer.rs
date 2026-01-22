use crate::error::MinervaResult;
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};

/// Memory system information
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct SystemMemory {
    pub total_mb: u64,
    pub available_mb: u64,
    pub used_mb: u64,
}

impl SystemMemory {
    /// Get available memory percentage
    #[allow(dead_code)]
    pub fn available_percent(&self) -> f32 {
        if self.total_mb == 0 {
            0.0
        } else {
            (self.available_mb as f32 / self.total_mb as f32) * 100.0
        }
    }

    /// Check if under memory pressure (>80% used)
    #[allow(dead_code)]
    pub fn under_pressure(&self) -> bool {
        self.available_percent() < 20.0
    }
}

/// Cache optimization strategies
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default)]
#[allow(dead_code)]
pub enum OptimizationStrategy {
    /// Conservative: Keep 60% of available memory free
    Conservative,
    /// Balanced: Keep 40% of available memory free (default)
    #[default]
    Balanced,
    /// Aggressive: Keep 20% of available memory free
    Aggressive,
}

/// Cache optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(dead_code)]
pub struct OptimizationConfig {
    pub strategy: OptimizationStrategy,
    pub min_cache_mb: u64,
    pub max_cache_mb: u64,
    pub update_interval_ms: u64,
    pub auto_optimize: bool,
}

impl Default for OptimizationConfig {
    fn default() -> Self {
        Self {
            strategy: OptimizationStrategy::Balanced,
            min_cache_mb: 1000,
            max_cache_mb: 50000,
            update_interval_ms: 5000,
            auto_optimize: true,
        }
    }
}

/// Optimization statistics
#[derive(Debug, Clone, Default)]
#[allow(dead_code)]
pub struct OptimizationStats {
    pub total_optimizations: u64,
    pub cache_size_increases: u64,
    pub cache_size_decreases: u64,
    pub last_optimization: Option<Instant>,
    pub avg_cache_size_mb: u64,
}

/// Cache optimizer for dynamic sizing
#[derive(Debug)]
#[allow(dead_code)]
pub struct CacheOptimizer {
    config: OptimizationConfig,
    stats: OptimizationStats,
    current_cache_size_mb: u64,
}

impl CacheOptimizer {
    /// Create new optimizer with default config
    #[allow(dead_code)]
    pub fn new() -> Self {
        Self {
            config: OptimizationConfig::default(),
            stats: OptimizationStats::default(),
            current_cache_size_mb: 3000,
        }
    }

    /// Create with custom configuration
    #[allow(dead_code)]
    pub fn with_config(config: OptimizationConfig) -> Self {
        Self {
            config,
            stats: OptimizationStats::default(),
            current_cache_size_mb: 3000,
        }
    }

    /// Calculate recommended cache size based on system memory
    #[allow(dead_code)]
    pub fn calculate_optimal_size(&self, system_memory: &SystemMemory) -> u64 {
        let reserved_percent = match self.config.strategy {
            OptimizationStrategy::Conservative => 60,
            OptimizationStrategy::Balanced => 40,
            OptimizationStrategy::Aggressive => 20,
        };

        let reserved_mb =
            (system_memory.total_mb as f32 * (reserved_percent as f32 / 100.0)) as u64;
        let available_for_cache = system_memory.available_mb.saturating_sub(reserved_mb);

        available_for_cache
            .max(self.config.min_cache_mb)
            .min(self.config.max_cache_mb)
    }

    /// Optimize cache size based on system state
    #[allow(dead_code)]
    pub fn optimize(&mut self, system_memory: &SystemMemory) -> MinervaResult<Option<u64>> {
        if !self.config.auto_optimize {
            return Ok(None);
        }

        let optimal_size = self.calculate_optimal_size(system_memory);
        let old_size = self.current_cache_size_mb;

        if optimal_size != old_size {
            if optimal_size > old_size {
                self.stats.cache_size_increases += 1;
            } else {
                self.stats.cache_size_decreases += 1;
            }

            self.current_cache_size_mb = optimal_size;
            self.stats.total_optimizations += 1;
            self.stats.last_optimization = Some(Instant::now());
            self.stats.avg_cache_size_mb = (self.stats.avg_cache_size_mb + optimal_size) / 2;

            tracing::info!("Cache optimized: {} MB -> {} MB", old_size, optimal_size);

            Ok(Some(optimal_size))
        } else {
            Ok(None)
        }
    }

    /// Get current cache size
    #[allow(dead_code)]
    pub fn current_size(&self) -> u64 {
        self.current_cache_size_mb
    }

    /// Set cache size (respects min/max bounds)
    #[allow(dead_code)]
    pub fn set_size(&mut self, size_mb: u64) {
        let bounded = size_mb
            .max(self.config.min_cache_mb)
            .min(self.config.max_cache_mb);
        self.current_cache_size_mb = bounded;
    }

    /// Get optimization statistics
    #[allow(dead_code)]
    pub fn stats(&self) -> OptimizationStats {
        self.stats.clone()
    }

    /// Get configuration
    #[allow(dead_code)]
    pub fn config(&self) -> &OptimizationConfig {
        &self.config
    }

    /// Update configuration
    #[allow(dead_code)]
    pub fn set_config(&mut self, config: OptimizationConfig) {
        self.config = config;
    }

    /// Get time since last optimization
    #[allow(dead_code)]
    pub fn time_since_last_optimization(&self) -> Option<Duration> {
        self.stats.last_optimization.map(|t| t.elapsed())
    }

    /// Check if optimization is due
    #[allow(dead_code)]
    pub fn should_optimize(&self) -> bool {
        match self.stats.last_optimization {
            None => true,
            Some(last) => last.elapsed().as_millis() >= self.config.update_interval_ms as u128,
        }
    }
}

impl Default for CacheOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_system_memory_available_percent() {
        let mem = SystemMemory {
            total_mb: 10000,
            available_mb: 5000,
            used_mb: 5000,
        };
        assert_eq!(mem.available_percent(), 50.0);
    }

    #[test]
    fn test_system_memory_under_pressure() {
        let mem_low = SystemMemory {
            total_mb: 10000,
            available_mb: 1000,
            used_mb: 9000,
        };
        assert!(mem_low.under_pressure());

        let mem_ok = SystemMemory {
            total_mb: 10000,
            available_mb: 5000,
            used_mb: 5000,
        };
        assert!(!mem_ok.under_pressure());
    }

    #[test]
    fn test_optimization_strategy_default() {
        let strategy = OptimizationStrategy::default();
        assert!(matches!(strategy, OptimizationStrategy::Balanced));
    }

    #[test]
    fn test_optimization_config_default() {
        let config = OptimizationConfig::default();
        assert!(config.auto_optimize);
        assert_eq!(config.min_cache_mb, 1000);
        assert_eq!(config.max_cache_mb, 50000);
    }

    #[test]
    fn test_cache_optimizer_creation() {
        let optimizer = CacheOptimizer::new();
        assert_eq!(optimizer.current_size(), 3000);
    }

    #[test]
    fn test_cache_optimizer_calculate_optimal_size() {
        let optimizer = CacheOptimizer::new();
        let mem = SystemMemory {
            total_mb: 16000,
            available_mb: 8000,
            used_mb: 8000,
        };

        let optimal = optimizer.calculate_optimal_size(&mem);
        assert!(optimal > 0);
        assert!(optimal <= optimizer.config.max_cache_mb);
    }

    #[test]
    fn test_cache_optimizer_conservative_strategy() {
        let config = OptimizationConfig {
            strategy: OptimizationStrategy::Conservative,
            ..Default::default()
        };
        let optimizer = CacheOptimizer::with_config(config);

        let mem = SystemMemory {
            total_mb: 16000,
            available_mb: 8000,
            used_mb: 8000,
        };

        let optimal = optimizer.calculate_optimal_size(&mem);
        assert!(optimal > 0);
    }

    #[test]
    fn test_cache_optimizer_aggressive_strategy() {
        let config = OptimizationConfig {
            strategy: OptimizationStrategy::Aggressive,
            ..Default::default()
        };
        let optimizer = CacheOptimizer::with_config(config);

        let mem = SystemMemory {
            total_mb: 16000,
            available_mb: 8000,
            used_mb: 8000,
        };

        let optimal = optimizer.calculate_optimal_size(&mem);
        assert!(optimal > 0);
    }

    #[test]
    fn test_cache_optimizer_set_size() {
        let mut optimizer = CacheOptimizer::new();
        optimizer.set_size(5000);
        assert_eq!(optimizer.current_size(), 5000);
    }

    #[test]
    fn test_cache_optimizer_set_size_respects_bounds() {
        let mut optimizer = CacheOptimizer::new();
        optimizer.set_size(100000);
        assert!(optimizer.current_size() <= optimizer.config.max_cache_mb);
    }

    #[test]
    fn test_cache_optimizer_stats() {
        let optimizer = CacheOptimizer::new();
        let stats = optimizer.stats();
        assert_eq!(stats.total_optimizations, 0);
    }

    #[test]
    fn test_cache_optimizer_should_optimize() {
        let optimizer = CacheOptimizer::new();
        assert!(optimizer.should_optimize());
    }

    #[test]
    fn test_optimization_strategy_all_variants() {
        let strategies = [
            OptimizationStrategy::Conservative,
            OptimizationStrategy::Balanced,
            OptimizationStrategy::Aggressive,
        ];

        for strategy in strategies {
            assert!(matches!(
                strategy,
                OptimizationStrategy::Conservative
                    | OptimizationStrategy::Balanced
                    | OptimizationStrategy::Aggressive
            ));
        }
    }

    #[test]
    fn test_system_memory_zero_total() {
        let mem = SystemMemory {
            total_mb: 0,
            available_mb: 0,
            used_mb: 0,
        };
        assert_eq!(mem.available_percent(), 0.0);
    }

    #[test]
    fn test_optimization_stats_default() {
        let stats = OptimizationStats::default();
        assert_eq!(stats.total_optimizations, 0);
        assert!(stats.last_optimization.is_none());
    }
}
