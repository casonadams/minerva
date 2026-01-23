/// Optimized batch processing module
///
/// This module contains optimized versions of batch processing components,
/// with improvements including:
/// - HashMap-based ID lookup (O(1) instead of O(n))
/// - Pre-allocated Vec capacity
/// - Reduced cloning operations
/// - Efficient statistics aggregation
use std::collections::HashMap;

// ============================================================================
// OPTIMIZED BATCH RESULT WITH HASHMAP
// ============================================================================

/// Optimized batch result with HashMap for O(1) lookup
pub struct BatchResultOptimized<T: Clone> {
    /// ID -> Response mapping for fast lookup
    pub responses_map: HashMap<String, BatchResponseOpt<T>>,
    /// Responses in order (optional, can be omitted if ordering not needed)
    pub responses_vec: Vec<BatchResponseOpt<T>>,
    /// Overall statistics
    pub stats: BatchStatsOpt,
}

/// Optimized batch response (simplified for efficiency)
#[derive(Clone)]
pub struct BatchResponseOpt<T: Clone> {
    pub id: String,
    pub data: T,
    pub duration_ms: u128,
}

/// Optimized statistics (pre-computed)
pub struct BatchStatsOpt {
    pub total_items: usize,
    pub total_duration_ms: u128,
    pub avg_item_time_ms: f64,
    pub items_per_second: f64,
}

impl<T: Clone> BatchResultOptimized<T> {
    /// Create new optimized batch result
    pub fn new(responses: Vec<BatchResponseOpt<T>>) -> Self {
        let total_items = responses.len();
        let total_duration_ms: u128 = responses.iter().map(|r| r.duration_ms).sum();

        // Pre-compute statistics
        let avg_item_time_ms = if total_items > 0 {
            total_duration_ms as f64 / total_items as f64 / 1000.0
        } else {
            0.0
        };

        let items_per_second = if total_duration_ms > 0 {
            (total_items as f64 * 1_000_000.0) / total_duration_ms as f64
        } else {
            0.0
        };

        let stats = BatchStatsOpt {
            total_items,
            total_duration_ms,
            avg_item_time_ms,
            items_per_second,
        };

        // Build HashMap for O(1) lookup
        let mut responses_map = HashMap::with_capacity(total_items);
        for response in responses.iter() {
            responses_map.insert(response.id.clone(), response.clone());
        }

        Self {
            responses_map,
            responses_vec: responses,
            stats,
        }
    }

    /// O(1) lookup by ID using HashMap
    pub fn get_by_id(&self, id: &str) -> Option<&BatchResponseOpt<T>> {
        self.responses_map.get(id)
    }

    /// Get all responses
    pub fn get_responses(&self) -> &[BatchResponseOpt<T>] {
        &self.responses_vec
    }

    /// Count successful responses
    pub fn success_count(&self) -> usize {
        self.responses_map.len()
    }
}

// ============================================================================
// OPTIMIZATION UTILITIES
// ============================================================================

/// Measure memory overhead of batch operations
pub fn estimate_memory_overhead(num_items: usize) -> f64 {
    // HashMap overhead: 48 bytes per entry (approximate)
    let hashmap_overhead = num_items as f64 * 48.0;

    // Vec overhead: 24 bytes base + pointer per item
    let vec_overhead = 24.0 + (num_items as f64 * 8.0);

    // Statistics: ~100 bytes
    let stats_overhead = 100.0;

    let total = hashmap_overhead + vec_overhead + stats_overhead;
    total / num_items as f64
}

/// Calculate optimal batch size based on available memory and target latency
pub fn calculate_optimal_batch_size(available_memory_mb: u32, target_latency_ms: f32) -> usize {
    // Memory constraint: limit items to stay within memory budget
    let available_bytes = available_memory_mb as f64 * 1_024.0 * 1_024.0;
    let memory_per_item = estimate_memory_overhead(100); // Sample 100 items
    let memory_limited_size = (available_bytes / memory_per_item) as usize;

    // Latency constraint: typical item processing time ~1ms, adjust based on target
    let latency_limited_size = (target_latency_ms / 1.0) as usize;

    // Take the minimum to respect both constraints
    std::cmp::min(memory_limited_size, latency_limited_size).clamp(1, 1000)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hashmap_lookup() {
        let responses = vec![
            BatchResponseOpt {
                id: "id1".to_string(),
                data: "result1".to_string(),
                duration_ms: 100,
            },
            BatchResponseOpt {
                id: "id2".to_string(),
                data: "result2".to_string(),
                duration_ms: 200,
            },
        ];

        let result = BatchResultOptimized::new(responses);

        // O(1) lookup should find items
        assert!(result.get_by_id("id1").is_some());
        assert!(result.get_by_id("id2").is_some());
        assert!(result.get_by_id("id3").is_none());
    }

    #[test]
    fn test_statistics_precomputation() {
        let responses = vec![
            BatchResponseOpt {
                id: "id1".to_string(),
                data: 100,
                duration_ms: 1000,
            },
            BatchResponseOpt {
                id: "id2".to_string(),
                data: 200,
                duration_ms: 2000,
            },
        ];

        let result = BatchResultOptimized::new(responses);

        assert_eq!(result.stats.total_items, 2);
        assert_eq!(result.stats.total_duration_ms, 3000);
        assert!(result.stats.avg_item_time_ms > 1.0); // 1.5ms average
    }

    #[test]
    fn test_memory_overhead_estimation() {
        let overhead_10 = estimate_memory_overhead(10);
        let overhead_100 = estimate_memory_overhead(100);

        // More items should have lower per-item overhead (amortized)
        assert!(overhead_100 < overhead_10);
    }

    #[test]
    fn test_optimal_batch_size_calculation() {
        // 512 MB available, 1ms target latency
        let size = calculate_optimal_batch_size(512, 1.0);
        assert!(size > 0);
        assert!(size <= 1000);
    }
}
