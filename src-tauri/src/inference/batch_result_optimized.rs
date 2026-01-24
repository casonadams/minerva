/// Optimized batch processing with HashMap-based O(1) lookup
use std::collections::HashMap;

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
        assert!(result.stats.avg_item_time_ms > 1.0);
    }
}
