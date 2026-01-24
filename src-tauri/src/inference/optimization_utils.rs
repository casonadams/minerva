/// Batch optimization utilities for memory and latency
///
/// Provides functions to estimate memory overhead and calculate optimal batch sizes

/// Estimate memory overhead per item in batch operations
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
    let memory_per_item = estimate_memory_overhead(100);
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
