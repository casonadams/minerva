/// Optimized batch processing module
///
/// This module contains optimized versions of batch processing components,
/// with improvements including:
/// - HashMap-based ID lookup (O(1) instead of O(n))
/// - Pre-allocated Vec capacity
/// - Reduced cloning operations
/// - Efficient statistics aggregation
pub use crate::inference::batch_result_optimized::{
    BatchResponseOpt, BatchResultOptimized, BatchStatsOpt,
};
pub use crate::inference::optimization_utils::{
    calculate_optimal_batch_size, estimate_memory_overhead,
};
