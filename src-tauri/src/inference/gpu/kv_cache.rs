/// KV Cache for efficient token generation
///
/// Caches key and value tensors from previous forward passes to avoid
/// redundant computation during generation phase.
///
/// Without KV cache: O(n²) complexity (recompute all tokens)
/// With KV cache: O(n) complexity (only new token)
use ndarray::Array2;

/// KV Cache stores cached keys and values
/// For simplicity in Day 1, we're using 2D arrays
/// Production version would use 3D for batching
pub struct KVCache {
    pub k: Option<Vec<Array2<f32>>>,
    pub v: Option<Vec<Array2<f32>>>,
    pub seq_len: usize,
}

impl KVCache {
    /// Create new empty cache
    pub fn new() -> Self {
        Self {
            k: None,
            v: None,
            seq_len: 0,
        }
    }

    /// Append new key-value pairs to cache
    ///
    /// Returns true if cache was updated
    pub fn append(&mut self, _new_k: Array2<f32>, _new_v: Array2<f32>) -> bool {
        // Day 1 placeholder - KV cache will be implemented in optimization phase
        // For now, just track that we have new tokens
        self.seq_len += 1;
        true
    }

    /// Reset cache (for new sequence)
    pub fn reset(&mut self) {
        self.k = None;
        self.v = None;
        self.seq_len = 0;
    }

    /// Get current sequence length
    pub fn len(&self) -> usize {
        self.seq_len
    }

    /// Check if cache is empty
    pub fn is_empty(&self) -> bool {
        self.seq_len == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_kv_cache_creation() {
        let cache = KVCache::new();
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn test_kv_cache_append() {
        let mut cache = KVCache::new();

        // Create sample tensors: (seq=2, hidden=4)
        let k1 = Array2::zeros((2, 4));
        let v1 = Array2::zeros((2, 4));

        let updated = cache.append(k1, v1);
        assert!(updated);
        assert_eq!(cache.len(), 1);
        assert!(!cache.is_empty());
    }

    #[test]
    fn test_kv_cache_reset() {
        let mut cache = KVCache::new();
        let k1 = Array2::zeros((2, 4));
        let v1 = Array2::zeros((2, 4));
        cache.append(k1, v1);

        cache.reset();
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);
    }
}
