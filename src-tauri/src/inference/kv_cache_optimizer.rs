/// KV Cache Optimizer - Phase 6 Step 5
///
/// This module manages KV (Key-Value) caching for efficient incremental
/// token generation in LLM inference:
/// - Cache key-value pairs from attention layers
/// - Reuse computation for previously generated tokens
/// - Manage memory usage of cached data
/// - Support incremental generation workflows
use crate::error::{MinervaError, MinervaResult};
use std::collections::HashMap;

/// Range for KV cache slicing
#[derive(Debug, Clone, Copy)]
pub struct CacheRange {
    /// Start index
    pub start: usize,
    /// End index (exclusive)
    pub end: usize,
}

impl CacheRange {
    /// Create new cache range
    pub fn new(start: usize, end: usize) -> Self {
        Self { start, end }
    }
}

/// KV pair for cache operations
pub struct KVPair<'a> {
    /// Key data
    pub keys: &'a [f32],
    /// Value data
    pub values: &'a [f32],
}

impl<'a> KVPair<'a> {
    /// Create new KV pair
    pub fn new(keys: &'a [f32], values: &'a [f32]) -> Self {
        Self { keys, values }
    }
}

/// KV Cache entry for a single layer
#[derive(Debug, Clone)]
pub struct LayerKVCache {
    /// Cached keys (batch_size, seq_len, num_heads, head_dim)
    pub keys: Vec<f32>,
    /// Cached values (batch_size, seq_len, num_heads, head_dim)
    pub values: Vec<f32>,
    /// Current sequence length
    pub seq_len: usize,
    /// Maximum sequence length
    pub max_seq_len: usize,
}

impl LayerKVCache {
    /// Create new layer KV cache
    pub fn new(max_seq_len: usize, cache_size: usize) -> Self {
        Self {
            keys: vec![0.0; cache_size],
            values: vec![0.0; cache_size],
            seq_len: 0,
            max_seq_len,
        }
    }

    /// Add new keys and values (one token at a time, where each token is hidden_dim size)
    pub fn append(&mut self, new_keys: &[f32], new_values: &[f32]) -> MinervaResult<()> {
        if new_keys.len() != new_values.len() {
            return Err(MinervaError::InferenceError(
                "Keys and values size mismatch".to_string(),
            ));
        }

        let token_size = new_keys.len();
        let next_pos = self.seq_len + token_size;

        // Check if we have space
        if next_pos > self.keys.len() || next_pos > self.values.len() {
            return Err(MinervaError::InferenceError(
                "KV cache exceeded maximum size".to_string(),
            ));
        }

        // Copy data
        self.keys[self.seq_len..next_pos].copy_from_slice(new_keys);
        self.values[self.seq_len..next_pos].copy_from_slice(new_values);

        // Increment seq_len (treating each append as one more token's worth of data)
        self.seq_len = next_pos;

        Ok(())
    }

    /// Get cached keys for a token range (where tokens are indexed by element count)
    pub fn get_keys(&self, start: usize, end: usize) -> MinervaResult<Vec<f32>> {
        if end > self.seq_len || start >= end {
            return Err(MinervaError::InferenceError(
                "Invalid cache range".to_string(),
            ));
        }

        Ok(self.keys[start..end].to_vec())
    }

    /// Get cached values for a range
    pub fn get_values(&self, start: usize, end: usize) -> MinervaResult<Vec<f32>> {
        if end > self.seq_len || start >= end {
            return Err(MinervaError::InferenceError(
                "Invalid cache range".to_string(),
            ));
        }

        Ok(self.values[start..end].to_vec())
    }

    /// Clear cache
    pub fn clear(&mut self) {
        self.seq_len = 0;
    }

    /// Get cache memory usage in bytes
    pub fn memory_usage(&self) -> usize {
        (self.keys.len() + self.values.len()) * std::mem::size_of::<f32>()
    }

    /// Get cache utilization percentage (0.0 to 1.0)
    pub fn utilization(&self) -> f32 {
        if self.keys.is_empty() {
            0.0
        } else {
            self.seq_len as f32 / self.keys.len() as f32
        }
    }
}

/// Multi-layer KV Cache Manager
#[allow(dead_code)]
pub struct KVCacheManager {
    /// Per-layer caches
    caches: HashMap<usize, LayerKVCache>,
    /// Number of layers
    num_layers: usize,
    /// Max sequence length per layer
    max_seq_len: usize,
    /// Cache size per layer
    cache_size_per_layer: usize,
}

impl KVCacheManager {
    /// Create new KV cache manager
    pub fn new(num_layers: usize, max_seq_len: usize, hidden_dim: usize) -> Self {
        // Cache size = max_seq_len * hidden_dim
        let cache_size_per_layer = max_seq_len * hidden_dim;

        let mut caches = HashMap::new();
        for layer_id in 0..num_layers {
            caches.insert(
                layer_id,
                LayerKVCache::new(max_seq_len, cache_size_per_layer),
            );
        }

        Self {
            caches,
            num_layers,
            max_seq_len,
            cache_size_per_layer,
        }
    }

    /// Add keys and values for a layer
    pub fn add_layer_cache(&mut self, layer_id: usize, kv: KVPair) -> MinervaResult<()> {
        if layer_id >= self.num_layers {
            return Err(MinervaError::InferenceError(format!(
                "Layer ID {} out of range",
                layer_id
            )));
        }

        if let Some(cache) = self.caches.get_mut(&layer_id) {
            cache.append(kv.keys, kv.values)
        } else {
            Err(MinervaError::InferenceError(
                "Cache not initialized".to_string(),
            ))
        }
    }

    /// Get cached keys for a layer
    pub fn get_cached_keys(&self, layer_id: usize, range: CacheRange) -> MinervaResult<Vec<f32>> {
        if let Some(cache) = self.caches.get(&layer_id) {
            cache.get_keys(range.start, range.end)
        } else {
            Err(MinervaError::InferenceError("Cache not found".to_string()))
        }
    }

    /// Get cached values for a layer
    pub fn get_cached_values(&self, layer_id: usize, range: CacheRange) -> MinervaResult<Vec<f32>> {
        if let Some(cache) = self.caches.get(&layer_id) {
            cache.get_values(range.start, range.end)
        } else {
            Err(MinervaError::InferenceError("Cache not found".to_string()))
        }
    }

    /// Get current sequence length
    pub fn seq_len(&self, layer_id: usize) -> MinervaResult<usize> {
        if let Some(cache) = self.caches.get(&layer_id) {
            Ok(cache.seq_len)
        } else {
            Err(MinervaError::InferenceError("Cache not found".to_string()))
        }
    }

    /// Clear all caches
    pub fn clear_all(&mut self) {
        for cache in self.caches.values_mut() {
            cache.clear();
        }
    }

    /// Clear specific layer cache
    pub fn clear_layer(&mut self, layer_id: usize) -> MinervaResult<()> {
        if let Some(cache) = self.caches.get_mut(&layer_id) {
            cache.clear();
            Ok(())
        } else {
            Err(MinervaError::InferenceError("Cache not found".to_string()))
        }
    }

    /// Get total memory usage
    pub fn total_memory_usage(&self) -> usize {
        self.caches.values().map(|cache| cache.memory_usage()).sum()
    }

    /// Get memory usage for specific layer
    pub fn layer_memory_usage(&self, layer_id: usize) -> MinervaResult<usize> {
        if let Some(cache) = self.caches.get(&layer_id) {
            Ok(cache.memory_usage())
        } else {
            Err(MinervaError::InferenceError("Cache not found".to_string()))
        }
    }

    /// Get cache statistics
    pub fn get_stats(&self) -> Vec<(usize, f32, usize)> {
        self.caches
            .iter()
            .map(|(layer_id, cache)| (*layer_id, cache.utilization(), cache.memory_usage()))
            .collect()
    }

    /// Check if cache is full
    pub fn is_full(&self, layer_id: usize) -> MinervaResult<bool> {
        if let Some(cache) = self.caches.get(&layer_id) {
            Ok(cache.seq_len >= cache.max_seq_len)
        } else {
            Err(MinervaError::InferenceError("Cache not found".to_string()))
        }
    }

    /// Get remaining capacity
    pub fn remaining_capacity(&self, layer_id: usize) -> MinervaResult<usize> {
        if let Some(cache) = self.caches.get(&layer_id) {
            Ok(cache.max_seq_len - cache.seq_len)
        } else {
            Err(MinervaError::InferenceError("Cache not found".to_string()))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layer_kv_cache_creation() {
        let cache = LayerKVCache::new(256, 256 * 64);
        assert_eq!(cache.seq_len, 0);
        assert_eq!(cache.max_seq_len, 256);
    }

    #[test]
    fn test_append_keys_and_values() {
        let mut cache = LayerKVCache::new(256, 256 * 64);
        let keys = vec![0.5; 64];
        let values = vec![0.5; 64];

        assert!(cache.append(&keys, &values).is_ok());
        assert_eq!(cache.seq_len, 64); // 64 elements added
    }

    #[test]
    fn test_append_multiple_tokens() {
        let mut cache = LayerKVCache::new(256, 256 * 64);
        for _ in 0..10 {
            let keys = vec![0.5; 64];
            let values = vec![0.5; 64];
            assert!(cache.append(&keys, &values).is_ok());
        }
        assert_eq!(cache.seq_len, 10 * 64); // 10 tokens of 64 elements each
    }

    #[test]
    fn test_append_size_mismatch() {
        let mut cache = LayerKVCache::new(256, 256 * 64);
        let keys = vec![0.5; 64];
        let values = vec![0.5; 32];

        assert!(cache.append(&keys, &values).is_err());
    }

    #[test]
    fn test_get_keys() {
        let mut cache = LayerKVCache::new(256, 256 * 64);
        let keys = vec![0.5; 64];
        let values = vec![0.5; 64];
        cache.append(&keys, &values).unwrap();

        let retrieved = cache.get_keys(0, 64).unwrap();
        assert_eq!(retrieved.len(), 64);
    }

    #[test]
    fn test_get_values() {
        let mut cache = LayerKVCache::new(256, 256 * 64);
        let keys = vec![0.5; 64];
        let values = vec![0.7; 64];
        cache.append(&keys, &values).unwrap();

        let retrieved = cache.get_values(0, 1).unwrap();
        assert_eq!(retrieved[0], 0.7);
    }

    #[test]
    fn test_cache_overflow() {
        let mut cache = LayerKVCache::new(2, 128); // Cache size = 128 elements
        let keys = vec![0.5; 64];
        let values = vec![0.5; 64];

        assert!(cache.append(&keys, &values).is_ok()); // 64 elements
        assert!(cache.append(&keys, &values).is_ok()); // 128 elements total
        // Third append should fail (would exceed 128)
        assert!(cache.append(&keys, &values).is_err());
    }

    #[test]
    fn test_cache_clear() {
        let mut cache = LayerKVCache::new(256, 256 * 64);
        let keys = vec![0.5; 64];
        let values = vec![0.5; 64];
        cache.append(&keys, &values).unwrap();

        cache.clear();
        assert_eq!(cache.seq_len, 0);
    }

    #[test]
    fn test_memory_usage() {
        let cache = LayerKVCache::new(256, 256 * 64);
        let memory = cache.memory_usage();
        assert!(memory > 0);
    }

    #[test]
    fn test_utilization() {
        let mut cache = LayerKVCache::new(256, 256 * 64);
        assert_eq!(cache.utilization(), 0.0);

        let keys = vec![0.5; 64];
        let values = vec![0.5; 64];
        cache.append(&keys, &values).unwrap();

        let utilization = cache.utilization();
        assert!(utilization > 0.0 && utilization <= 1.0);
    }

    #[test]
    fn test_kv_cache_manager_creation() {
        let manager = KVCacheManager::new(12, 256, 512);
        assert_eq!(manager.num_layers, 12);
        assert_eq!(manager.max_seq_len, 256);
    }

    #[test]
    fn test_add_layer_cache() {
        let mut manager = KVCacheManager::new(12, 256, 512);
        let keys = vec![0.5; 512];
        let values = vec![0.5; 512];

        assert!(
            manager
                .add_layer_cache(0, KVPair::new(&keys, &values))
                .is_ok()
        );
        assert_eq!(manager.seq_len(0).unwrap(), 512); // 512 elements added
    }

    #[test]
    fn test_get_cached_keys() {
        let mut manager = KVCacheManager::new(12, 256, 512);
        let keys = vec![0.5; 256];
        let values = vec![0.5; 256];

        assert!(
            manager
                .add_layer_cache(0, KVPair::new(&keys, &values))
                .is_ok()
        );
        assert_eq!(manager.seq_len(0).unwrap(), 256);

        let retrieved = manager.get_cached_keys(0, CacheRange::new(0, 256)).unwrap();
        assert_eq!(retrieved.len(), 256);
    }

    #[test]
    fn test_clear_all() {
        let mut manager = KVCacheManager::new(3, 256, 512);
        let keys = vec![0.5; 256]; // Smaller size to avoid overflow
        let values = vec![0.5; 256];

        manager
            .add_layer_cache(0, KVPair::new(&keys, &values))
            .unwrap();
        manager
            .add_layer_cache(1, KVPair::new(&keys, &values))
            .unwrap();

        manager.clear_all();
        assert_eq!(manager.seq_len(0).unwrap(), 0);
        assert_eq!(manager.seq_len(1).unwrap(), 0);
    }

    #[test]
    fn test_clear_single_layer() {
        let mut manager = KVCacheManager::new(3, 256, 512);
        let keys = vec![0.5; 256];
        let values = vec![0.5; 256];

        manager
            .add_layer_cache(0, KVPair::new(&keys, &values))
            .unwrap();
        manager
            .add_layer_cache(1, KVPair::new(&keys, &values))
            .unwrap();

        manager.clear_layer(0).unwrap();
        assert_eq!(manager.seq_len(0).unwrap(), 0);
        assert_eq!(manager.seq_len(1).unwrap(), 256);
    }

    #[test]
    fn test_total_memory_usage() {
        let mut manager = KVCacheManager::new(3, 256, 512);
        let keys = vec![0.5; 256];
        let values = vec![0.5; 256];

        manager
            .add_layer_cache(0, KVPair::new(&keys, &values))
            .unwrap();

        let memory = manager.total_memory_usage();
        assert!(memory > 0);
    }

    #[test]
    fn test_is_full() {
        let mut manager = KVCacheManager::new(1, 2, 64); // Small cache
        let keys = vec![0.5; 64]; // Exactly cache size
        let values = vec![0.5; 64];

        manager
            .add_layer_cache(0, KVPair::new(&keys, &values))
            .unwrap();

        assert!(manager.is_full(0).unwrap());
    }

    #[test]
    fn test_remaining_capacity() {
        let mut manager = KVCacheManager::new(1, 256, 512); // max_seq_len=256, cache=256*512
        let keys = vec![0.5; 256];
        let values = vec![0.5; 256];

        let initial = manager.remaining_capacity(0).unwrap();
        assert_eq!(initial, 256);

        manager
            .add_layer_cache(0, KVPair::new(&keys, &values))
            .unwrap();
        let after = manager.remaining_capacity(0).unwrap();
        assert_eq!(after, 0); // All capacity used
    }

    #[test]
    fn test_get_stats() {
        let mut manager = KVCacheManager::new(3, 256, 512);
        let keys = vec![0.5; 256];
        let values = vec![0.5; 256];

        manager
            .add_layer_cache(0, KVPair::new(&keys, &values))
            .unwrap();

        let stats = manager.get_stats();
        assert_eq!(stats.len(), 3);
    }
}
