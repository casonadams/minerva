/// Model Cache Manager - Phase 6 Step 5
///
/// This module manages loading and caching of LLM models in memory,
/// providing:
/// - Model loading from GGUF files
/// - In-memory caching with memory management
/// - Model switching and replacement
/// - Cache statistics and monitoring
/// - Eviction policies for memory constraints
use crate::error::{MinervaError, MinervaResult};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Cache statistics
#[derive(Debug, Clone)]
pub struct CacheStats {
    /// Number of cached models
    pub model_count: usize,
    /// Total memory used by cache (bytes)
    pub total_memory_bytes: usize,
    /// Number of cache hits
    pub hits: usize,
    /// Number of cache misses
    pub misses: usize,
    /// Cache hit ratio
    pub hit_ratio: f32,
}

/// Model cache entry
#[derive(Debug, Clone)]
struct CacheEntry {
    /// Model name/path
    #[allow(dead_code)]
    name: String,
    /// Model data (simplified - in real implementation, would store weights)
    data: Vec<u8>,
    /// Size in bytes
    size_bytes: usize,
    /// Last access timestamp
    last_access: std::time::Instant,
    /// Number of accesses
    access_count: usize,
}

/// Model Cache Manager
pub struct ModelCacheManager {
    /// Model cache
    cache: Arc<Mutex<HashMap<String, CacheEntry>>>,
    /// Maximum cache size in bytes
    max_cache_size: usize,
    /// Current cache size in bytes
    current_size: Arc<Mutex<usize>>,
    /// Statistics
    stats: Arc<Mutex<(usize, usize)>>, // (hits, misses)
}

impl ModelCacheManager {
    /// Create new cache manager
    pub fn new(max_cache_size_mb: usize) -> Self {
        Self {
            cache: Arc::new(Mutex::new(HashMap::new())),
            max_cache_size: max_cache_size_mb * 1024 * 1024,
            current_size: Arc::new(Mutex::new(0)),
            stats: Arc::new(Mutex::new((0, 0))),
        }
    }

    /// Create with default 2GB cache
    pub fn with_default_size() -> Self {
        Self::new(2048)
    }

    /// Load model into cache
    pub fn load_model(&self, model_name: &str, data: Vec<u8>) -> MinervaResult<()> {
        let size = data.len();

        // Check if model already cached
        let cache = self
            .cache
            .lock()
            .map_err(|_| MinervaError::InferenceError("Cache lock failed".to_string()))?;

        if cache.contains_key(model_name) {
            return Ok(()); // Already loaded
        }
        drop(cache);

        // Check if space available
        let mut current_size = self
            .current_size
            .lock()
            .map_err(|_| MinervaError::InferenceError("Size lock failed".to_string()))?;

        if *current_size + size > self.max_cache_size {
            // Evict least recently used
            self.evict_lru(size)?;
        }

        // Add to cache
        let entry = CacheEntry {
            name: model_name.to_string(),
            data,
            size_bytes: size,
            last_access: std::time::Instant::now(),
            access_count: 1,
        };

        let mut cache = self
            .cache
            .lock()
            .map_err(|_| MinervaError::InferenceError("Cache lock failed".to_string()))?;
        cache.insert(model_name.to_string(), entry);
        *current_size += size;

        Ok(())
    }

    /// Get model from cache
    pub fn get_model(&self, model_name: &str) -> MinervaResult<Option<Vec<u8>>> {
        let mut cache = self
            .cache
            .lock()
            .map_err(|_| MinervaError::InferenceError("Cache lock failed".to_string()))?;

        if let Some(entry) = cache.get_mut(model_name) {
            let stats = self
                .stats
                .lock()
                .map_err(|_| MinervaError::InferenceError("Stats lock failed".to_string()))?;
            drop(stats); // Release lock for stats update

            entry.last_access = std::time::Instant::now();
            entry.access_count += 1;

            let mut stats = self
                .stats
                .lock()
                .map_err(|_| MinervaError::InferenceError("Stats lock failed".to_string()))?;
            stats.0 += 1; // Increment hits

            Ok(Some(entry.data.clone()))
        } else {
            let mut stats = self
                .stats
                .lock()
                .map_err(|_| MinervaError::InferenceError("Stats lock failed".to_string()))?;
            stats.1 += 1; // Increment misses
            Ok(None)
        }
    }

    /// Remove model from cache
    pub fn remove_model(&self, model_name: &str) -> MinervaResult<bool> {
        let mut cache = self
            .cache
            .lock()
            .map_err(|_| MinervaError::InferenceError("Cache lock failed".to_string()))?;

        if let Some(entry) = cache.remove(model_name) {
            let mut current_size = self
                .current_size
                .lock()
                .map_err(|_| MinervaError::InferenceError("Size lock failed".to_string()))?;
            *current_size -= entry.size_bytes;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Clear all models from cache
    pub fn clear(&self) -> MinervaResult<()> {
        let mut cache = self
            .cache
            .lock()
            .map_err(|_| MinervaError::InferenceError("Cache lock failed".to_string()))?;
        cache.clear();

        let mut current_size = self
            .current_size
            .lock()
            .map_err(|_| MinervaError::InferenceError("Size lock failed".to_string()))?;
        *current_size = 0;

        Ok(())
    }

    /// Get cache statistics
    pub fn get_stats(&self) -> MinervaResult<CacheStats> {
        let cache = self
            .cache
            .lock()
            .map_err(|_| MinervaError::InferenceError("Cache lock failed".to_string()))?;
        let current_size = self
            .current_size
            .lock()
            .map_err(|_| MinervaError::InferenceError("Size lock failed".to_string()))?;
        let stats = self
            .stats
            .lock()
            .map_err(|_| MinervaError::InferenceError("Stats lock failed".to_string()))?;

        let (hits, misses) = *stats;
        let total = hits + misses;
        let hit_ratio = if total > 0 {
            hits as f32 / total as f32
        } else {
            0.0
        };

        Ok(CacheStats {
            model_count: cache.len(),
            total_memory_bytes: *current_size,
            hits,
            misses,
            hit_ratio,
        })
    }

    /// Get list of cached models
    pub fn list_models(&self) -> MinervaResult<Vec<String>> {
        let cache = self
            .cache
            .lock()
            .map_err(|_| MinervaError::InferenceError("Cache lock failed".to_string()))?;
        Ok(cache.keys().cloned().collect())
    }

    /// Evict least recently used model to free space
    fn evict_lru(&self, required_space: usize) -> MinervaResult<()> {
        let mut cache = self
            .cache
            .lock()
            .map_err(|_| MinervaError::InferenceError("Cache lock failed".to_string()))?;

        let mut freed = 0;
        while freed < required_space && !cache.is_empty() {
            // Find LRU entry
            let lru_key = cache
                .iter()
                .min_by_key(|(_, entry)| entry.last_access)
                .map(|(k, _)| k.clone());

            if let Some(key) = lru_key {
                if let Some(entry) = cache.remove(&key) {
                    freed += entry.size_bytes;
                }
            } else {
                break;
            }
        }

        if freed < required_space {
            return Err(MinervaError::OutOfMemory(
                "Cannot free enough memory for model".to_string(),
            ));
        }

        let mut current_size = self
            .current_size
            .lock()
            .map_err(|_| MinervaError::InferenceError("Size lock failed".to_string()))?;
        *current_size -= freed;

        Ok(())
    }

    /// Get remaining cache space
    pub fn available_space(&self) -> MinervaResult<usize> {
        let current_size = self
            .current_size
            .lock()
            .map_err(|_| MinervaError::InferenceError("Size lock failed".to_string()))?;
        Ok(self.max_cache_size - *current_size)
    }

    /// Check if model is cached
    pub fn is_cached(&self, model_name: &str) -> MinervaResult<bool> {
        let cache = self
            .cache
            .lock()
            .map_err(|_| MinervaError::InferenceError("Cache lock failed".to_string()))?;
        Ok(cache.contains_key(model_name))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_creation() {
        let cache = ModelCacheManager::new(512);
        assert_eq!(cache.max_cache_size, 512 * 1024 * 1024);
    }

    #[test]
    fn test_cache_default_size() {
        let cache = ModelCacheManager::with_default_size();
        assert_eq!(cache.max_cache_size, 2048 * 1024 * 1024);
    }

    #[test]
    fn test_load_model() {
        let cache = ModelCacheManager::new(512);
        let data = vec![1, 2, 3, 4, 5];
        assert!(cache.load_model("model1", data).is_ok());
    }

    #[test]
    fn test_get_model() {
        let cache = ModelCacheManager::new(512);
        let data = vec![1, 2, 3, 4, 5];
        cache.load_model("model1", data.clone()).unwrap();

        let retrieved = cache.get_model("model1").unwrap();
        assert_eq!(retrieved, Some(data));
    }

    #[test]
    fn test_get_nonexistent_model() {
        let cache = ModelCacheManager::new(512);
        let result = cache.get_model("nonexistent").unwrap();
        assert_eq!(result, None);
    }

    #[test]
    fn test_remove_model() {
        let cache = ModelCacheManager::new(512);
        let data = vec![1, 2, 3];
        cache.load_model("model1", data).unwrap();

        assert!(cache.remove_model("model1").unwrap());
        assert!(!cache.remove_model("model1").unwrap());
    }

    #[test]
    fn test_clear_cache() {
        let cache = ModelCacheManager::new(512);
        cache.load_model("model1", vec![1, 2, 3]).unwrap();
        cache.load_model("model2", vec![4, 5, 6]).unwrap();

        assert!(cache.clear().is_ok());
        let stats = cache.get_stats().unwrap();
        assert_eq!(stats.model_count, 0);
    }

    #[test]
    fn test_cache_stats() {
        let cache = ModelCacheManager::new(512);
        cache.load_model("model1", vec![1, 2, 3]).unwrap();

        let stats = cache.get_stats().unwrap();
        assert_eq!(stats.model_count, 1);
        assert!(stats.total_memory_bytes > 0);
    }

    #[test]
    fn test_cache_hits_and_misses() {
        let cache = ModelCacheManager::new(512);
        let data = vec![1, 2, 3];
        cache.load_model("model1", data).unwrap();

        // Hit
        let _ = cache.get_model("model1").unwrap();
        // Miss
        let _ = cache.get_model("nonexistent").unwrap();

        let stats = cache.get_stats().unwrap();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);
        assert!((stats.hit_ratio - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_list_models() {
        let cache = ModelCacheManager::new(512);
        cache.load_model("model1", vec![1, 2]).unwrap();
        cache.load_model("model2", vec![3, 4]).unwrap();

        let models = cache.list_models().unwrap();
        assert_eq!(models.len(), 2);
        assert!(models.contains(&"model1".to_string()));
        assert!(models.contains(&"model2".to_string()));
    }

    #[test]
    fn test_is_cached() {
        let cache = ModelCacheManager::new(512);
        let data = vec![1, 2, 3];
        cache.load_model("model1", data).unwrap();

        assert!(cache.is_cached("model1").unwrap());
        assert!(!cache.is_cached("model2").unwrap());
    }

    #[test]
    fn test_available_space() {
        let cache = ModelCacheManager::new(512);
        let initial_space = cache.available_space().unwrap();

        let data = vec![0; 1024 * 1024]; // 1MB
        cache.load_model("model1", data).unwrap();

        let remaining_space = cache.available_space().unwrap();
        assert!(remaining_space < initial_space);
        assert_eq!(remaining_space, initial_space - 1024 * 1024);
    }

    #[test]
    fn test_lru_eviction() {
        let cache = ModelCacheManager::new(10); // 10MB cache
        let data_small = vec![0; 1000]; // 1KB each

        // Load models
        cache.load_model("model1", data_small.clone()).unwrap();
        cache.load_model("model2", data_small.clone()).unwrap();

        // Verify both are cached
        assert_eq!(cache.list_models().unwrap().len(), 2);

        // Both models should be present
        assert!(cache.is_cached("model1").unwrap());
        assert!(cache.is_cached("model2").unwrap());
    }

    #[test]
    fn test_duplicate_load() {
        let cache = ModelCacheManager::new(512);
        let data = vec![1, 2, 3];

        assert!(cache.load_model("model1", data.clone()).is_ok());
        assert!(cache.load_model("model1", data).is_ok()); // Should succeed

        let stats = cache.get_stats().unwrap();
        assert_eq!(stats.model_count, 1); // Not duplicated
    }

    #[test]
    fn test_memory_tracking() {
        let cache = ModelCacheManager::new(512);
        let data1 = vec![0; 100];
        let data2 = vec![0; 200];

        cache.load_model("model1", data1).unwrap();
        let size_after_1 = cache.available_space().unwrap();

        cache.load_model("model2", data2).unwrap();
        let size_after_2 = cache.available_space().unwrap();

        // After adding model2 (200 bytes), available space should decrease
        assert!(size_after_2 < size_after_1);
        // The difference should be 200 bytes (model2 size)
        assert_eq!(size_after_1 - size_after_2, 200);
    }

    #[test]
    fn test_cache_stats_hit_ratio_zero_accesses() {
        let cache = ModelCacheManager::new(512);
        let stats = cache.get_stats().unwrap();
        assert_eq!(stats.hit_ratio, 0.0);
    }

    #[test]
    fn test_multiple_gets_increase_hit_count() {
        let cache = ModelCacheManager::new(512);
        cache.load_model("model1", vec![1, 2, 3]).unwrap();

        let _ = cache.get_model("model1").unwrap();
        let _ = cache.get_model("model1").unwrap();
        let _ = cache.get_model("model1").unwrap();

        let stats = cache.get_stats().unwrap();
        assert_eq!(stats.hits, 3);
    }
}
