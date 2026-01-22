use super::InferenceEngine;
use crate::error::{MinervaError, MinervaResult};
use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Instant;

/// Cache statistics for tracking performance
#[derive(Debug, Clone, Default)]
#[allow(dead_code)]
pub struct CacheStats {
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    pub preloads: u64,
}

impl CacheStats {
    /// Calculate hit rate as percentage
    #[allow(dead_code)]
    pub fn hit_rate(&self) -> f32 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            (self.hits as f32 / total as f32) * 100.0
        }
    }
}

/// Strategy for cache eviction when full
#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
pub enum EvictionPolicy {
    /// Least Recently Used
    Lru,
    /// Least Frequently Used
    Lfu,
    /// First In, First Out
    Fifo,
}

/// Preloading strategy for models
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub enum PreloadStrategy {
    /// Eager: load immediately
    Eager,
    /// Lazy: load on first use
    Lazy,
    /// Scheduled: load at specific time
    Scheduled,
}

/// Cached model entry with metadata
#[derive(Debug)]
#[allow(dead_code)]
pub struct CacheEntry {
    pub engine: InferenceEngine,
    pub last_used: Instant,
    pub access_count: u64,
    pub preloaded: bool,
}

/// High-performance model cache with multiple strategies
#[derive(Debug)]
#[allow(dead_code)]
pub struct ModelCache {
    cache: HashMap<String, CacheEntry>,
    max_size: usize,
    policy: EvictionPolicy,
    stats: CacheStats,
}

impl ModelCache {
    /// Create new cache with capacity and eviction policy
    #[allow(dead_code)]
    pub fn new(max_size: usize, policy: EvictionPolicy) -> Self {
        Self {
            cache: HashMap::new(),
            max_size,
            policy,
            stats: CacheStats::default(),
        }
    }

    /// Load model with caching
    #[allow(dead_code)]
    pub fn load(&mut self, id: &str, path: PathBuf) -> MinervaResult<()> {
        if self.cache.contains_key(id) {
            self.stats.hits += 1;
            return Ok(());
        }

        self.stats.misses += 1;

        // Evict if at capacity
        if self.cache.len() >= self.max_size {
            self.evict_one()?;
        }

        // Create and load engine
        let mut engine = InferenceEngine::new(path);
        engine.load_model()?;

        self.cache.insert(
            id.to_string(),
            CacheEntry {
                engine,
                last_used: Instant::now(),
                access_count: 0,
                preloaded: false,
            },
        );

        tracing::info!("Model cached: {}", id);
        Ok(())
    }

    /// Preload model (load immediately without usage)
    #[allow(dead_code)]
    pub fn preload(&mut self, id: &str, path: PathBuf) -> MinervaResult<()> {
        if self.cache.contains_key(id) {
            return Ok(());
        }

        self.stats.preloads += 1;

        if self.cache.len() >= self.max_size {
            self.evict_one()?;
        }

        let mut engine = InferenceEngine::new(path);
        engine.load_model()?;

        self.cache.insert(
            id.to_string(),
            CacheEntry {
                engine,
                last_used: Instant::now(),
                access_count: 0,
                preloaded: true,
            },
        );

        tracing::info!("Model preloaded: {}", id);
        Ok(())
    }

    /// Get mutable reference and update stats
    #[allow(dead_code)]
    pub fn get_mut(&mut self, id: &str) -> MinervaResult<&mut InferenceEngine> {
        self.cache
            .get_mut(id)
            .map(|entry| {
                entry.last_used = Instant::now();
                entry.access_count += 1;
                self.stats.hits += 1;
                &mut entry.engine
            })
            .ok_or_else(|| {
                self.stats.misses += 1;
                MinervaError::ModelNotFound(format!("Model not in cache: {}", id))
            })
    }

    /// Get immutable reference
    #[allow(dead_code)]
    pub fn get(&self, id: &str) -> Option<&InferenceEngine> {
        self.cache.get(id).map(|entry| &entry.engine)
    }

    /// Remove model from cache
    #[allow(dead_code)]
    pub fn remove(&mut self, id: &str) -> MinervaResult<()> {
        self.cache
            .remove(id)
            .ok_or_else(|| MinervaError::ModelNotFound(format!("Model not in cache: {}", id)))?;
        tracing::info!("Model removed from cache: {}", id);
        Ok(())
    }

    /// Check if model is cached
    #[allow(dead_code)]
    pub fn contains(&self, id: &str) -> bool {
        self.cache.contains_key(id)
    }

    /// Get list of cached model IDs
    #[allow(dead_code)]
    pub fn list(&self) -> Vec<String> {
        self.cache.keys().cloned().collect()
    }

    /// Get cache size
    #[allow(dead_code)]
    pub fn size(&self) -> usize {
        self.cache.len()
    }

    /// Get cache capacity
    #[allow(dead_code)]
    pub fn capacity(&self) -> usize {
        self.max_size
    }

    /// Get cache statistics
    #[allow(dead_code)]
    pub fn stats(&self) -> CacheStats {
        self.stats.clone()
    }

    /// Clear all entries
    #[allow(dead_code)]
    pub fn clear(&mut self) {
        for (id, mut entry) in self.cache.drain() {
            entry.engine.unload_model();
            tracing::info!("Cache entry cleared: {}", id);
        }
    }

    /// Evict one entry based on policy
    fn evict_one(&mut self) -> MinervaResult<()> {
        let victim = match self.policy {
            EvictionPolicy::Lru => self.find_lru(),
            EvictionPolicy::Lfu => self.find_lfu(),
            EvictionPolicy::Fifo => self.find_oldest(),
        };

        if let Some(id) = victim {
            if let Some(mut entry) = self.cache.remove(&id) {
                entry.engine.unload_model();
                self.stats.evictions += 1;
                tracing::info!("Model evicted from cache: {} ({:?})", id, self.policy);
                Ok(())
            } else {
                Err(MinervaError::InferenceError(
                    "Failed to evict model".to_string(),
                ))
            }
        } else {
            Err(MinervaError::InferenceError(
                "No models to evict".to_string(),
            ))
        }
    }

    /// Find least recently used model ID
    fn find_lru(&self) -> Option<String> {
        self.cache
            .iter()
            .min_by_key(|(_, entry)| entry.last_used)
            .map(|(id, _)| id.clone())
    }

    /// Find least frequently used model ID
    fn find_lfu(&self) -> Option<String> {
        self.cache
            .iter()
            .min_by_key(|(_, entry)| entry.access_count)
            .map(|(id, _)| id.clone())
    }

    /// Find oldest model ID (FIFO)
    fn find_oldest(&self) -> Option<String> {
        self.find_lru()
    }
}

impl Default for ModelCache {
    fn default() -> Self {
        Self::new(3, EvictionPolicy::Lru)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_creation() {
        let cache = ModelCache::new(2, EvictionPolicy::Lru);
        assert_eq!(cache.size(), 0);
        assert_eq!(cache.capacity(), 2);
    }

    #[test]
    fn test_cache_default() {
        let cache = ModelCache::default();
        assert_eq!(cache.capacity(), 3);
        assert_eq!(cache.size(), 0);
    }

    #[test]
    fn test_cache_contains() {
        let cache = ModelCache::new(2, EvictionPolicy::Lru);
        assert!(!cache.contains("test"));
    }

    #[test]
    fn test_cache_list_empty() {
        let cache = ModelCache::new(2, EvictionPolicy::Lru);
        assert!(cache.list().is_empty());
    }

    #[test]
    fn test_cache_stats_default() {
        let cache = ModelCache::new(2, EvictionPolicy::Lru);
        let stats = cache.stats();
        assert_eq!(stats.hits, 0);
        assert_eq!(stats.misses, 0);
        assert_eq!(stats.hit_rate(), 0.0);
    }

    #[test]
    fn test_cache_stats_hit_rate() {
        let stats = CacheStats {
            hits: 80,
            misses: 20,
            ..Default::default()
        };
        assert_eq!(stats.hit_rate(), 80.0);
    }

    #[test]
    fn test_cache_remove_nonexistent() {
        let mut cache = ModelCache::new(2, EvictionPolicy::Lru);
        assert!(cache.remove("nonexistent").is_err());
    }

    #[test]
    fn test_cache_get_nonexistent() {
        let mut cache = ModelCache::new(2, EvictionPolicy::Lru);
        assert!(cache.get_mut("nonexistent").is_err());
    }

    #[test]
    fn test_cache_policy_lru() {
        let cache = ModelCache::new(2, EvictionPolicy::Lru);
        assert!(matches!(cache.policy, EvictionPolicy::Lru));
    }

    #[test]
    fn test_cache_policy_lfu() {
        let cache = ModelCache::new(2, EvictionPolicy::Lfu);
        assert!(matches!(cache.policy, EvictionPolicy::Lfu));
    }

    #[test]
    fn test_preload_strategy_eager() {
        let strategy = PreloadStrategy::Eager;
        assert!(matches!(strategy, PreloadStrategy::Eager));
    }

    #[test]
    fn test_cache_entry_preloaded() {
        use std::path::PathBuf;
        let engine = InferenceEngine::new(PathBuf::from("/test/model.gguf"));
        let entry = CacheEntry {
            engine,
            last_used: Instant::now(),
            access_count: 0,
            preloaded: true,
        };

        assert!(entry.preloaded);
        assert_eq!(entry.access_count, 0);
    }

    #[test]
    fn test_cache_stats_increments() {
        let mut stats = CacheStats::default();
        assert_eq!(stats.hits, 0);
        stats.hits += 1;
        assert_eq!(stats.hits, 1);
    }

    #[test]
    fn test_eviction_policy_clone() {
        let policy = EvictionPolicy::Lru;
        let cloned = policy;
        assert!(matches!(cloned, EvictionPolicy::Lru));
    }
}
