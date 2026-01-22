use super::InferenceEngine;
use super::model_cache::{CacheStats, EvictionPolicy, ModelCache};
use crate::error::{MinervaError, MinervaResult};
use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Instant;

/// Usage statistics for a model context
#[derive(Debug, Clone, Default)]
#[allow(dead_code)]
pub struct UsageStats {
    pub total_requests: u64,
    pub total_tokens_generated: u64,
    pub total_time_ms: u128,
    pub last_used: Option<Instant>,
}

/// A loaded model context with its engine and metadata
#[derive(Debug)]
#[allow(dead_code)]
pub struct ModelContext {
    pub engine: InferenceEngine,
    pub last_used: Option<Instant>,
    pub usage_stats: UsageStats,
}

impl ModelContext {
    /// Create a new model context
    #[allow(dead_code)]
    pub fn new(engine: InferenceEngine) -> Self {
        Self {
            engine,
            last_used: None,
            usage_stats: UsageStats::default(),
        }
    }

    /// Update last used timestamp
    #[allow(dead_code)]
    pub fn update_last_used(&mut self) {
        self.last_used = Some(Instant::now());
        self.usage_stats.total_requests += 1;
    }

    /// Get time since last use in seconds
    #[allow(dead_code)]
    pub fn time_since_last_use(&self) -> Option<u64> {
        self.last_used.map(|last| last.elapsed().as_secs())
    }
}

/// Manages multiple loaded model contexts with caching
#[allow(dead_code)]
#[derive(Debug)]
pub struct ContextManager {
    models: HashMap<String, ModelContext>,
    max_models_loaded: usize,
    cache: ModelCache,
    memory_estimated_mb: u64,
}

impl ContextManager {
    /// Create a new context manager
    pub fn new(max_models_loaded: usize) -> Self {
        Self {
            models: HashMap::new(),
            max_models_loaded,
            cache: ModelCache::new(max_models_loaded, EvictionPolicy::Lru),
            memory_estimated_mb: 0,
        }
    }

    /// Create context manager with custom eviction policy
    #[allow(dead_code)]
    pub fn with_policy(max_models_loaded: usize, policy: EvictionPolicy) -> Self {
        Self {
            models: HashMap::new(),
            max_models_loaded,
            cache: ModelCache::new(max_models_loaded, policy),
            memory_estimated_mb: 0,
        }
    }

    /// Load a model into the context manager
    #[allow(dead_code)]
    pub fn load_model(&mut self, id: &str, path: PathBuf) -> MinervaResult<()> {
        // Check if already loaded
        if self.models.contains_key(id) {
            return Ok(());
        }

        // Check if we're at capacity
        if self.models.len() >= self.max_models_loaded {
            // Unload the least recently used model
            self.unload_least_recently_used();
        }

        // Create new engine
        let mut engine = InferenceEngine::new(path);

        // Try to load the model
        engine.load_model()?;

        // Add to context manager
        let context = ModelContext::new(engine);
        self.models.insert(id.to_string(), context);

        tracing::info!("Model loaded into context manager: {}", id);

        Ok(())
    }

    /// Unload a model from the context manager
    #[allow(dead_code)]
    pub fn unload_model(&mut self, id: &str) -> MinervaResult<()> {
        if let Some(mut context) = self.models.remove(id) {
            context.engine.unload_model();
            tracing::info!("Model unloaded from context manager: {}", id);
            Ok(())
        } else {
            Err(MinervaError::ModelNotFound(format!(
                "Model not loaded: {}",
                id
            )))
        }
    }

    /// Get mutable reference to a model's engine
    #[allow(dead_code)]
    pub fn get_model_mut(&mut self, id: &str) -> MinervaResult<&mut InferenceEngine> {
        self.models
            .get_mut(id)
            .map(|context| {
                context.update_last_used();
                &mut context.engine
            })
            .ok_or_else(|| MinervaError::ModelNotFound(format!("Model not loaded: {}", id)))
    }

    /// Get immutable reference to a model's engine
    #[allow(dead_code)]
    pub fn get_model(&self, id: &str) -> MinervaResult<&InferenceEngine> {
        self.models
            .get(id)
            .map(|context| &context.engine)
            .ok_or_else(|| MinervaError::ModelNotFound(format!("Model not loaded: {}", id)))
    }

    /// Get usage statistics for a model
    #[allow(dead_code)]
    pub fn get_usage_stats(&self, id: &str) -> MinervaResult<UsageStats> {
        self.models
            .get(id)
            .map(|context| context.usage_stats.clone())
            .ok_or_else(|| MinervaError::ModelNotFound(format!("Model not loaded: {}", id)))
    }

    /// Get list of loaded model IDs
    #[allow(dead_code)]
    pub fn get_loaded_models(&self) -> Vec<String> {
        self.models.keys().cloned().collect()
    }

    /// Check if a model is loaded
    #[allow(dead_code)]
    pub fn is_loaded(&self, id: &str) -> bool {
        self.models.contains_key(id)
    }

    /// Clear all loaded models
    #[allow(dead_code)]
    pub fn clear(&mut self) {
        for (id, mut context) in self.models.drain() {
            context.engine.unload_model();
            tracing::info!("Model cleared: {}", id);
        }
    }

    /// Unload the least recently used model
    fn unload_least_recently_used(&mut self) {
        let lru_id = self
            .models
            .iter()
            .min_by_key(|(_, context)| {
                context
                    .last_used
                    .map(|t| t.elapsed().as_secs())
                    .unwrap_or(0)
            })
            .map(|(id, _)| id.clone());

        if let Some(id) = lru_id
            && let Some(mut context) = self.models.remove(&id)
        {
            context.engine.unload_model();
            tracing::info!("Least recently used model unloaded: {}", id);
        }
    }

    /// Get total number of loaded models
    #[allow(dead_code)]
    pub fn loaded_count(&self) -> usize {
        self.models.len()
    }

    /// Get max number of models that can be loaded
    #[allow(dead_code)]
    pub fn max_models(&self) -> usize {
        self.max_models_loaded
    }

    /// Get cache statistics
    #[allow(dead_code)]
    pub fn cache_stats(&self) -> CacheStats {
        self.cache.stats()
    }

    /// Get estimated memory usage in MB
    #[allow(dead_code)]
    pub fn estimated_memory_mb(&self) -> u64 {
        self.memory_estimated_mb
    }

    /// Update estimated memory (rough approximation)
    #[allow(dead_code)]
    pub fn update_memory_estimate(&mut self) {
        self.memory_estimated_mb = (self.models.len() as u64) * 5000;
    }

    /// Check if memory pressure is high (>80% capacity)
    #[allow(dead_code)]
    pub fn has_memory_pressure(&self) -> bool {
        (self.models.len() as f32 / self.max_models_loaded as f32) > 0.8
    }

    /// Get cache hit rate
    #[allow(dead_code)]
    pub fn cache_hit_rate(&self) -> f32 {
        self.cache.stats().hit_rate()
    }

    /// Preload a model (load without marking as used)
    #[allow(dead_code)]
    pub fn preload_model(&mut self, id: &str, path: PathBuf) -> MinervaResult<()> {
        if self.models.contains_key(id) {
            return Ok(());
        }

        if self.models.len() >= self.max_models_loaded {
            self.unload_least_recently_used();
        }

        let mut engine = InferenceEngine::new(path);
        engine.load_model()?;

        let mut context = ModelContext::new(engine);
        context.last_used = None;
        self.models.insert(id.to_string(), context);

        self.update_memory_estimate();
        tracing::info!("Model preloaded: {}", id);

        Ok(())
    }
}

impl Default for ContextManager {
    fn default() -> Self {
        Self::new(3) // Default: max 3 models loaded
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_context_manager_creation() {
        let manager = ContextManager::new(2);
        assert_eq!(manager.loaded_count(), 0);
        assert_eq!(manager.max_models(), 2);
    }

    #[test]
    fn test_usage_stats_default() {
        let stats = UsageStats::default();
        assert_eq!(stats.total_requests, 0);
        assert_eq!(stats.total_tokens_generated, 0);
        assert_eq!(stats.total_time_ms, 0);
    }

    #[test]
    fn test_model_context_creation() {
        let engine = InferenceEngine::new(PathBuf::from("/test/model.gguf"));
        let context = ModelContext::new(engine);

        assert!(!context.engine.is_loaded());
        assert_eq!(context.usage_stats.total_requests, 0);
    }

    #[test]
    fn test_model_context_update_last_used() {
        let engine = InferenceEngine::new(PathBuf::from("/test/model.gguf"));
        let mut context = ModelContext::new(engine);

        assert!(context.last_used.is_none());
        context.update_last_used();
        assert!(context.last_used.is_some());
        assert_eq!(context.usage_stats.total_requests, 1);
    }

    #[test]
    fn test_load_nonexistent_model() {
        let mut manager = ContextManager::new(2);
        let result = manager.load_model("test", PathBuf::from("/nonexistent/model.gguf"));

        assert!(result.is_err());
        assert_eq!(manager.loaded_count(), 0);
    }

    #[test]
    fn test_unload_nonexistent_model() {
        let mut manager = ContextManager::new(2);
        let result = manager.unload_model("nonexistent");

        assert!(result.is_err());
    }

    #[test]
    fn test_get_loaded_models() {
        let manager = ContextManager::new(2);
        let loaded = manager.get_loaded_models();

        assert!(loaded.is_empty());
    }

    #[test]
    fn test_is_loaded() {
        let manager = ContextManager::new(2);

        assert!(!manager.is_loaded("test"));
    }

    #[test]
    fn test_default_context_manager() {
        let manager = ContextManager::default();
        assert_eq!(manager.max_models(), 3);
        assert_eq!(manager.loaded_count(), 0);
    }

    #[test]
    fn test_cache_stats_default() {
        let manager = ContextManager::new(2);
        let stats = manager.cache_stats();
        assert_eq!(stats.hits, 0);
        assert_eq!(stats.misses, 0);
    }

    #[test]
    fn test_memory_estimate() {
        let mut manager = ContextManager::new(2);
        manager.update_memory_estimate();
        assert_eq!(manager.estimated_memory_mb(), 0);
    }

    #[test]
    fn test_memory_pressure_low() {
        let manager = ContextManager::new(3);
        assert!(!manager.has_memory_pressure());
    }

    #[test]
    fn test_cache_hit_rate_zero() {
        let manager = ContextManager::new(2);
        assert_eq!(manager.cache_hit_rate(), 0.0);
    }

    #[test]
    fn test_context_manager_with_policy() {
        use super::super::model_cache::EvictionPolicy;
        let manager = ContextManager::with_policy(2, EvictionPolicy::Lfu);
        assert_eq!(manager.max_models(), 2);
        assert_eq!(manager.loaded_count(), 0);
    }

    #[test]
    fn test_preload_model_nonexistent() {
        let mut manager = ContextManager::new(2);
        let result = manager.preload_model("test", PathBuf::from("/nonexistent/model.gguf"));
        assert!(result.is_err());
    }
}
