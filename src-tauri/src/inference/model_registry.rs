use crate::error::{MinervaError, MinervaResult};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

/// Metadata about a cached model
#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(dead_code)]
pub struct ModelMetadata {
    pub id: String,
    pub path: PathBuf,
    pub size_mb: u64,
    pub cached: bool,
    pub last_accessed: Option<u64>,
    pub access_count: u64,
    pub hash: String,
}

impl ModelMetadata {
    /// Create metadata for a model file
    #[allow(dead_code)]
    pub fn from_path(id: &str, path: PathBuf) -> MinervaResult<Self> {
        let metadata = std::fs::metadata(&path)?;
        let size_mb = metadata.len() / (1024 * 1024);
        let hash = Self::compute_hash(&path)?;

        Ok(Self {
            id: id.to_string(),
            path,
            size_mb,
            cached: false,
            last_accessed: None,
            access_count: 0,
            hash,
        })
    }

    /// Compute file hash for integrity checking
    fn compute_hash(path: &Path) -> MinervaResult<String> {
        use std::fs::File;
        use std::io::Read;

        let mut file = File::open(path)?;
        let mut buffer = [0; 8192];
        let mut hash = 0u64;

        loop {
            let n = file.read(&mut buffer)?;
            if n == 0 {
                break;
            }
            for &byte in &buffer[..n] {
                hash = hash.wrapping_mul(31).wrapping_add(byte as u64);
            }
        }

        Ok(format!("{:x}", hash))
    }

    /// Update last accessed timestamp
    #[allow(dead_code)]
    pub fn touch(&mut self) {
        self.last_accessed = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .ok()
            .map(|d| d.as_secs());
        self.access_count += 1;
    }

    /// Get age in seconds since last access
    #[allow(dead_code)]
    pub fn age_seconds(&self) -> Option<u64> {
        let now = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .ok()?
            .as_secs();

        self.last_accessed.map(|last| now.saturating_sub(last))
    }

    /// Check if file still exists and matches hash
    #[allow(dead_code)]
    pub fn verify(&self) -> MinervaResult<bool> {
        if !self.path.exists() {
            return Ok(false);
        }

        let current_hash = Self::compute_hash(&self.path)?;
        Ok(current_hash == self.hash)
    }
}

/// Registry of available and cached models
#[derive(Debug)]
#[allow(dead_code)]
pub struct ModelRegistry {
    models: HashMap<String, ModelMetadata>,
    cache_dir: Option<PathBuf>,
    max_cache_size_mb: u64,
    current_cache_size_mb: u64,
}

impl ModelRegistry {
    /// Create new registry
    #[allow(dead_code)]
    pub fn new() -> Self {
        Self {
            models: HashMap::new(),
            cache_dir: None,
            max_cache_size_mb: 50000, // 50GB default
            current_cache_size_mb: 0,
        }
    }

    /// Create registry with cache directory
    #[allow(dead_code)]
    pub fn with_cache_dir(cache_dir: PathBuf) -> MinervaResult<Self> {
        if !cache_dir.exists() {
            std::fs::create_dir_all(&cache_dir)?;
        }

        Ok(Self {
            models: HashMap::new(),
            cache_dir: Some(cache_dir),
            max_cache_size_mb: 50000,
            current_cache_size_mb: 0,
        })
    }

    /// Register a model
    #[allow(dead_code)]
    pub fn register(&mut self, id: &str, path: PathBuf) -> MinervaResult<()> {
        let metadata = ModelMetadata::from_path(id, path)?;
        self.models.insert(id.to_string(), metadata);
        Ok(())
    }

    /// Get model metadata
    #[allow(dead_code)]
    pub fn get(&self, id: &str) -> Option<&ModelMetadata> {
        self.models.get(id)
    }

    /// Get mutable reference
    #[allow(dead_code)]
    pub fn get_mut(&mut self, id: &str) -> Option<&mut ModelMetadata> {
        self.models.get_mut(id)
    }

    /// Mark model as cached
    #[allow(dead_code)]
    pub fn mark_cached(&mut self, id: &str) -> MinervaResult<()> {
        self.models
            .get_mut(id)
            .ok_or_else(|| MinervaError::ModelNotFound(format!("Model not found: {}", id)))?
            .cached = true;
        Ok(())
    }

    /// Mark model as accessed
    #[allow(dead_code)]
    pub fn access(&mut self, id: &str) -> MinervaResult<()> {
        self.models
            .get_mut(id)
            .ok_or_else(|| MinervaError::ModelNotFound(format!("Model not found: {}", id)))?
            .touch();
        Ok(())
    }

    /// Get all registered models
    #[allow(dead_code)]
    pub fn list(&self) -> Vec<&ModelMetadata> {
        self.models.values().collect()
    }

    /// Get cached models
    #[allow(dead_code)]
    pub fn list_cached(&self) -> Vec<&ModelMetadata> {
        self.models.values().filter(|m| m.cached).collect()
    }

    /// Get total size of cached models
    #[allow(dead_code)]
    pub fn cached_size_mb(&self) -> u64 {
        self.models
            .values()
            .filter(|m| m.cached)
            .map(|m| m.size_mb)
            .sum()
    }

    /// Check if adding model would exceed cache limit
    #[allow(dead_code)]
    pub fn would_exceed_limit(&self, model_size_mb: u64) -> bool {
        self.cached_size_mb() + model_size_mb > self.max_cache_size_mb
    }

    /// Get models sorted by age (oldest first)
    #[allow(dead_code)]
    pub fn oldest_cached(&self) -> Vec<&ModelMetadata> {
        let mut models: Vec<_> = self.models.values().filter(|m| m.cached).collect();
        models.sort_by_key(|m| m.age_seconds());
        models
    }

    /// Get models sorted by access count (least used first)
    #[allow(dead_code)]
    pub fn least_used_cached(&self) -> Vec<&ModelMetadata> {
        let mut models: Vec<_> = self.models.values().filter(|m| m.cached).collect();
        models.sort_by_key(|m| m.access_count);
        models
    }

    /// Set maximum cache size
    #[allow(dead_code)]
    pub fn set_max_cache_size(&mut self, size_mb: u64) {
        self.max_cache_size_mb = size_mb;
    }

    /// Get cache usage percentage
    #[allow(dead_code)]
    pub fn cache_usage_percent(&self) -> f32 {
        if self.max_cache_size_mb == 0 {
            0.0
        } else {
            (self.cached_size_mb() as f32 / self.max_cache_size_mb as f32) * 100.0
        }
    }

    /// Remove model from registry
    #[allow(dead_code)]
    pub fn remove(&mut self, id: &str) -> Option<ModelMetadata> {
        self.models.remove(id)
    }

    /// Clear all models
    #[allow(dead_code)]
    pub fn clear(&mut self) {
        self.models.clear();
    }

    /// Verify all models exist and haven't been corrupted
    #[allow(dead_code)]
    pub fn verify_all(&self) -> MinervaResult<Vec<(String, bool)>> {
        let mut results = Vec::new();
        for (id, metadata) in &self.models {
            let valid = metadata.verify()?;
            results.push((id.clone(), valid));
        }
        Ok(results)
    }
}

impl Default for ModelRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_registry_creation() {
        let registry = ModelRegistry::new();
        assert_eq!(registry.list().len(), 0);
    }

    #[test]
    fn test_registry_with_defaults() {
        let registry = ModelRegistry::default();
        assert_eq!(registry.max_cache_size_mb, 50000);
        assert_eq!(registry.cached_size_mb(), 0);
    }

    #[test]
    fn test_cache_usage_percent_zero() {
        let registry = ModelRegistry::new();
        assert_eq!(registry.cache_usage_percent(), 0.0);
    }

    #[test]
    fn test_registry_list_empty() {
        let registry = ModelRegistry::new();
        assert!(registry.list().is_empty());
    }

    #[test]
    fn test_registry_list_cached_empty() {
        let registry = ModelRegistry::new();
        assert!(registry.list_cached().is_empty());
    }

    #[test]
    fn test_registry_get_nonexistent() {
        let registry = ModelRegistry::new();
        assert!(registry.get("nonexistent").is_none());
    }

    #[test]
    fn test_registry_oldest_cached_empty() {
        let registry = ModelRegistry::new();
        assert!(registry.oldest_cached().is_empty());
    }

    #[test]
    fn test_registry_least_used_empty() {
        let registry = ModelRegistry::new();
        assert!(registry.least_used_cached().is_empty());
    }

    #[test]
    fn test_registry_would_exceed_limit() {
        let mut registry = ModelRegistry::new();
        registry.set_max_cache_size(1000);
        assert!(!registry.would_exceed_limit(500));
        assert!(registry.would_exceed_limit(1500));
    }

    #[test]
    fn test_registry_set_max_cache_size() {
        let mut registry = ModelRegistry::new();
        registry.set_max_cache_size(5000);
        assert_eq!(registry.max_cache_size_mb, 5000);
    }

    #[test]
    fn test_registry_remove_nonexistent() {
        let mut registry = ModelRegistry::new();
        assert!(registry.remove("nonexistent").is_none());
    }

    #[test]
    fn test_registry_clear() {
        let mut registry = ModelRegistry::new();
        registry.clear();
        assert!(registry.list().is_empty());
    }

    #[test]
    fn test_model_metadata_touch() {
        use std::path::PathBuf;
        let mut metadata = ModelMetadata {
            id: "test".to_string(),
            path: PathBuf::from("/test/model.gguf"),
            size_mb: 5000,
            cached: false,
            last_accessed: None,
            access_count: 0,
            hash: "abc123".to_string(),
        };

        assert_eq!(metadata.access_count, 0);
        metadata.touch();
        assert_eq!(metadata.access_count, 1);
        assert!(metadata.last_accessed.is_some());
    }

    #[test]
    fn test_model_metadata_age_seconds() {
        use std::path::PathBuf;
        let mut metadata = ModelMetadata {
            id: "test".to_string(),
            path: PathBuf::from("/test/model.gguf"),
            size_mb: 5000,
            cached: false,
            last_accessed: Some(0),
            access_count: 0,
            hash: "abc123".to_string(),
        };

        let age = metadata.age_seconds();
        assert!(age.is_some());
        assert!(age.unwrap() > 0);
    }

    #[test]
    fn test_metadata_hash_consistency() {
        use std::path::PathBuf;
        let path = PathBuf::from("/nonexistent/model.gguf");
        let result = ModelMetadata::from_path("test", path);
        assert!(result.is_err());
    }
}
