/// Download Cache Management
///
/// Track downloaded models, versions, and metadata.
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

// ============================================================================
// Cache Types
// ============================================================================

/// Cache entry for downloaded model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheEntry {
    /// Model ID
    pub model_id: String,
    /// Local path
    pub path: PathBuf,
    /// Revision/branch
    pub revision: String,
    /// Total size (bytes)
    pub size_bytes: u64,
    /// Files count
    pub file_count: usize,
    /// Download time (unix timestamp)
    pub downloaded_at: u64,
    /// Last accessed (unix timestamp)
    pub last_accessed: u64,
}

/// Download cache
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DownloadCache {
    /// Cached entries
    entries: Vec<CacheEntry>,
}

impl DownloadCache {
    /// Create cache
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Add entry
    pub fn add(&mut self, entry: CacheEntry) {
        self.entries.push(entry);
    }

    /// Get entry
    pub fn get(&self, model_id: &str) -> Option<&CacheEntry> {
        self.entries.iter().find(|e| e.model_id == model_id)
    }

    /// List all
    pub fn list(&self) -> &[CacheEntry] {
        &self.entries
    }

    /// Remove entry
    pub fn remove(&mut self, model_id: &str) {
        self.entries.retain(|e| e.model_id != model_id);
    }

    /// Total size
    pub fn total_size(&self) -> u64 {
        self.entries.iter().map(|e| e.size_bytes).sum()
    }
}

impl Default for DownloadCache {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_new() {
        let cache = DownloadCache::new();
        assert!(cache.list().is_empty());
    }

    #[test]
    fn test_cache_add() {
        let mut cache = DownloadCache::new();
        let entry = CacheEntry {
            model_id: "test".to_string(),
            path: PathBuf::from("/tmp"),
            revision: "main".to_string(),
            size_bytes: 1000,
            file_count: 5,
            downloaded_at: 0,
            last_accessed: 0,
        };
        cache.add(entry);
        assert_eq!(cache.list().len(), 1);
    }

    #[test]
    fn test_cache_get() {
        let mut cache = DownloadCache::new();
        let entry = CacheEntry {
            model_id: "test".to_string(),
            path: PathBuf::from("/tmp"),
            revision: "main".to_string(),
            size_bytes: 1000,
            file_count: 5,
            downloaded_at: 0,
            last_accessed: 0,
        };
        cache.add(entry);
        assert!(cache.get("test").is_some());
    }

    #[test]
    fn test_cache_remove() {
        let mut cache = DownloadCache::new();
        let entry = CacheEntry {
            model_id: "test".to_string(),
            path: PathBuf::from("/tmp"),
            revision: "main".to_string(),
            size_bytes: 1000,
            file_count: 5,
            downloaded_at: 0,
            last_accessed: 0,
        };
        cache.add(entry);
        cache.remove("test");
        assert!(cache.list().is_empty());
    }
}
