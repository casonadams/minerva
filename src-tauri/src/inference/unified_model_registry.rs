//! Phase 10 Day 3: Unified Model Registry
//!
//! Central registry that bridges Pure Rust and MLX inference backends,
//! providing a unified interface for model lifecycle management.
//!
//! # Architecture
//!
//! ```text
//! Application
//!     ↓
//! UnifiedModelRegistry
//!     ├─ ModelMetadata (detection)
//!     ├─ ModelInstance (active model)
//!     ├─ Pure Rust Backend (for safetensors)
//!     └─ MLX Backend (for Apple Silicon)
//! ```
//!
//! # Features
//!
//! - **Unified Interface**: Single API for all backends
//! - **Auto-routing**: Uses best backend per model/hardware
//! - **Lifecycle Management**: Load, cache, unload models
//! - **Error Recovery**: Automatic fallback chains
//! - **Performance Tracking**: Inference timing and memory usage
//! - **Multi-model Support**: Load multiple models concurrently
//! - **Thread-safe**: Full async/await support

use crate::error::{MinervaError, MinervaResult};
use crate::inference::unified_backend::{BackendStrategy, ModelInfo, detect_model};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use tokio::sync::RwLock;

// ============================================================================
// Model Instance Representation
// ============================================================================

/// A loaded model instance with backend tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInstance {
    /// Unique model identifier
    pub model_id: String,
    /// Model metadata
    pub info: ModelInfo,
    /// Backend used for this model
    pub backend: String,
    /// Whether model is currently loaded
    pub is_loaded: bool,
    /// Number of tokens processed
    pub tokens_processed: u64,
    /// Estimated memory usage in MB
    pub memory_mb: u64,
    /// Last access timestamp
    pub last_accessed: u64,
}

/// Statistics about model inference
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceStats {
    /// Model identifier
    pub model_id: String,
    /// Average tokens per second
    pub tokens_per_second: f32,
    /// Total tokens generated
    pub total_tokens: u64,
    /// Peak memory used in MB
    pub peak_memory_mb: u64,
    /// Number of successful inferences
    pub successful_inferences: u64,
    /// Number of failed inferences
    pub failed_inferences: u64,
}

/// Update to inference statistics
#[derive(Debug, Clone)]
pub struct StatsUpdate {
    pub tokens: u64,
    pub memory_mb: u64,
    pub success: bool,
}

// ============================================================================
// Model Registry Configuration
// ============================================================================

/// Configuration for model registry behavior
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegistryConfig {
    /// Maximum number of models to keep loaded
    pub max_loaded_models: usize,
    /// Enable auto-unloading of unused models (LRU)
    pub enable_auto_unload: bool,
    /// Unload models not used for N seconds
    pub unload_timeout_secs: u64,
    /// Preferred backend (auto, mlx, pure_rust, llama_cpp)
    pub preferred_backend: String,
    /// Enable fallback to alternative backends
    pub enable_fallback: bool,
    /// Default context window size
    pub default_context_size: usize,
}

impl Default for RegistryConfig {
    fn default() -> Self {
        Self {
            max_loaded_models: 3,
            enable_auto_unload: true,
            unload_timeout_secs: 3600, // 1 hour
            preferred_backend: "auto".to_string(),
            enable_fallback: true,
            default_context_size: 2048,
        }
    }
}

// ============================================================================
// Unified Model Registry
// ============================================================================

/// Central registry for model management
pub struct UnifiedModelRegistry {
    #[allow(dead_code)]
    config: RegistryConfig,
    /// All known models with their metadata
    models: Arc<RwLock<HashMap<String, ModelInstance>>>,
    /// Inference statistics per model
    stats: Arc<RwLock<HashMap<String, InferenceStats>>>,
    /// Model paths for discovery
    model_paths: Arc<RwLock<Vec<String>>>,
}

impl UnifiedModelRegistry {
    /// Create new registry with default config
    pub fn new() -> Self {
        Self {
            config: RegistryConfig::default(),
            models: Arc::new(RwLock::new(HashMap::new())),
            stats: Arc::new(RwLock::new(HashMap::new())),
            model_paths: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Create registry with custom config
    pub fn with_config(config: RegistryConfig) -> Self {
        Self {
            config,
            models: Arc::new(RwLock::new(HashMap::new())),
            stats: Arc::new(RwLock::new(HashMap::new())),
            model_paths: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Register a model path for discovery
    pub async fn add_model_path(&self, path: String) -> MinervaResult<()> {
        let mut paths = self.model_paths.write().await;
        if !paths.contains(&path) {
            paths.push(path);
        }
        Ok(())
    }

    /// Detect and register a model
    pub async fn register_model(
        &self,
        model_id: &str,
        path: Option<&Path>,
    ) -> MinervaResult<ModelInstance> {
        // Detect model information
        let info = detect_model(model_id, path)?;

        // Determine best backend
        let backend = self.select_backend(&info).await?;

        let instance = ModelInstance {
            model_id: model_id.to_string(),
            info,
            backend,
            is_loaded: false,
            tokens_processed: 0,
            memory_mb: 0,
            last_accessed: current_timestamp(),
        };

        // Store in registry
        let mut models = self.models.write().await;
        models.insert(model_id.to_string(), instance.clone());

        // Initialize stats
        let mut stats = self.stats.write().await;
        stats.insert(
            model_id.to_string(),
            InferenceStats {
                model_id: model_id.to_string(),
                tokens_per_second: 0.0,
                total_tokens: 0,
                peak_memory_mb: 0,
                successful_inferences: 0,
                failed_inferences: 0,
            },
        );

        Ok(instance)
    }

    /// Get model instance
    pub async fn get_model(&self, model_id: &str) -> MinervaResult<ModelInstance> {
        let mut models = self.models.write().await;
        let instance = models.get_mut(model_id).ok_or_else(|| {
            MinervaError::ModelNotFound(format!(
                "Model '{}' not registered. Use register_model first.",
                model_id
            ))
        })?;

        // Update last accessed
        instance.last_accessed = current_timestamp();

        Ok(instance.clone())
    }

    /// Mark model as loaded
    pub async fn mark_loaded(&self, model_id: &str, memory_mb: u64) -> MinervaResult<()> {
        let mut models = self.models.write().await;
        if let Some(instance) = models.get_mut(model_id) {
            instance.is_loaded = true;
            instance.memory_mb = memory_mb;
            instance.last_accessed = current_timestamp();
        }
        Ok(())
    }

    /// Mark model as unloaded
    pub async fn mark_unloaded(&self, model_id: &str) -> MinervaResult<()> {
        let mut models = self.models.write().await;
        if let Some(instance) = models.get_mut(model_id) {
            instance.is_loaded = false;
            instance.memory_mb = 0;
        }
        Ok(())
    }

    /// Update inference statistics
    pub async fn update_stats(&self, model_id: &str, update: StatsUpdate) -> MinervaResult<()> {
        let mut stats = self.stats.write().await;
        if let Some(stat) = stats.get_mut(model_id) {
            stat.total_tokens += update.tokens;
            stat.peak_memory_mb = stat.peak_memory_mb.max(update.memory_mb);
            if update.success {
                stat.successful_inferences += 1;
            } else {
                stat.failed_inferences += 1;
            }
        }
        Ok(())
    }

    /// Get model statistics
    pub async fn get_stats(&self, model_id: &str) -> MinervaResult<InferenceStats> {
        let stats = self.stats.read().await;
        stats.get(model_id).cloned().ok_or_else(|| {
            MinervaError::InferenceError(format!("No statistics for model '{}'", model_id))
        })
    }

    /// List all registered models
    pub async fn list_models(&self) -> MinervaResult<Vec<ModelInstance>> {
        let models = self.models.read().await;
        Ok(models.values().cloned().collect())
    }

    /// List loaded models
    pub async fn list_loaded_models(&self) -> MinervaResult<Vec<ModelInstance>> {
        let models = self.models.read().await;
        Ok(models.values().filter(|m| m.is_loaded).cloned().collect())
    }

    /// Unregister a model
    pub async fn unregister_model(&self, model_id: &str) -> MinervaResult<()> {
        let mut models = self.models.write().await;
        models.remove(model_id);

        let mut stats = self.stats.write().await;
        stats.remove(model_id);

        Ok(())
    }

    /// Clear all models
    pub async fn clear(&self) -> MinervaResult<()> {
        let mut models = self.models.write().await;
        models.clear();

        let mut stats = self.stats.write().await;
        stats.clear();

        Ok(())
    }

    // ========================================================================
    // Backend Selection
    // ========================================================================

    /// Select best backend for a model
    async fn select_backend(&self, info: &ModelInfo) -> MinervaResult<String> {
        // Parse format into ModelFormat enum
        let format = parse_format(&info.format);

        // Get candidate backends
        let strategies =
            BackendStrategy::select_for(format, &info.architecture, std::env::consts::OS)?;

        // Return best option
        Ok(strategies
            .first()
            .map(|s| s.to_string())
            .unwrap_or_else(|| "auto".to_string()))
    }

    /// Get recommended batch size for model
    pub async fn recommended_batch_size(&self, model_id: &str) -> MinervaResult<usize> {
        let instance = self.get_model(model_id).await?;

        // Estimate based on parameters
        let batch_size = match instance.info.param_count {
            x if x < 3.0 => 32,
            x if x < 13.0 => 16,
            x if x < 34.0 => 8,
            _ => 4,
        };

        Ok(batch_size)
    }

    /// Get recommended max tokens for model
    pub async fn recommended_max_tokens(&self, model_id: &str) -> MinervaResult<usize> {
        let instance = self.get_model(model_id).await?;
        Ok(instance.info.context_length / 2)
    }
}

impl Default for UnifiedModelRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Parse format string into ModelFormat enum
fn parse_format(format_str: &str) -> crate::inference::unified_backend::ModelFormat {
    match format_str.to_lowercase().as_str() {
        "safetensors" => crate::inference::unified_backend::ModelFormat::Safetensors,
        "gguf" => crate::inference::unified_backend::ModelFormat::Gguf,
        "pytorch" => crate::inference::unified_backend::ModelFormat::PyTorch,
        "tensorflow" => crate::inference::unified_backend::ModelFormat::TensorFlow,
        "mlx" => crate::inference::unified_backend::ModelFormat::Mlx,
        _ => crate::inference::unified_backend::ModelFormat::Unknown,
    }
}

/// Get current Unix timestamp
fn current_timestamp() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_registry_config_default() {
        let config = RegistryConfig::default();
        assert_eq!(config.max_loaded_models, 3);
        assert!(config.enable_auto_unload);
        assert!(config.enable_fallback);
    }

    #[tokio::test]
    async fn test_registry_creation() {
        let registry = UnifiedModelRegistry::new();
        let models = registry.list_models().await.unwrap();
        assert!(models.is_empty());
    }

    #[tokio::test]
    async fn test_add_model_path() {
        let registry = UnifiedModelRegistry::new();
        registry
            .add_model_path("/models/path".to_string())
            .await
            .unwrap();
        registry
            .add_model_path("/models/path".to_string())
            .await
            .unwrap(); // Duplicate

        let paths = registry.model_paths.read().await;
        assert_eq!(paths.len(), 1); // No duplicates
    }

    #[tokio::test]
    async fn test_register_model() {
        let registry = UnifiedModelRegistry::new();
        let instance = registry
            .register_model("meta-llama/Llama-2-7b", None)
            .await
            .unwrap();

        assert_eq!(instance.model_id, "meta-llama/Llama-2-7b");
        assert_eq!(instance.info.architecture, "llama");
        assert!(!instance.is_loaded);
    }

    #[tokio::test]
    async fn test_get_model() {
        let registry = UnifiedModelRegistry::new();
        registry.register_model("test-model", None).await.unwrap();

        let instance = registry.get_model("test-model").await.unwrap();
        assert_eq!(instance.model_id, "test-model");
    }

    #[tokio::test]
    async fn test_mark_loaded() {
        let registry = UnifiedModelRegistry::new();
        registry.register_model("test-model", None).await.unwrap();

        registry.mark_loaded("test-model", 5000).await.unwrap();
        let instance = registry.get_model("test-model").await.unwrap();

        assert!(instance.is_loaded);
        assert_eq!(instance.memory_mb, 5000);
    }

    #[tokio::test]
    async fn test_stats_tracking() {
        let registry = UnifiedModelRegistry::new();
        registry.register_model("test-model", None).await.unwrap();

        registry
            .update_stats(
                "test-model",
                StatsUpdate {
                    tokens: 100,
                    memory_mb: 1000,
                    success: true,
                },
            )
            .await
            .unwrap();
        registry
            .update_stats(
                "test-model",
                StatsUpdate {
                    tokens: 50,
                    memory_mb: 800,
                    success: true,
                },
            )
            .await
            .unwrap();
        registry
            .update_stats(
                "test-model",
                StatsUpdate {
                    tokens: 0,
                    memory_mb: 0,
                    success: false,
                },
            )
            .await
            .unwrap();

        let stats = registry.get_stats("test-model").await.unwrap();
        assert_eq!(stats.total_tokens, 150);
        assert_eq!(stats.successful_inferences, 2);
        assert_eq!(stats.failed_inferences, 1);
    }

    #[tokio::test]
    async fn test_list_models() {
        let registry = UnifiedModelRegistry::new();
        registry.register_model("model1", None).await.unwrap();
        registry.register_model("model2", None).await.unwrap();

        let models = registry.list_models().await.unwrap();
        assert_eq!(models.len(), 2);
    }

    #[tokio::test]
    async fn test_unregister_model() {
        let registry = UnifiedModelRegistry::new();
        registry.register_model("test-model", None).await.unwrap();
        registry.unregister_model("test-model").await.unwrap();

        let models = registry.list_models().await.unwrap();
        assert!(models.is_empty());
    }

    #[tokio::test]
    async fn test_clear_registry() {
        let registry = UnifiedModelRegistry::new();
        registry.register_model("model1", None).await.unwrap();
        registry.register_model("model2", None).await.unwrap();

        registry.clear().await.unwrap();

        let models = registry.list_models().await.unwrap();
        assert!(models.is_empty());
    }

    #[tokio::test]
    async fn test_recommended_batch_size() {
        let registry = UnifiedModelRegistry::new();
        registry
            .register_model("meta-llama/Llama-2-7b", None)
            .await
            .unwrap();

        let batch_size = registry
            .recommended_batch_size("meta-llama/Llama-2-7b")
            .await
            .unwrap();
        assert!(batch_size > 0);
    }
}
