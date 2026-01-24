use crate::error::MinervaResult;
use crate::middleware::RateLimiter;
use crate::models::ModelRegistry;
use crate::observability::metrics::MetricsCollector;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::Mutex;

pub type SharedModelRegistry = Arc<Mutex<ModelRegistry>>;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(dead_code)]
pub struct ModelLoadRequest {
    pub model_id: String,
    pub model_path: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(dead_code)]
pub struct ModelOperationResponse {
    pub success: bool,
    pub message: String,
    pub model_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(dead_code)]
pub struct ModelStatsResponse {
    pub loaded_models: Vec<String>,
    pub total_loaded: usize,
    pub estimated_memory_mb: u64,
}

#[derive(Clone)]
#[allow(dead_code)]
pub struct ServerState {
    pub model_registry: SharedModelRegistry,
    pub metrics: Arc<MetricsCollector>,
    pub rate_limiter: Arc<RateLimiter>,
}

impl ServerState {
    #[allow(dead_code)]
    pub fn new() -> Self {
        Self {
            model_registry: Arc::new(Mutex::new(ModelRegistry::new())),
            metrics: Arc::new(MetricsCollector::new()),
            rate_limiter: Arc::new(RateLimiter::new(100.0, 10.0)),
        }
    }

    /// Create server state and load discovered models
    #[allow(dead_code)]
    pub fn with_discovered_models(models_dir: std::path::PathBuf) -> MinervaResult<Self> {
        let mut registry = ModelRegistry::new();
        registry.discover(&models_dir)?;

        Ok(Self {
            model_registry: Arc::new(Mutex::new(registry)),
            metrics: Arc::new(MetricsCollector::new()),
            rate_limiter: Arc::new(RateLimiter::new(100.0, 10.0)),
        })
    }
}

impl Default for ServerState {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_server_state_creation() {
        let state = ServerState::new();
        assert!(state
            .model_registry
            .blocking_lock()
            .list_models()
            .is_empty());
    }

    #[test]
    fn test_server_state_default() {
        let state = ServerState::default();
        assert!(state
            .model_registry
            .blocking_lock()
            .list_models()
            .is_empty());
    }
}
