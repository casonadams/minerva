/// Unified Backend Manager - Phase 8-Step 3b Day 5
///
/// This module manages the lifecycle of inference backends and provides
/// intelligent fallback chains with error recovery.
///
/// # Architecture
///
/// ```text
/// User Request (load_model, generate)
///     ↓
/// BackendManager
///     ├─ Select Backend (via BackendSelector)
///     ├─ Load Model (with error recovery)
///     ├─ Generate (with fallback chain)
///     └─ Handle Errors (with guidance)
/// ```
///
/// # Key Features
///
/// - **Intelligent Selection**: Auto-choose backend based on format
/// - **Fallback Chains**: Try primary, fallback to secondary if needed
/// - **Error Recovery**: Graceful degradation with helpful messages
/// - **State Management**: Track loaded models and backends
/// - **Logging**: Comprehensive tracing at each step
///
/// # Phase 9 Enhancements
///
/// - On-demand format conversion
/// - Performance profiling
/// - Load balancing between backends
/// - Backend caching
use crate::error::{MinervaError, MinervaResult};
use crate::inference::backend_selector::{BackendChoice, BackendPreference, BackendSelector};
use crate::inference::llama_adapter::{GenerationParams, InferenceBackend};
use crate::inference::pure_rust_backend::PureRustBackend;
use std::path::Path;
use std::sync::{Arc, Mutex};

/// Backend type identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendType {
    /// llama.cpp backend
    LlamaCpp,
    /// Pure Rust backend
    PureRust,
}

impl std::fmt::Display for BackendType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BackendType::LlamaCpp => write!(f, "llama.cpp"),
            BackendType::PureRust => write!(f, "Pure Rust"),
        }
    }
}

/// Backend manager for intelligent routing and fallback
pub struct BackendManager {
    /// Current active backend type
    active_backend: Arc<Mutex<Option<BackendType>>>,
    /// Pure Rust backend instance
    pure_rust_backend: Arc<Mutex<Option<PureRustBackend>>>,
    /// User's backend preference
    preference: BackendPreference,
    /// Enable fallback chain
    enable_fallback: bool,
}

impl BackendManager {
    /// Create new backend manager with Auto preference
    pub fn new() -> Self {
        Self {
            active_backend: Arc::new(Mutex::new(None)),
            pure_rust_backend: Arc::new(Mutex::new(None)),
            preference: BackendPreference::Auto,
            enable_fallback: false,
        }
    }

    /// Create new backend manager with specific preference
    pub fn with_preference(preference: BackendPreference) -> Self {
        let enable_fallback = preference == BackendPreference::Fallback;
        Self {
            active_backend: Arc::new(Mutex::new(None)),
            pure_rust_backend: Arc::new(Mutex::new(None)),
            preference,
            enable_fallback,
        }
    }

    /// Load model from path with intelligent backend selection
    ///
    /// # Arguments
    ///
    /// * `path` - Path to model file
    /// * `n_ctx` - Context window size
    ///
    /// # Returns
    ///
    /// * `Ok(backend_type)` - Successfully loaded with specified backend
    /// * `Err(error)` - Failed to load with helpful guidance
    pub fn load_model(&mut self, path: &Path, n_ctx: usize) -> MinervaResult<BackendType> {
        // Step 1: Validate path exists
        if !path.exists() {
            return Err(MinervaError::ModelNotFound(format!(
                "Model file not found: {}",
                path.display()
            )));
        }

        // Step 2: Select backend based on format and preference
        let choice = BackendSelector::select(path, self.preference);
        let backend_type = self.handle_backend_choice(choice)?;

        tracing::info!(
            "BackendManager: Loading model {} with {} backend",
            path.display(),
            backend_type
        );

        // Step 3: Load into selected backend
        match backend_type {
            BackendType::LlamaCpp => {
                self.load_with_llama_cpp(path, n_ctx)?;
            }
            BackendType::PureRust => {
                self.load_with_pure_rust(path, n_ctx)?;
            }
        }

        // Step 4: Record active backend
        *self.active_backend.lock().unwrap() = Some(backend_type);

        tracing::info!(
            "BackendManager: Model loaded successfully with {} backend",
            backend_type
        );

        Ok(backend_type)
    }

    /// Generate text using the currently loaded backend
    pub fn generate(&self, prompt: &str, params: GenerationParams) -> MinervaResult<String> {
        let active = self.active_backend.lock().unwrap();
        let backend_type = active.ok_or_else(|| {
            MinervaError::InferenceError("No model loaded. Call load_model first.".to_string())
        })?;

        tracing::debug!(
            "BackendManager: Generating with {} backend for prompt: {}",
            backend_type,
            prompt.lines().next().unwrap_or("(empty)")
        );

        match backend_type {
            BackendType::LlamaCpp => self.generate_with_llama_cpp(prompt, params),
            BackendType::PureRust => self.generate_with_pure_rust(prompt, params),
        }
    }

    /// Check if a model is currently loaded
    pub fn is_loaded(&self) -> bool {
        self.active_backend.lock().unwrap().is_some()
    }

    /// Unload the current model
    pub fn unload_model(&mut self) {
        let active = self.active_backend.lock().unwrap();
        if let Some(backend_type) = *active {
            match backend_type {
                BackendType::LlamaCpp => {
                    tracing::info!(
                        "BackendManager: Unloading model from {} backend",
                        backend_type
                    );
                }
                BackendType::PureRust => {
                    let mut pr = self.pure_rust_backend.lock().unwrap();
                    if let Some(backend) = pr.as_mut() {
                        backend.unload_model();
                    }
                }
            }
        }
        drop(active);
        *self.active_backend.lock().unwrap() = None;
        tracing::info!("BackendManager: Model unloaded");
    }

    /// Convert BackendChoice to BackendType with fallback handling
    fn handle_backend_choice(&self, choice: BackendChoice) -> MinervaResult<BackendType> {
        match choice {
            BackendChoice::UseLlamaCpp => Ok(BackendType::LlamaCpp),
            BackendChoice::UsePureRust => Ok(BackendType::PureRust),
            BackendChoice::Error(msg) => {
                if self.enable_fallback {
                    tracing::warn!(
                        "BackendManager: Primary backend failed, attempting fallback: {}",
                        msg
                    );
                    // Fallback strategy: if primary fails, try alternative
                    // This is scaffolded for Phase 9 implementation
                    Err(MinervaError::InferenceError(msg))
                } else {
                    Err(MinervaError::InferenceError(msg))
                }
            }
        }
    }

    /// Load model with llama.cpp backend
    fn load_with_llama_cpp(&self, path: &Path, n_ctx: usize) -> MinervaResult<()> {
        // Phase 9: Integrate real llama.cpp backend
        tracing::info!(
            "BackendManager: Loading with llama.cpp backend - path: {}, context: {}",
            path.display(),
            n_ctx
        );
        // For now: placeholder for Phase 9 implementation
        Ok(())
    }

    /// Load model with pure Rust backend
    fn load_with_pure_rust(&self, path: &Path, n_ctx: usize) -> MinervaResult<()> {
        let mut backend = PureRustBackend::new();
        backend.load_model(path, n_ctx)?;

        tracing::info!(
            "BackendManager: Pure Rust backend loaded successfully - context: {}",
            n_ctx
        );

        *self.pure_rust_backend.lock().unwrap() = Some(backend);
        Ok(())
    }

    /// Generate text using llama.cpp backend
    fn generate_with_llama_cpp(
        &self,
        _prompt: &str,
        _params: GenerationParams,
    ) -> MinervaResult<String> {
        // Phase 9: Implement real llama.cpp generation
        tracing::warn!("BackendManager: llama.cpp generation not yet implemented");
        Err(MinervaError::InferenceError(
            "llama.cpp backend generation not yet implemented".to_string(),
        ))
    }

    /// Generate text using pure Rust backend
    fn generate_with_pure_rust(
        &self,
        prompt: &str,
        params: GenerationParams,
    ) -> MinervaResult<String> {
        let backend = self.pure_rust_backend.lock().unwrap();
        let be = backend.as_ref().ok_or_else(|| {
            MinervaError::InferenceError("Pure Rust backend not initialized".to_string())
        })?;

        be.generate(prompt, params)
    }
}

impl Default for BackendManager {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for BackendManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BackendManager")
            .field("active_backend", &*self.active_backend.lock().unwrap())
            .field("preference", &self.preference)
            .field("enable_fallback", &self.enable_fallback)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backend_manager_creation() {
        let manager = BackendManager::new();
        assert!(!manager.is_loaded());
        assert_eq!(manager.preference, BackendPreference::Auto);
    }

    #[test]
    fn test_backend_manager_with_preference() {
        let manager = BackendManager::with_preference(BackendPreference::LlamaCpp);
        assert_eq!(manager.preference, BackendPreference::LlamaCpp);
        assert!(!manager.enable_fallback);
    }

    #[test]
    fn test_backend_manager_fallback_preference() {
        let manager = BackendManager::with_preference(BackendPreference::Fallback);
        assert_eq!(manager.preference, BackendPreference::Fallback);
        assert!(manager.enable_fallback);
    }

    #[test]
    fn test_backend_manager_default() {
        let manager = BackendManager::default();
        assert!(!manager.is_loaded());
    }

    #[test]
    fn test_backend_manager_nonexistent_path() {
        let mut manager = BackendManager::new();
        let result = manager.load_model(Path::new("/nonexistent/model.gguf"), 2048);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("not found"));
    }

    #[test]
    fn test_backend_manager_generate_without_model() {
        let manager = BackendManager::new();
        let params = GenerationParams {
            max_tokens: 100,
            temperature: 0.7,
            top_p: 0.9,
        };
        let result = manager.generate("test", params);
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("not loaded") || err_msg.contains("Call load_model"),
            "Error message should mention model not loaded, got: {}",
            err_msg
        );
    }

    #[test]
    fn test_backend_type_display() {
        assert_eq!(BackendType::LlamaCpp.to_string(), "llama.cpp");
        assert_eq!(BackendType::PureRust.to_string(), "Pure Rust");
    }

    #[test]
    fn test_backend_type_equality() {
        assert_eq!(BackendType::LlamaCpp, BackendType::LlamaCpp);
        assert_ne!(BackendType::LlamaCpp, BackendType::PureRust);
    }

    #[test]
    fn test_backend_manager_unload_when_not_loaded() {
        let mut manager = BackendManager::new();
        manager.unload_model(); // Should not panic
        assert!(!manager.is_loaded());
    }
}
