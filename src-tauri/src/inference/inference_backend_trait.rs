/// Inference Backend Trait
///
/// Defines the interface for pluggable inference backends.
/// This abstraction allows flexible switching between:
/// - Mock backends for testing
/// - Real llama.cpp inference
/// - Future ONNX, Hugging Face, or other backend implementations
use crate::error::MinervaResult;
use std::path::Path;

/// Parameters for text generation
#[derive(Debug, Clone, Copy)]
pub struct GenerationParams {
    /// Maximum number of tokens to generate
    pub max_tokens: usize,
    /// Sampling temperature (0.0-2.0)
    pub temperature: f32,
    /// Top-p (nucleus) sampling parameter
    pub top_p: f32,
}

/// Trait defining the inference backend interface
///
/// This allows pluggable backends (mock, real llama.cpp, onnx, etc.)
/// Implementations must be Send + Sync for thread safety.
#[allow(dead_code)]
pub trait InferenceBackend: Send + Sync {
    /// Load model from path
    fn load_model(&mut self, path: &Path, n_ctx: usize) -> MinervaResult<()>;

    /// Unload model and free resources
    fn unload_model(&mut self);

    /// Generate text from prompt
    fn generate(&self, prompt: &str, params: GenerationParams) -> MinervaResult<String>;

    /// Tokenize text into token IDs
    fn tokenize(&self, text: &str) -> MinervaResult<Vec<i32>>;

    /// Detokenize token IDs back to text
    fn detokenize(&self, tokens: &[i32]) -> MinervaResult<String>;

    /// Check if model is loaded
    fn is_loaded(&self) -> bool;

    /// Get model context size
    fn context_size(&self) -> usize;

    /// Get number of threads
    fn thread_count(&self) -> usize;
}
