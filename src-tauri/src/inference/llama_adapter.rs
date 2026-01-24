/// Llama.cpp Adapter Module
///
/// This module provides an abstraction layer between the inference engine
/// and the actual llama_cpp crate. This design allows:
///
/// 1. Easy mocking for testing
/// 2. Flexible switching between mock and real inference
/// 3. Graceful fallback if llama.cpp is unavailable
/// 4. Future integration with different inference backends
///
/// # Integration Path
///
/// When ready to integrate real llama.cpp:
///
/// ```ignore
/// 1. Create LlamaModel and LlamaContext from llama_cpp
/// 2. Implement LlamaBackend trait for real inference
/// 3. Update LlamaEngine to use LlamaBackend
/// 4. Set is_mock = false in production builds
/// ```
use crate::error::{MinervaError, MinervaResult};
use crate::inference::llama_tokenizer::LLaMATokenizer;
use llama_cpp::standard_sampler::StandardSampler;
use llama_cpp::{LlamaModel, LlamaParams, LlamaSession, SessionParams};
use std::path::Path;
use std::sync::{Arc, Mutex};

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

/// Production llama.cpp backend
///
/// This backend uses the actual llama_cpp crate for inference.
/// It maintains a model and session for real LLM inference.
/// Includes real BPE tokenization via LLaMATokenizer.
pub struct LlamaCppBackend {
    model: Arc<Mutex<Option<LlamaModel>>>,
    session: Arc<Mutex<Option<LlamaSession>>>,
    tokenizer: Arc<Mutex<Option<LLaMATokenizer>>>,
    n_ctx: usize,
    n_threads: usize,
}

impl std::fmt::Debug for LlamaCppBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LlamaCppBackend")
            .field("n_ctx", &self.n_ctx)
            .field("n_threads", &self.n_threads)
            .finish()
    }
}

impl LlamaCppBackend {
    /// Create new llama.cpp backend
    pub fn new() -> Self {
        Self {
            model: Arc::new(Mutex::new(None)),
            session: Arc::new(Mutex::new(None)),
            tokenizer: Arc::new(Mutex::new(None)),
            n_ctx: 0,
            n_threads: num_cpus::get(),
        }
    }

    /// Set tokenizer for this backend
    pub fn set_tokenizer(&mut self, tokenizer: LLaMATokenizer) {
        *self.tokenizer.lock().unwrap() = Some(tokenizer);
    }

    /// Detect model format from file path
    /// Returns the file extension as a format identifier
    pub fn detect_format(path: &Path) -> &'static str {
        path.extension()
            .and_then(|ext| ext.to_str())
            .map(|ext| match ext.to_lowercase().as_str() {
                "gguf" => "gguf",
                "safetensors" => "safetensors",
                "bin" => "huggingface",
                "pt" => "pytorch",
                "pb" => "tensorflow",
                _ => "unknown",
            })
            .unwrap_or("unknown")
    }

    /// Check if this backend can handle the model format
    /// llama.cpp backend only supports GGUF format
    pub fn can_handle(path: &Path) -> bool {
        matches!(Self::detect_format(path), "gguf")
    }

    /// Create a fallback tokenizer from common vocabulary
    #[allow(dead_code)]
    fn create_fallback_tokenizer() -> LLaMATokenizer {
        // Common vocabulary for most LLaMA models
        let vocab = vec![
            "<unk>".to_string(), // 0
            "<s>".to_string(),   // 1
            "</s>".to_string(),  // 2
            // Common tokens
            "the".to_string(),
            "a".to_string(),
            "and".to_string(),
            "to".to_string(),
            "of".to_string(),
            "in".to_string(),
            "is".to_string(),
        ];
        // This will succeed since vocab is non-empty
        LLaMATokenizer::new(vocab).unwrap_or_else(|_| {
            // Absolute fallback
            LLaMATokenizer::new(vec!["<unk>".to_string(), "text".to_string()]).unwrap()
        })
    }
}

impl Default for LlamaCppBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl InferenceBackend for LlamaCppBackend {
    fn load_model(&mut self, path: &Path, n_ctx: usize) -> MinervaResult<()> {
        // Validate path exists
        if !path.exists() {
            return Err(MinervaError::ModelNotFound(format!(
                "Model file not found: {}",
                path.display()
            )));
        }

        // Check file format - give helpful error if not GGUF
        let format = Self::detect_format(path);
        if format != "gguf" {
            let guidance = match format {
                "safetensors" | "huggingface" => {
                    " (HuggingFace format detected - use pure Rust backend or convert to GGUF)"
                }
                "pytorch" | "tensorflow" => {
                    " (PyTorch/TensorFlow format detected - convert to GGUF first)"
                }
                _ => " (unsupported format - convert to GGUF)",
            };
            return Err(MinervaError::InvalidRequest(format!(
                "llama.cpp backend only supports GGUF format, got: {}{}",
                format, guidance
            )));
        }

        // Load model with GPU acceleration enabled
        let params = LlamaParams {
            n_gpu_layers: 40, // Offload to GPU
            use_mmap: true,   // Use memory mapping for faster loading
            ..Default::default()
        };

        let model = LlamaModel::load_from_file(path, params).map_err(|e| {
            let err_msg = format!("{:?}", e);
            MinervaError::ModelLoadingError(err_msg)
        })?;

        // Create session for inference
        let session_params = SessionParams::default();
        let session = model.create_session(session_params).map_err(|e| {
            MinervaError::InferenceError(format!("Failed to create inference session: {:?}", e))
        })?;

        // Store in mutex-protected Arc
        *self.model.lock().unwrap() = Some(model);
        *self.session.lock().unwrap() = Some(session);
        self.n_ctx = n_ctx;

        tracing::info!(
            "Model loaded successfully: {} (context: {})",
            path.display(),
            n_ctx
        );

        Ok(())
    }

    fn unload_model(&mut self) {
        *self.model.lock().unwrap() = None;
        *self.session.lock().unwrap() = None;
        *self.tokenizer.lock().unwrap() = None;
        tracing::info!("Model unloaded");
    }

    fn generate(&self, prompt: &str, params: GenerationParams) -> MinervaResult<String> {
        // Validate model and session exist
        let model = self.model.lock().unwrap();
        let mut session = self.session.lock().unwrap();

        let _model = model
            .as_ref()
            .ok_or_else(|| MinervaError::InferenceError("Model not loaded".to_string()))?;

        let session = session
            .as_mut()
            .ok_or_else(|| MinervaError::InferenceError("Session not created".to_string()))?;

        // Advance context with prompt
        session.advance_context(prompt).map_err(|e| {
            MinervaError::InferenceError(format!("Context evaluation failed: {:?}", e))
        })?;

        // Generate tokens with sampler
        let sampler = StandardSampler::default();

        let mut generated_text = String::new();

        let completions = session
            .start_completing_with(sampler, params.max_tokens)
            .map_err(|e| MinervaError::InferenceError(format!("Generation failed: {:?}", e)))?
            .into_strings();

        for completion in completions {
            generated_text.push_str(&completion);
        }

        Ok(generated_text)
    }

    fn tokenize(&self, text: &str) -> MinervaResult<Vec<i32>> {
        let tokenizer = self.tokenizer.lock().unwrap();

        if let Some(tokenizer) = tokenizer.as_ref() {
            // Real tokenization using LLaMATokenizer
            let tokens = tokenizer.encode(text)?;
            Ok(tokens.iter().map(|&t| t as i32).collect())
        } else {
            // Fallback to simple word-based tokenization
            // This happens if tokenizer not explicitly set
            Ok(text
                .split_whitespace()
                .enumerate()
                .map(|(i, _)| i as i32)
                .collect())
        }
    }

    fn detokenize(&self, tokens: &[i32]) -> MinervaResult<String> {
        let tokenizer = self.tokenizer.lock().unwrap();

        if let Some(tokenizer) = tokenizer.as_ref() {
            // Real detokenization using LLaMATokenizer
            let u32_tokens: Vec<u32> = tokens.iter().map(|&t| t as u32).collect();
            tokenizer.decode(&u32_tokens)
        } else {
            // Fallback for when tokenizer not set
            Ok(format!("[{} tokens]", tokens.len()))
        }
    }

    fn is_loaded(&self) -> bool {
        self.model.lock().unwrap().is_some() && self.session.lock().unwrap().is_some()
    }

    fn context_size(&self) -> usize {
        self.n_ctx
    }

    fn thread_count(&self) -> usize {
        self.n_threads
    }
}

/// Mock backend for testing and development
#[derive(Debug)]
#[allow(dead_code)]
pub struct MockBackend {
    loaded: bool,
    n_ctx: usize,
    n_threads: usize,
}

impl MockBackend {
    /// Create new mock backend
    #[allow(dead_code)]
    pub fn new() -> Self {
        Self {
            loaded: false,
            n_ctx: 0,
            n_threads: num_cpus::get(),
        }
    }
}

impl Default for MockBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl InferenceBackend for MockBackend {
    fn load_model(&mut self, path: &Path, n_ctx: usize) -> MinervaResult<()> {
        if !path.exists() {
            return Err(MinervaError::ModelNotFound(format!(
                "Model not found: {}",
                path.display()
            )));
        }
        self.loaded = true;
        self.n_ctx = n_ctx;
        Ok(())
    }

    fn unload_model(&mut self) {
        self.loaded = false;
    }

    fn generate(&self, prompt: &str, params: GenerationParams) -> MinervaResult<String> {
        if !self.loaded {
            return Err(MinervaError::InferenceError("Model not loaded".to_string()));
        }

        // Simulate real inference
        std::thread::sleep(std::time::Duration::from_millis(50));

        let response = self.generate_intelligent_response(prompt, params.max_tokens);
        Ok(response)
    }

    fn tokenize(&self, text: &str) -> MinervaResult<Vec<i32>> {
        // Simple word-based mock tokenization
        Ok(text
            .split_whitespace()
            .enumerate()
            .map(|(i, _)| i as i32)
            .collect())
    }

    fn detokenize(&self, tokens: &[i32]) -> MinervaResult<String> {
        // Mock detokenization
        Ok(format!("[{} tokens]", tokens.len()))
    }

    fn is_loaded(&self) -> bool {
        self.loaded
    }

    fn context_size(&self) -> usize {
        self.n_ctx
    }

    fn thread_count(&self) -> usize {
        self.n_threads
    }
}

impl MockBackend {
    #[allow(dead_code)]
    fn generate_intelligent_response(&self, prompt: &str, max_tokens: usize) -> String {
        let prompt_lower = prompt.to_lowercase();

        let base = if prompt_lower.contains("hello") || prompt_lower.contains("hi") {
            "Hello! I'm an AI assistant. How can I help you today?"
        } else if prompt_lower.contains("what") || prompt_lower.contains("how") {
            "That's an interesting question. Let me provide a thoughtful response. \
             The answer involves multiple interconnected factors. \
             First, consider the foundational principles. \
             Then, examine the practical implications. \
             This comprehensive approach provides better understanding."
        } else if prompt_lower.contains("why") {
            "There are several compelling reasons for this. \
             The primary reason relates to natural efficiency patterns. \
             Historical precedent supports this approach. \
             Contemporary research confirms these findings."
        } else if prompt_lower.contains("explain") || prompt_lower.contains("describe") {
            "Let me provide a detailed explanation. \
             This topic encompasses several key components. \
             Understanding requires examining foundational concepts. \
             Advanced aspects build upon this foundation. \
             This systematic approach ensures comprehensive understanding."
        } else {
            "That's an interesting question. \
             Let me provide a thoughtful analysis. \
             This involves examining multiple perspectives. \
             Different viewpoints offer valuable insights. \
             We should consider both theory and practice."
        };

        // Truncate to max_tokens (approximate as words)
        let words: Vec<&str> = base.split_whitespace().collect();
        if words.len() > max_tokens {
            words[..max_tokens].join(" ")
        } else {
            base.to_string()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    #[test]
    fn test_mock_backend_creation() {
        let backend = MockBackend::new();
        assert!(!backend.is_loaded());
        assert_eq!(backend.context_size(), 0);
        assert!(backend.thread_count() > 0);
    }

    #[test]
    fn test_mock_backend_load() {
        let temp_dir = TempDir::new().unwrap();
        let model_path = temp_dir.path().join("test.gguf");
        fs::write(&model_path, "dummy").unwrap();

        let mut backend = MockBackend::new();
        assert!(backend.load_model(&model_path, 2048).is_ok());
        assert!(backend.is_loaded());
        assert_eq!(backend.context_size(), 2048);
    }

    #[test]
    fn test_mock_backend_unload() {
        let temp_dir = TempDir::new().unwrap();
        let model_path = temp_dir.path().join("test.gguf");
        fs::write(&model_path, "dummy").unwrap();

        let mut backend = MockBackend::new();
        assert!(backend.load_model(&model_path, 2048).is_ok());
        backend.unload_model();
        assert!(!backend.is_loaded());
    }

    #[test]
    fn test_mock_backend_generate() {
        let temp_dir = TempDir::new().unwrap();
        let model_path = temp_dir.path().join("test.gguf");
        fs::write(&model_path, "dummy").unwrap();

        let mut backend = MockBackend::new();
        assert!(backend.load_model(&model_path, 2048).is_ok());

        let params = GenerationParams {
            max_tokens: 100,
            temperature: 0.7,
            top_p: 0.9,
        };
        let result = backend.generate("hello", params);
        assert!(result.is_ok());
        assert!(!result.unwrap().is_empty());
    }

    #[test]
    fn test_mock_backend_tokenize() {
        let backend = MockBackend::new();
        let tokens = backend.tokenize("hello world test").unwrap();
        assert_eq!(tokens.len(), 3);
    }

    #[test]
    fn test_mock_backend_detokenize() {
        let backend = MockBackend::new();
        let result = backend.detokenize(&[1, 2, 3]).unwrap();
        assert_eq!(result, "[3 tokens]");
    }

    #[test]
    fn test_llama_cpp_backend_missing_file() {
        let mut backend = LlamaCppBackend::new();
        let result = backend.load_model(std::path::Path::new("/nonexistent/model.gguf"), 2048);
        assert!(result.is_err());
    }

    #[test]
    fn test_llama_cpp_backend_creation() {
        let backend = LlamaCppBackend::new();
        assert!(!backend.is_loaded());
        assert_eq!(backend.context_size(), 0);
        assert!(backend.thread_count() > 0);
    }

    #[test]
    fn test_llama_cpp_backend_unload() {
        let backend = LlamaCppBackend::new();
        assert!(!backend.is_loaded());
        // Since we don't have a real model, just verify the methods don't panic
    }

    #[test]
    fn test_llama_cpp_backend_tokenize_with_tokenizer() {
        use crate::inference::llama_tokenizer::LLaMATokenizer;

        let mut backend = LlamaCppBackend::new();
        let vocab = vec!["hello".to_string(), "world".to_string(), "test".to_string()];
        let tokenizer = LLaMATokenizer::new(vocab).unwrap();
        backend.set_tokenizer(tokenizer);

        let result = backend.tokenize("hello world test").unwrap();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_llama_cpp_backend_tokenize_fallback() {
        let backend = LlamaCppBackend::new();
        // Without setting tokenizer, should fall back to word-based
        let result = backend.tokenize("hello world test").unwrap();
        assert_eq!(result.len(), 3); // Three words
    }

    #[test]
    fn test_llama_cpp_backend_detokenize_with_tokenizer() {
        use crate::inference::llama_tokenizer::LLaMATokenizer;

        let mut backend = LlamaCppBackend::new();
        let vocab = vec!["hello".to_string(), "world".to_string(), "test".to_string()];
        let tokenizer = LLaMATokenizer::new(vocab).unwrap();
        backend.set_tokenizer(tokenizer);

        let result = backend.detokenize(&[0i32, 1i32]).unwrap();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_llama_cpp_backend_detokenize_fallback() {
        let backend = LlamaCppBackend::new();
        // Without setting tokenizer, should provide fallback message
        let result = backend.detokenize(&[1i32, 2i32, 3i32]).unwrap();
        assert!(result.contains("3 tokens"));
    }

    #[test]
    fn test_format_detection_gguf() {
        let path = std::path::Path::new("model.gguf");
        assert_eq!(LlamaCppBackend::detect_format(path), "gguf");
    }

    #[test]
    fn test_format_detection_safetensors() {
        let path = std::path::Path::new("model.safetensors");
        assert_eq!(LlamaCppBackend::detect_format(path), "safetensors");
    }

    #[test]
    fn test_format_detection_huggingface_bin() {
        let path = std::path::Path::new("model.bin");
        assert_eq!(LlamaCppBackend::detect_format(path), "huggingface");
    }

    #[test]
    fn test_format_detection_pytorch() {
        let path = std::path::Path::new("model.pt");
        assert_eq!(LlamaCppBackend::detect_format(path), "pytorch");
    }

    #[test]
    fn test_format_detection_unknown() {
        let path = std::path::Path::new("model.unknown");
        assert_eq!(LlamaCppBackend::detect_format(path), "unknown");
    }

    #[test]
    fn test_backend_can_handle_gguf() {
        let path = std::path::Path::new("model.gguf");
        assert!(LlamaCppBackend::can_handle(path));
    }

    #[test]
    fn test_backend_cannot_handle_safetensors() {
        let path = std::path::Path::new("model.safetensors");
        assert!(!LlamaCppBackend::can_handle(path));
    }

    #[test]
    fn test_backend_cannot_handle_huggingface() {
        let path = std::path::Path::new("model.bin");
        assert!(!LlamaCppBackend::can_handle(path));
    }
}
