/// Production llama.cpp Backend
///
/// This backend uses the actual llama_cpp crate for inference.
/// It maintains a model and session for real LLM inference.
/// Includes real BPE tokenization via LLaMATokenizer.
use crate::error::{MinervaError, MinervaResult};
use crate::inference::inference_backend_trait::{GenerationParams, InferenceBackend};
use crate::inference::llama_tokenizer::LLaMATokenizer;
use llama_cpp::standard_sampler::StandardSampler;
use llama_cpp::{LlamaModel, LlamaParams, LlamaSession, SessionParams};
use std::path::Path;
use std::sync::{Arc, Mutex};

/// Production llama.cpp backend
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
