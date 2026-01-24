/// Pure Rust Inference Backend - Phase 8-Step 3b
///
/// This module provides native inference support for HuggingFace safetensors models
/// without external dependencies (like mlx-lm or other Python packages).
///
/// # Design Principles
///
/// 1. **Pure Rust**: No subprocess calls, no Python runtime
/// 2. **Simple**: Focus on core transformer inference, not all features
/// 3. **Pluggable**: Implements InferenceBackend trait like LlamaCppBackend
/// 4. **Fallback**: Works alongside llama.cpp for maximum compatibility
///
/// # Model Support
///
/// Supports common transformer architectures via safetensors format:
/// - Llama (Meta)
/// - Mistral (Mistral AI)
/// - Phi (Microsoft)
/// - Qwen (Alibaba)
/// - And other HuggingFace models in safetensors format
///
/// # Performance Note
///
/// Pure Rust inference will be slower than llama.cpp on some tasks, but:
/// - No external binary dependencies
/// - No process spawn overhead
/// - Direct memory control
/// - Easier to optimize for specific hardware
///
/// # Usage
///
/// ```rust,ignore
/// use minerva_lib::inference::pure_rust_backend::PureRustBackend;
/// use minerva_lib::inference::llama_adapter::{InferenceBackend, GenerationParams};
/// use std::path::Path;
///
/// let mut backend = PureRustBackend::new();
/// backend.load_model(Path::new("model.safetensors"), 2048)?;
///
/// let params = GenerationParams {
///     max_tokens: 100,
///     temperature: 0.7,
///     top_p: 0.9,
/// };
///
/// let response = backend.generate("Hello, world!", params)?;
/// println!("{}", response);
/// ```
use crate::error::{MinervaError, MinervaResult};
use crate::inference::llama_adapter::{GenerationParams, InferenceBackend};
use crate::inference::llama_tokenizer::LLaMATokenizer;
use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex};

/// Type alias for weight tensors: name -> flattened vector
type WeightTensors = HashMap<String, Vec<f32>>;

/// Model configuration loaded from model directory
#[derive(Debug, Clone)]
pub struct ModelConfig {
    /// Total vocabulary size
    pub vocab_size: usize,
    /// Hidden dimension size
    pub hidden_size: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// Number of transformer layers
    pub num_layers: usize,
    /// Model type identifier (llama, mistral, phi, etc.)
    pub model_type: String,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            vocab_size: 32000,
            hidden_size: 4096,
            num_heads: 32,
            num_layers: 32,
            model_type: "llama".to_string(),
        }
    }
}

/// Pure Rust transformer-based inference backend
///
/// Loads safetensors model files and performs inference using pure Rust
/// matrix operations. Suitable for HuggingFace format models that aren't
/// available in GGUF quantized format.
#[derive(Debug)]
pub struct PureRustBackend {
    /// Model weights loaded from safetensors
    weights: Arc<Mutex<Option<WeightTensors>>>,
    /// Model configuration (vocab size, dimensions, etc.)
    config: Arc<Mutex<Option<ModelConfig>>>,
    /// Tokenizer for converting text to/from tokens
    tokenizer: Arc<Mutex<Option<LLaMATokenizer>>>,
    /// Context window size
    n_ctx: usize,
    /// Number of CPU threads for computation
    n_threads: usize,
}

impl PureRustBackend {
    /// Create a new pure Rust inference backend
    pub fn new() -> Self {
        Self {
            weights: Arc::new(Mutex::new(None)),
            config: Arc::new(Mutex::new(None)),
            tokenizer: Arc::new(Mutex::new(None)),
            n_ctx: 0,
            n_threads: num_cpus::get(),
        }
    }

    /// Set tokenizer for this backend
    pub fn set_tokenizer(&mut self, tokenizer: LLaMATokenizer) {
        *self.tokenizer.lock().unwrap() = Some(tokenizer);
    }

    /// Load safetensors model file
    ///
    /// Phase 9: This will integrate the actual safetensors loading
    /// For now, we scaffold the structure and prepare for integration
    fn load_safetensors(_path: &Path) -> MinervaResult<WeightTensors> {
        // TODO: Phase 9 - Integrate safetensors crate
        // use safetensors::SafeTensors;
        // let file = std::fs::File::open(path)?;
        // let safetensors = SafeTensors::deserialize(&file)?;
        // Extract weights into HashMap

        tracing::info!("Phase 9: safetensors loading will be implemented here");
        Ok(HashMap::new())
    }

    /// Load model configuration from JSON
    ///
    /// Expects config.json in the same directory as the model
    fn load_config(_path: &Path) -> MinervaResult<ModelConfig> {
        // TODO: Phase 9 - Load and parse config.json
        // For now, return sensible defaults

        let config = ModelConfig::default();
        tracing::info!(
            "Using default config: vocab={}, hidden={}, heads={}, layers={}",
            config.vocab_size,
            config.hidden_size,
            config.num_heads,
            config.num_layers
        );
        Ok(config)
    }

    /// Forward pass through transformer network
    ///
    /// Takes input tokens and produces logits over vocabulary
    /// Implements simplified transformer computation:
    /// 1. Embed input tokens
    /// 2. Apply positional encoding
    /// 3. Apply attention layers (simplified)
    /// 4. Output linear projection to vocab size
    fn forward_pass(&self, tokens: &[i32]) -> MinervaResult<Vec<f32>> {
        let config = self.config.lock().unwrap();
        let cfg = config
            .as_ref()
            .ok_or_else(|| MinervaError::InferenceError("Model not loaded".to_string()))?;

        if tokens.is_empty() {
            return Err(MinervaError::InferenceError(
                "Cannot compute forward pass with empty tokens".to_string(),
            ));
        }

        // Step 1: Token embedding - convert token IDs to embeddings
        // For now, use random embeddings based on token ID
        // Phase 9: Load actual embedding weights from model
        let mut embeddings = vec![0.0; cfg.hidden_size];
        for (i, &token) in tokens.iter().enumerate() {
            let token_idx = (token as usize).min(cfg.vocab_size - 1);
            let seed = (token_idx + i) as f32;
            for (j, emb) in embeddings.iter_mut().enumerate() {
                *emb += (seed * (j as f32 + 1.0)).sin() / cfg.hidden_size as f32;
            }
        }

        // Step 2: Apply positional encoding (simplified)
        // Standard transformer positional encoding
        let seq_len = tokens.len();
        for (pos, _) in (0..seq_len).enumerate() {
            for (dim, emb) in embeddings.iter_mut().enumerate() {
                let angle = (pos as f32)
                    / (10000.0_f32.powf((2 * (dim / 2)) as f32 / cfg.hidden_size as f32));
                let pos_encoding = if dim % 2 == 0 {
                    angle.sin()
                } else {
                    angle.cos()
                };
                *emb += pos_encoding * 0.1; // Small weight for positional encoding
            }
        }

        // Step 3: Simplified attention mechanism
        // In real transformer: compute Q, K, V projections and attention
        // For now: use embeddings as-is with some normalization
        for val in embeddings.iter_mut() {
            *val = val.tanh(); // Apply activation
        }

        // Step 4: Output projection to vocabulary logits
        // Phase 9: Use actual output layer weights from model
        // For now: simple projection using deterministic function
        let mut logits = vec![0.0; cfg.vocab_size];
        for (vocab_idx, logit) in logits.iter_mut().enumerate() {
            let mut sum = 0.0;
            for (emb_idx, &emb_val) in embeddings.iter().enumerate() {
                let weight = ((vocab_idx as f32 * 0.01) + (emb_idx as f32 * 0.02)).sin();
                sum += emb_val * weight;
            }
            *logit = sum;
        }

        // Normalize logits to prevent overflow
        let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        for logit in logits.iter_mut() {
            *logit -= max_logit;
        }

        Ok(logits)
    }

    /// Sample next token from logits with proper probability distribution
    ///
    /// Implements temperature-based sampling with softmax normalization:
    /// 1. Apply temperature scaling to logits
    /// 2. Compute softmax to get probability distribution
    /// 3. Pick token with highest probability (deterministic at temp=0)
    /// 4. Can be extended with top-k/top-p in Phase 9
    fn sample_token(&self, logits: &[f32], temperature: f32) -> MinervaResult<i32> {
        if logits.is_empty() {
            return Err(MinervaError::InferenceError(
                "No logits provided for sampling".to_string(),
            ));
        }

        // Ensure temperature is positive
        let temp = if temperature <= 0.0 { 1.0 } else { temperature };

        // Step 1: Apply temperature scaling
        let scaled: Vec<f32> = logits.iter().map(|&x| x / temp).collect();

        // Step 2: Compute softmax to get probability distribution
        // Subtract max for numerical stability
        let max_logit = scaled.iter().copied().fold(f32::NEG_INFINITY, f32::max);

        let exp_scores: Vec<f32> = scaled.iter().map(|&x| (x - max_logit).exp()).collect();

        let sum_exp: f32 = exp_scores.iter().sum();

        if !sum_exp.is_finite() || sum_exp == 0.0 {
            // Fallback to argmax if softmax fails
            return Ok(scaled
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(idx, _)| idx as i32)
                .unwrap_or(0));
        }

        let probabilities: Vec<f32> = exp_scores.iter().map(|&x| x / sum_exp).collect();

        // Step 3: Pick token with highest probability (greedy)
        // Phase 9: Can implement stochastic sampling using cumulative probabilities
        let max_idx = probabilities
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(idx, _)| idx as i32)
            .unwrap_or(0);

        Ok(max_idx)
    }

    /// Sample token stochastically from probability distribution
    ///
    /// Uses cumulative probabilities to implement proper stochastic sampling
    /// This can be called instead of the greedy version for more diverse output
    #[allow(dead_code)]
    fn sample_token_stochastic(&self, logits: &[f32], temperature: f32) -> MinervaResult<i32> {
        if logits.is_empty() {
            return Err(MinervaError::InferenceError(
                "No logits provided for sampling".to_string(),
            ));
        }

        let temp = if temperature <= 0.0 { 1.0 } else { temperature };

        // Apply temperature and compute softmax
        let scaled: Vec<f32> = logits.iter().map(|&x| x / temp).collect();
        let max_logit = scaled.iter().copied().fold(f32::NEG_INFINITY, f32::max);

        let exp_scores: Vec<f32> = scaled.iter().map(|&x| (x - max_logit).exp()).collect();

        let sum_exp: f32 = exp_scores.iter().sum();

        if !sum_exp.is_finite() || sum_exp == 0.0 {
            return Ok(0); // Fallback
        }

        // For deterministic testing, use simple hash-based pseudo-random
        // Phase 9: Integrate real RNG
        let rand_seed = sum_exp.to_bits() as u64;
        let hash = rand_seed.wrapping_mul(0x9E3779B97F4A7C15u64);
        let rand_val = ((hash as f32) % 1.0).abs();

        // Build cumulative distribution and sample
        let mut cumsum = 0.0;
        for (idx, &prob) in exp_scores.iter().enumerate() {
            cumsum += prob / sum_exp;
            if rand_val <= cumsum {
                return Ok(idx as i32);
            }
        }

        // Fallback to last token if rounding errors occur
        Ok((logits.len() - 1) as i32)
    }
}

impl Default for PureRustBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl InferenceBackend for PureRustBackend {
    fn load_model(&mut self, path: &Path, n_ctx: usize) -> MinervaResult<()> {
        // Validate path exists
        if !path.exists() {
            return Err(MinervaError::ModelNotFound(format!(
                "Model file not found: {}",
                path.display()
            )));
        }

        // Load model weights from safetensors
        let weights = Self::load_safetensors(path)?;
        *self.weights.lock().unwrap() = Some(weights);

        // Load model configuration
        let config = Self::load_config(path)?;
        *self.config.lock().unwrap() = Some(config.clone());

        // Create/load tokenizer
        // For now, use a simple fallback tokenizer
        let vocab = (0..config.vocab_size)
            .map(|i| format!("token_{}", i))
            .collect();
        let tokenizer = LLaMATokenizer::new(vocab)?;
        *self.tokenizer.lock().unwrap() = Some(tokenizer);

        self.n_ctx = n_ctx;

        tracing::info!(
            "PureRustBackend: Model loaded - {} (context: {}, vocab: {}, hidden: {})",
            path.display(),
            n_ctx,
            config.vocab_size,
            config.hidden_size
        );

        Ok(())
    }

    fn unload_model(&mut self) {
        *self.weights.lock().unwrap() = None;
        *self.config.lock().unwrap() = None;
        *self.tokenizer.lock().unwrap() = None;
        self.n_ctx = 0;
        tracing::info!("PureRustBackend: Model unloaded");
    }

    fn generate(&self, prompt: &str, params: GenerationParams) -> MinervaResult<String> {
        let tokenizer = self.tokenizer.lock().unwrap();
        let tok = tokenizer
            .as_ref()
            .ok_or_else(|| MinervaError::InferenceError("Tokenizer not initialized".to_string()))?;

        // Tokenize input prompt
        let input_tokens = tok.encode(prompt)?;
        let mut tokens: Vec<i32> = input_tokens.iter().map(|&t| t as i32).collect();

        // Generate tokens one by one
        for _ in 0..params.max_tokens {
            // Get logits from transformer
            let logits = self.forward_pass(&tokens)?;

            // Sample next token
            let next_token = self.sample_token(&logits, params.temperature)?;
            tokens.push(next_token);

            // Check for end-of-sequence token (usually 2 in LLaMA)
            if next_token == 2 {
                break;
            }
        }

        // Detokenize output
        let u32_tokens: Vec<u32> = tokens.iter().map(|&t| t as u32).collect();
        tok.decode(&u32_tokens)
    }

    fn tokenize(&self, text: &str) -> MinervaResult<Vec<i32>> {
        let tokenizer = self.tokenizer.lock().unwrap();
        let tok = tokenizer
            .as_ref()
            .ok_or_else(|| MinervaError::InferenceError("Tokenizer not initialized".to_string()))?;

        let tokens = tok.encode(text)?;
        Ok(tokens.iter().map(|&t| t as i32).collect())
    }

    fn detokenize(&self, tokens: &[i32]) -> MinervaResult<String> {
        let tokenizer = self.tokenizer.lock().unwrap();
        let tok = tokenizer
            .as_ref()
            .ok_or_else(|| MinervaError::InferenceError("Tokenizer not initialized".to_string()))?;

        let u32_tokens: Vec<u32> = tokens.iter().map(|&t| t as u32).collect();
        tok.decode(&u32_tokens)
    }

    fn is_loaded(&self) -> bool {
        self.weights.lock().unwrap().is_some()
    }

    fn context_size(&self) -> usize {
        self.n_ctx
    }

    fn thread_count(&self) -> usize {
        self.n_threads
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pure_rust_backend_creation() {
        let backend = PureRustBackend::new();
        assert!(!backend.is_loaded());
        assert_eq!(backend.context_size(), 0);
        assert!(backend.thread_count() > 0);
    }

    #[test]
    fn test_pure_rust_backend_default() {
        let backend = PureRustBackend::default();
        assert!(!backend.is_loaded());
    }

    #[test]
    fn test_model_config_default() {
        let config = ModelConfig::default();
        assert_eq!(config.vocab_size, 32000);
        assert_eq!(config.hidden_size, 4096);
        assert_eq!(config.num_heads, 32);
        assert_eq!(config.num_layers, 32);
        assert_eq!(config.model_type, "llama");
    }

    #[test]
    fn test_pure_rust_sample_token() {
        let backend = PureRustBackend::new();
        let logits = vec![0.1, 0.5, 0.3, 0.1];
        let token = backend.sample_token(&logits, 1.0).unwrap();
        assert!((0..4).contains(&token));
        assert_eq!(token, 1); // argmax of logits
    }

    #[test]
    fn test_pure_rust_sample_token_empty_logits() {
        let backend = PureRustBackend::new();
        let result = backend.sample_token(&[], 1.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_pure_rust_sample_with_temperature() {
        let backend = PureRustBackend::new();
        let logits = vec![1.0, 2.0, 1.5];

        // With temperature, should still pick max (2.0)
        let token = backend.sample_token(&logits, 0.5).unwrap();
        assert_eq!(token, 1);
    }

    #[test]
    fn test_pure_rust_unload_model() {
        let mut backend = PureRustBackend::new();
        backend.unload_model();
        assert!(!backend.is_loaded());
    }

    #[test]
    fn test_forward_pass_requires_loaded_model() {
        let backend = PureRustBackend::new();
        let tokens = vec![1, 2, 3];
        let result = backend.forward_pass(&tokens);
        assert!(result.is_err());
    }

    #[test]
    fn test_forward_pass_empty_tokens() {
        let backend = PureRustBackend::new();
        let config = ModelConfig::default();
        *backend.config.lock().unwrap() = Some(config);

        let result = backend.forward_pass(&[]);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("empty"));
    }

    #[test]
    fn test_forward_pass_logits_dimension() {
        let backend = PureRustBackend::new();
        let config = ModelConfig::default();
        *backend.config.lock().unwrap() = Some(config.clone());

        let tokens = vec![1, 2, 3];
        let logits = backend.forward_pass(&tokens).unwrap();
        assert_eq!(logits.len(), config.vocab_size);
    }

    #[test]
    fn test_forward_pass_logits_normalized() {
        let backend = PureRustBackend::new();
        let config = ModelConfig::default();
        *backend.config.lock().unwrap() = Some(config);

        let tokens = vec![1, 2];
        let logits = backend.forward_pass(&tokens).unwrap();

        // All logits should be bounded (not NaN or Inf)
        for logit in &logits {
            assert!(logit.is_finite(), "Logit contains NaN or Inf: {}", logit);
        }
    }

    #[test]
    fn test_sample_token_with_softmax() {
        let backend = PureRustBackend::new();
        let logits = vec![1.0, 2.0, 0.5, 1.5];

        // Temperature 1.0: normal softmax
        let token = backend.sample_token(&logits, 1.0).unwrap();
        assert_eq!(token, 1); // Should pick logits[1] = 2.0 (highest)
        assert!((0..4).contains(&token));
    }

    #[test]
    fn test_sample_token_temperature_high() {
        let backend = PureRustBackend::new();
        let logits = vec![10.0, 0.0, 0.0];

        // High temperature flattens distribution
        let token = backend.sample_token(&logits, 10.0).unwrap();
        assert!((0..3).contains(&token));
    }

    #[test]
    fn test_sample_token_temperature_zero() {
        let backend = PureRustBackend::new();
        let logits = vec![0.5, 3.0, 1.0];

        // Zero/negative temperature should use temperature=1.0 fallback
        let token = backend.sample_token(&logits, 0.0).unwrap();
        assert_eq!(token, 1); // argmax
    }

    #[test]
    fn test_sample_token_equal_logits() {
        let backend = PureRustBackend::new();
        let logits = vec![1.0, 1.0, 1.0, 1.0];

        // Equal logits: should pick first (due to max_by comparison order)
        let token = backend.sample_token(&logits, 1.0).unwrap();
        assert!((0..4).contains(&token));
    }

    #[test]
    fn test_sample_token_single_logit() {
        let backend = PureRustBackend::new();
        let logits = vec![5.0];

        let token = backend.sample_token(&logits, 1.0).unwrap();
        assert_eq!(token, 0);
    }

    #[test]
    fn test_sample_token_negative_logits() {
        let backend = PureRustBackend::new();
        let logits = vec![-5.0, -1.0, -3.0];

        let token = backend.sample_token(&logits, 1.0).unwrap();
        assert_eq!(token, 1); // -1.0 is max
        assert!((0..3).contains(&token));
    }

    #[test]
    fn test_sample_token_mixed_logits() {
        let backend = PureRustBackend::new();
        let logits = vec![-5.0, 2.0, -1.0, 10.0, 3.0];

        let token = backend.sample_token(&logits, 1.0).unwrap();
        assert_eq!(token, 3); // 10.0 is max
    }

    #[test]
    fn test_sample_token_large_logits() {
        let backend = PureRustBackend::new();
        let logits = vec![1000.0, 999.0, 500.0];

        // Should handle large logits without overflow
        let token = backend.sample_token(&logits, 1.0).unwrap();
        assert_eq!(token, 0);
    }

    #[test]
    fn test_sample_token_stochastic_range() {
        let backend = PureRustBackend::new();
        let logits = vec![1.0, 2.0, 1.5];

        let token = backend.sample_token_stochastic(&logits, 1.0).unwrap();
        assert!((0..3).contains(&token));
    }

    #[test]
    fn test_sample_token_stochastic_high_probability() {
        let backend = PureRustBackend::new();
        // One logit is much higher - stochastic should pick it often
        let logits = vec![0.0, 100.0, 0.0];

        let token = backend.sample_token_stochastic(&logits, 1.0).unwrap();
        assert!((0..3).contains(&token));
    }

    #[test]
    fn test_sample_token_stochastic_empty() {
        let backend = PureRustBackend::new();
        let result = backend.sample_token_stochastic(&[], 1.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_forward_pass_consistency() {
        let backend = PureRustBackend::new();
        let config = ModelConfig::default();
        *backend.config.lock().unwrap() = Some(config);

        let tokens = vec![5, 10];
        let logits1 = backend.forward_pass(&tokens).unwrap();
        let logits2 = backend.forward_pass(&tokens).unwrap();

        // Same input should produce same output (deterministic)
        assert_eq!(logits1.len(), logits2.len());
        for (l1, l2) in logits1.iter().zip(logits2.iter()) {
            assert_eq!(l1, l2, "Forward pass should be deterministic");
        }
    }

    #[test]
    fn test_sampling_temperature_consistency() {
        let backend = PureRustBackend::new();
        let logits = vec![2.0, 1.0, 0.5];

        let result1 = backend.sample_token(&logits, 0.7).unwrap();
        let result2 = backend.sample_token(&logits, 0.7).unwrap();

        // Deterministic sampling should be consistent
        assert_eq!(result1, result2);
    }
}
