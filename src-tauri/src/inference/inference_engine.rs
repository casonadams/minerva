use super::transformer_components::{
    Activation, EmbeddingConfig, PositionConfig, TransformerBlockConfig, TransformerBlockWeights,
    add_position_encoding, create_position_encoding, embed_tokens, transformer_block,
};
use super::transformer_layers::{LayerNormConfig as LNCfg, layer_norm};
/// Complete Inference Engine - Phase 9 Day 4
///
/// This module integrates all transformer components into a single,
/// production-ready inference pipeline for running language models.
///
/// The forward pass combines:
/// 1. Token embedding
/// 2. Position encoding
/// 3. Transformer blocks (repeated num_layers times)
/// 4. Final layer normalization
/// 5. Output projection to vocabulary logits
///
/// # Example Flow
///
/// ```text
/// Input: [1, 5, 42] (token IDs)
///     ↓
/// Embedding: (3, 4096)
///     ↓
/// Position Encoding + Add: (3, 4096)
///     ↓
/// Transformer Blocks (32 layers): (3, 4096) → (3, 4096)
///     ↓
/// Final LayerNorm: (3, 4096)
///     ↓
/// Output Projection: (3, 4096) @ (4096, vocab_size) = (3, vocab_size)
///     ↓
/// Softmax: (3, vocab_size) probabilities
/// ```
use crate::error::{MinervaError, MinervaResult};

// ============================================================================
// Inference Engine Configuration
// ============================================================================

/// Complete inference engine configuration
#[derive(Debug, Clone)]
pub struct InferenceEngineConfig {
    /// Vocabulary size (token embedding matrix rows)
    pub vocab_size: usize,
    /// Hidden dimension (embedding dimension)
    pub hidden_size: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// Number of transformer layers
    pub num_layers: usize,
    /// FFN intermediate size (usually 4 * hidden_size)
    pub intermediate_size: usize,
    /// Activation function to use
    pub activation: Activation,
    /// Whether to use causal masking (autoregressive)
    pub causal: bool,
    /// Layer norm epsilon for numerical stability
    pub eps: f32,
    /// Maximum sequence length supported
    pub max_seq_len: usize,
}

impl InferenceEngineConfig {
    /// Create a config for a LLaMA-like model
    pub fn llama(vocab_size: usize, hidden_size: usize, num_heads: usize) -> Self {
        Self {
            vocab_size,
            hidden_size,
            num_heads,
            num_layers: 32,
            intermediate_size: (hidden_size * 8) / 3, // LLaMA uses 8/3
            activation: Activation::SiLU,
            causal: true,
            eps: 1e-6,
            max_seq_len: 2048,
        }
    }

    /// Create a config for a BERT-like model (non-causal)
    pub fn bert(vocab_size: usize, hidden_size: usize, num_heads: usize) -> Self {
        Self {
            vocab_size,
            hidden_size,
            num_heads,
            num_layers: 12,
            intermediate_size: hidden_size * 4,
            activation: Activation::GELU,
            causal: false,
            eps: 1e-12,
            max_seq_len: 512,
        }
    }

    /// Create a config for a small debug model
    pub fn tiny(vocab_size: usize) -> Self {
        Self {
            vocab_size,
            hidden_size: 64,
            num_heads: 2,
            num_layers: 2,
            intermediate_size: 128,
            activation: Activation::ReLU,
            causal: true,
            eps: 1e-6,
            max_seq_len: 128,
        }
    }
}

// ============================================================================
// Model Weights Container
// ============================================================================

/// Container for all model weights
///
/// This holds all the learnable parameters needed for inference.
/// In production, these would be loaded from checkpoint files.
#[derive(Debug, Clone)]
pub struct ModelWeights {
    /// Token embedding matrix: (vocab_size, hidden_size)
    pub embeddings: Vec<f32>,

    /// Per-layer weights for transformer blocks
    pub layers: Vec<LayerWeights>,

    /// Final layer norm scale: (hidden_size,)
    pub final_norm_scale: Vec<f32>,

    /// Output projection: (hidden_size, vocab_size)
    pub output_proj: Vec<f32>,
}

/// Weights for a single transformer layer
#[derive(Debug, Clone)]
pub struct LayerWeights {
    /// Attention layer norm scale: (hidden_size,)
    pub attn_norm_scale: Vec<f32>,

    /// FFN layer norm scale: (hidden_size,)
    pub ffn_norm_scale: Vec<f32>,

    /// FFN up projection: (hidden_size, intermediate_size)
    pub ff_up: Vec<f32>,

    /// FFN down projection: (intermediate_size, hidden_size)
    pub ff_down: Vec<f32>,
}

// ============================================================================
// Inference Engine
// ============================================================================

/// Complete inference engine for language model forward pass
pub struct InferenceEngine {
    config: InferenceEngineConfig,
    weights: ModelWeights,
}

impl InferenceEngine {
    /// Create a new inference engine
    pub fn new(config: InferenceEngineConfig, weights: ModelWeights) -> MinervaResult<Self> {
        // Validate weight shapes
        if weights.embeddings.len() != config.vocab_size * config.hidden_size {
            return Err(MinervaError::InferenceError(format!(
                "Embedding shape mismatch: expected {}, got {}",
                config.vocab_size * config.hidden_size,
                weights.embeddings.len()
            )));
        }

        if weights.layers.len() != config.num_layers {
            return Err(MinervaError::InferenceError(format!(
                "Number of layers mismatch: expected {}, got {}",
                config.num_layers,
                weights.layers.len()
            )));
        }

        if weights.final_norm_scale.len() != config.hidden_size {
            return Err(MinervaError::InferenceError(
                "Final norm scale shape mismatch".to_string(),
            ));
        }

        if weights.output_proj.len() != config.hidden_size * config.vocab_size {
            return Err(MinervaError::InferenceError(
                "Output projection shape mismatch".to_string(),
            ));
        }

        Ok(Self { config, weights })
    }

    /// Run forward pass on token sequence
    ///
    /// # Arguments
    /// * `tokens`: Token IDs (0 to vocab_size-1)
    ///
    /// # Returns
    /// Logits for each token position: (seq_len, vocab_size)
    pub fn forward(&self, tokens: &[usize]) -> MinervaResult<Vec<f32>> {
        let seq_len = tokens.len();

        if seq_len == 0 {
            return Err(MinervaError::InferenceError(
                "Empty token sequence".to_string(),
            ));
        }

        if seq_len > self.config.max_seq_len {
            return Err(MinervaError::InferenceError(format!(
                "Sequence length {} exceeds max {}",
                seq_len, self.config.max_seq_len
            )));
        }

        // Step 1: Token embedding
        let embed_config = EmbeddingConfig {
            vocab_size: self.config.vocab_size,
            hidden_size: self.config.hidden_size,
        };
        let mut x = embed_tokens(tokens, &self.weights.embeddings, &embed_config)?;

        // Step 2: Position encoding
        let pos_config = PositionConfig {
            seq_len,
            hidden_size: self.config.hidden_size,
            base: 10000.0,
        };
        let position_encoding = create_position_encoding(&pos_config);
        x = add_position_encoding(&x, &position_encoding)?;

        // Step 3: Transformer blocks
        for layer_idx in 0..self.config.num_layers {
            let layer_weights = &self.weights.layers[layer_idx];

            let block_config = TransformerBlockConfig {
                seq_len,
                hidden_size: self.config.hidden_size,
                num_heads: self.config.num_heads,
                intermediate_size: self.config.intermediate_size,
                activation: self.config.activation,
                causal: self.config.causal,
                eps: self.config.eps,
            };

            let block_weights = TransformerBlockWeights {
                attn_scale: Some(&layer_weights.attn_norm_scale),
                ff_up: &layer_weights.ff_up,
                ff_down: &layer_weights.ff_down,
            };
            x = transformer_block(&x, &block_weights, &block_config)?;
        }

        // Step 4: Final layer norm
        let final_norm_config = LNCfg {
            seq_len,
            hidden_size: self.config.hidden_size,
            scale: Some(self.weights.final_norm_scale.clone()),
            eps: self.config.eps,
        };
        x = layer_norm(&x, &final_norm_config)?;

        // Step 5: Output projection
        let mut logits = vec![0.0; seq_len * self.config.vocab_size];

        for i in 0..seq_len {
            for j in 0..self.config.vocab_size {
                let mut sum = 0.0;
                for k in 0..self.config.hidden_size {
                    sum += x[i * self.config.hidden_size + k]
                        * self.weights.output_proj[k * self.config.vocab_size + j];
                }
                logits[i * self.config.vocab_size + j] = sum;
            }
        }

        Ok(logits)
    }

    /// Forward pass and convert to probabilities
    pub fn forward_with_softmax(&self, tokens: &[usize]) -> MinervaResult<Vec<f32>> {
        let logits = self.forward(tokens)?;
        let seq_len = tokens.len();
        let vocab_size = self.config.vocab_size;

        let mut probs = logits.clone();

        // Softmax per token position
        for i in 0..seq_len {
            let start = i * vocab_size;
            let end = start + vocab_size;
            let row = &mut probs[start..end];

            // Find max for numerical stability
            let max_val = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);

            // Compute exp and sum
            let mut sum_exp = 0.0;
            for val in row.iter_mut() {
                *val = (*val - max_val).exp();
                sum_exp += *val;
            }

            // Normalize
            if sum_exp > 0.0 {
                for val in row {
                    *val /= sum_exp;
                }
            }
        }

        Ok(probs)
    }

    /// Get the configuration
    pub fn config(&self) -> &InferenceEngineConfig {
        &self.config
    }

    /// Get the weights
    pub fn weights(&self) -> &ModelWeights {
        &self.weights
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn create_dummy_weights(config: &InferenceEngineConfig) -> ModelWeights {
        let mut layers = Vec::new();

        for _ in 0..config.num_layers {
            layers.push(LayerWeights {
                attn_norm_scale: vec![1.0; config.hidden_size],
                ffn_norm_scale: vec![1.0; config.hidden_size],
                ff_up: vec![0.1; config.hidden_size * config.intermediate_size],
                ff_down: vec![0.1; config.intermediate_size * config.hidden_size],
            });
        }

        ModelWeights {
            embeddings: vec![0.1; config.vocab_size * config.hidden_size],
            layers,
            final_norm_scale: vec![1.0; config.hidden_size],
            output_proj: vec![0.01; config.hidden_size * config.vocab_size],
        }
    }

    #[test]
    fn test_llama_config_creation() {
        let config = InferenceEngineConfig::llama(32000, 4096, 32);
        assert_eq!(config.vocab_size, 32000);
        assert_eq!(config.hidden_size, 4096);
        assert_eq!(config.num_heads, 32);
        assert_eq!(config.num_layers, 32);
        assert!(config.causal);
    }

    #[test]
    fn test_bert_config_creation() {
        let config = InferenceEngineConfig::bert(30522, 768, 12);
        assert_eq!(config.vocab_size, 30522);
        assert!(!config.causal); // BERT is non-causal
    }

    #[test]
    fn test_tiny_config_creation() {
        let config = InferenceEngineConfig::tiny(1000);
        assert_eq!(config.num_layers, 2);
        assert_eq!(config.hidden_size, 64);
    }

    #[test]
    fn test_engine_creation_valid() {
        let config = InferenceEngineConfig::tiny(100);
        let weights = create_dummy_weights(&config);
        let engine = InferenceEngine::new(config, weights);
        assert!(engine.is_ok());
    }

    #[test]
    fn test_engine_creation_invalid_embeddings() {
        let mut config = InferenceEngineConfig::tiny(100);
        let mut weights = create_dummy_weights(&config);

        // Wrong embedding shape
        weights.embeddings = vec![0.1; 50];
        config.vocab_size = 100;

        let result = InferenceEngine::new(config, weights);
        assert!(result.is_err());
    }

    #[test]
    fn test_forward_pass_shapes() {
        let config = InferenceEngineConfig::tiny(100);
        let weights = create_dummy_weights(&config);
        let engine = InferenceEngine::new(config, weights).unwrap();

        let tokens = vec![1, 2, 3];
        let logits = engine.forward(&tokens).unwrap();

        // Should have shape (seq_len, vocab_size)
        assert_eq!(logits.len(), 3 * 100);
    }

    #[test]
    fn test_forward_pass_empty_sequence() {
        let config = InferenceEngineConfig::tiny(100);
        let weights = create_dummy_weights(&config);
        let engine = InferenceEngine::new(config, weights).unwrap();

        let result = engine.forward(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_forward_pass_sequence_too_long() {
        let config = InferenceEngineConfig::tiny(100);
        let weights = create_dummy_weights(&config);
        let engine = InferenceEngine::new(config, weights).unwrap();

        let tokens = vec![1; engine.config().max_seq_len + 1];
        let result = engine.forward(&tokens);
        assert!(result.is_err());
    }

    #[test]
    fn test_forward_with_softmax_shapes() {
        let config = InferenceEngineConfig::tiny(100);
        let weights = create_dummy_weights(&config);
        let engine = InferenceEngine::new(config, weights).unwrap();

        let tokens = vec![1, 2, 3];
        let probs = engine.forward_with_softmax(&tokens).unwrap();

        assert_eq!(probs.len(), 3 * 100);
    }

    #[test]
    fn test_softmax_probabilities_sum_to_one() {
        let config = InferenceEngineConfig::tiny(100);
        let weights = create_dummy_weights(&config);
        let engine = InferenceEngine::new(config, weights).unwrap();

        let tokens = vec![1, 2];
        let probs = engine.forward_with_softmax(&tokens).unwrap();

        // Check that each position sums to approximately 1.0
        for pos in 0..2 {
            let start = pos * 100;
            let end = start + 100;
            let sum: f32 = probs[start..end].iter().sum();
            assert!((sum - 1.0).abs() < 0.001, "Position {} sum: {}", pos, sum);
        }
    }

    #[test]
    fn test_forward_numerical_stability() {
        let config = InferenceEngineConfig::tiny(100);
        let weights = create_dummy_weights(&config);
        let engine = InferenceEngine::new(config, weights).unwrap();

        let tokens = vec![1, 2, 3];
        let logits = engine.forward(&tokens).unwrap();

        // All values should be finite
        for &val in &logits {
            assert!(val.is_finite(), "Found non-finite value: {}", val);
        }
    }

    #[test]
    fn test_forward_single_token() {
        let config = InferenceEngineConfig::tiny(50);
        let weights = create_dummy_weights(&config);
        let engine = InferenceEngine::new(config, weights).unwrap();

        let logits = engine.forward(&[1]).unwrap();
        assert_eq!(logits.len(), 50);
    }

    #[test]
    fn test_config_consistency() {
        let config = InferenceEngineConfig::tiny(1000);
        let weights = create_dummy_weights(&config);
        let engine = InferenceEngine::new(config.clone(), weights).unwrap();

        assert_eq!(engine.config().vocab_size, 1000);
        assert_eq!(engine.config().hidden_size, 64);
        assert_eq!(engine.config().num_heads, 2);
    }
}
