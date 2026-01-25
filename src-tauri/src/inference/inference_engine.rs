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
use super::engine_config::InferenceEngineConfig;
use super::model_weights::ModelWeights;
use super::transformer_components::{
    EmbeddingConfig, PositionConfig, TransformerBlockConfig, TransformerBlockWeights,
    add_position_encoding, create_position_encoding, embed_tokens, transformer_block,
};
use super::transformer_layers::{LayerNormConfig as LNCfg, layer_norm};
use crate::error::{MinervaError, MinervaResult};

/// Complete inference engine for language model forward pass
pub struct InferenceEngine {
    config: InferenceEngineConfig,
    weights: ModelWeights,
}

impl InferenceEngine {
    /// Create a new inference engine
    ///
    /// Validates all weight shapes against configuration.
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
