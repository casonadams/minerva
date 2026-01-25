/// Model Weights Container
///
/// Holds all learnable parameters for a transformer model.
/// In production, these are loaded from checkpoint files.
/// This module provides the data structures to organize and access weights.
/// Container for all model weights
///
/// This holds all the learnable parameters needed for inference.
/// Weights are organized hierarchically:
/// - Embedding weights (shared token embeddings)
/// - Per-layer transformer weights
/// - Final layer normalization
/// - Output projection to vocabulary
#[derive(Debug, Clone)]
pub struct ModelWeights {
    /// Token embedding matrix: (vocab_size, hidden_size)
    /// Maps token IDs to dense embeddings
    pub embeddings: Vec<f32>,

    /// Per-layer weights for transformer blocks
    pub layers: Vec<LayerWeights>,

    /// Final layer norm scale: (hidden_size,)
    /// Applied after all transformer blocks
    pub final_norm_scale: Vec<f32>,

    /// Output projection: (hidden_size, vocab_size)
    /// Projects final hidden state to vocabulary logits
    pub output_proj: Vec<f32>,
}

/// Weights for a single transformer layer
///
/// Contains all parameters for one transformer block:
/// - Attention layer normalization
/// - FFN layer normalization
/// - FFN up and down projections
#[derive(Debug, Clone)]
pub struct LayerWeights {
    /// Attention layer norm scale: (hidden_size,)
    pub attn_norm_scale: Vec<f32>,

    /// FFN layer norm scale: (hidden_size,)
    pub ffn_norm_scale: Vec<f32>,

    /// FFN up projection: (hidden_size, intermediate_size)
    /// Expands hidden representation
    pub ff_up: Vec<f32>,

    /// FFN down projection: (intermediate_size, hidden_size)
    /// Projects back to hidden dimension
    pub ff_down: Vec<f32>,
}
