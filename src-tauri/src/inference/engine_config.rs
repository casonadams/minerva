/// Inference Engine Configuration
///
/// Defines model architecture parameters and preset configurations
/// for different model architectures (LLaMA, BERT, etc.)
use super::transformer_components::Activation;

/// Complete inference engine configuration
///
/// Contains all hyperparameters needed to define a transformer model architecture.
/// This includes model size, layer count, activation functions, and sequence limits.
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
    ///
    /// LLaMA-2 7B parameters:
    /// - vocab_size: 32000
    /// - hidden_size: 4096
    /// - num_heads: 32
    /// - intermediate_size uses 8/3 ratio
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
    ///
    /// BERT-base parameters:
    /// - vocab_size: 30522
    /// - hidden_size: 768
    /// - num_heads: 12
    /// - Uses GELU activation
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
    ///
    /// Tiny model for testing and prototyping:
    /// - Small embedding dimension: 64
    /// - Few attention heads: 2
    /// - Few layers: 2
    /// - Short sequence: 128
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
