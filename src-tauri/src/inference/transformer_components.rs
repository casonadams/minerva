use super::transformer_layers::{layer_norm, multi_head_attention};
use super::transformer_layers::{LayerNormConfig as LNCfg, MultiHeadConfig};
/// Transformer Components - Phase 9 Day 3
///
/// This module completes the transformer architecture with:
/// - Embedding layers (token + position encoding)
/// - Feedforward networks with activation functions
/// - Complete transformer blocks combining attention + feedforward
/// - Activation functions (GELU, SiLU, ReLU)
///
/// These components work with the core layers from `transformer_layers.rs`
/// to form the complete forward pass pipeline.
///
/// # Pipeline
///
/// ```text
/// Input tokens (seq_len,)
///     ↓
/// Embedding: (seq_len,) → (seq_len, hidden_size)
///     ↓
/// Position Encoding: add positional information
///     ↓
/// TransformerBlock (num_layers times):
///     ├─ LayerNorm → MultiHeadAttention → Residual
///     ├─ LayerNorm → Feedforward → Residual
///     └─ Repeat
///     ↓
/// Final LayerNorm
///     ↓
/// Output projection: (seq_len, hidden_size) → logits
/// ```
use crate::error::{MinervaError, MinervaResult};

// ============================================================================
// Activation Functions
// ============================================================================

/// GELU activation function
///
/// Gaussian Error Linear Unit: a smooth approximation of ReLU
/// Formula: x * Φ(x) where Φ is the cumulative normal distribution
/// Approximation: x * 0.5 * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
///
/// Used in most modern transformers (BERT, GPT, LLaMA)
#[inline]
fn gelu(x: f32) -> f32 {
    const COEFF: f32 = 0.044_715;
    const SQRT_2_PI: f32 = 0.797_885; // sqrt(2/π)

    let x_cube = x * x * x;
    let inner = SQRT_2_PI * (x + COEFF * x_cube);
    x * 0.5 * (1.0 + inner.tanh())
}

/// SiLU (Swish) activation function
///
/// Self-Gated Linear Unit: x * sigmoid(x)
/// Used in modern models like LLaMA
#[inline]
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// ReLU activation function
///
/// Rectified Linear Unit: max(0, x)
/// Classic activation, still used in some FFN layers
#[inline]
fn relu(x: f32) -> f32 {
    x.max(0.0)
}

/// Activation function selector
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Activation {
    GELU,
    SiLU,
    ReLU,
}

/// Apply activation function to entire vector
fn apply_activation(input: &[f32], activation: Activation) -> Vec<f32> {
    match activation {
        Activation::GELU => input.iter().map(|&x| gelu(x)).collect(),
        Activation::SiLU => input.iter().map(|&x| silu(x)).collect(),
        Activation::ReLU => input.iter().map(|&x| relu(x)).collect(),
    }
}

// ============================================================================
// Embedding Layer
// ============================================================================

/// Embedding configuration
#[derive(Debug, Clone)]
pub struct EmbeddingConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
}

/// Token embedding layer
///
/// Maps token IDs to dense embeddings
///
/// # Arguments
/// * `tokens`: Token IDs (shape: seq_len)
/// * `embeddings`: Full embedding matrix (shape: vocab_size × hidden_size)
/// * `config`: Embedding configuration
///
/// # Returns
/// Embedded representations (shape: seq_len × hidden_size)
pub fn embed_tokens(
    tokens: &[usize],
    embeddings: &[f32],
    config: &EmbeddingConfig,
) -> MinervaResult<Vec<f32>> {
    let EmbeddingConfig {
        vocab_size,
        hidden_size,
    } = config;

    if embeddings.len() != vocab_size * hidden_size {
        return Err(MinervaError::InferenceError(format!(
            "Embedding matrix shape mismatch: expected {}, got {}",
            vocab_size * hidden_size,
            embeddings.len()
        )));
    }

    let mut output = vec![0.0; tokens.len() * hidden_size];

    for (i, &token_id) in tokens.iter().enumerate() {
        if token_id >= *vocab_size {
            return Err(MinervaError::InferenceError(format!(
                "Token ID {} out of vocabulary size {}",
                token_id, vocab_size
            )));
        }

        // Copy embedding row for this token
        let start = token_id * hidden_size;
        let end = start + hidden_size;
        output[i * hidden_size..(i + 1) * hidden_size].copy_from_slice(&embeddings[start..end]);
    }

    Ok(output)
}

// ============================================================================
// Position Encoding
// ============================================================================

/// Positional encoding configuration
#[derive(Debug, Clone, Copy)]
pub struct PositionConfig {
    pub seq_len: usize,
    pub hidden_size: usize,
    pub base: f32, // Usually 10000 for absolute, 100000 for RoPE
}

/// Absolute positional encoding
///
/// Creates sine/cosine positional embeddings (like in original Transformer)
/// Formula: PE(pos, 2i) = sin(pos / 10000^(2i / d))
///          PE(pos, 2i+1) = cos(pos / 10000^(2i / d))
///
/// # Arguments
/// * `seq_len`: Sequence length
/// * `hidden_size`: Hidden dimension
/// * `base`: Base for exponential (typically 10000)
///
/// # Returns
/// Positional encodings (shape: seq_len × hidden_size)
pub fn create_position_encoding(config: &PositionConfig) -> Vec<f32> {
    let PositionConfig {
        seq_len,
        hidden_size,
        base,
    } = config;

    let seq_len_val = *seq_len;
    let hidden_size_val = *hidden_size;
    let base_val = *base;

    let mut pe = vec![0.0; seq_len_val * hidden_size_val];

    for pos in 0..seq_len_val {
        for i in 0..hidden_size_val {
            let div_term = (base_val).powf(2.0 * (i as f32 / 2.0) / hidden_size_val as f32);
            let arg = pos as f32 / div_term;

            if i % 2 == 0 {
                // Even indices: sin
                pe[pos * hidden_size_val + i] = arg.sin();
            } else {
                // Odd indices: cos
                pe[pos * hidden_size_val + i] = arg.cos();
            }
        }
    }

    pe
}

/// Add positional encoding to embeddings
pub fn add_position_encoding(
    embeddings: &[f32],
    position_encoding: &[f32],
) -> MinervaResult<Vec<f32>> {
    if embeddings.len() != position_encoding.len() {
        return Err(MinervaError::InferenceError(format!(
            "Shape mismatch: embeddings {} vs positions {}",
            embeddings.len(),
            position_encoding.len()
        )));
    }

    let mut result = embeddings.to_vec();
    for (i, &pe) in position_encoding.iter().enumerate() {
        result[i] += pe;
    }

    Ok(result)
}

// ============================================================================
// Feedforward Network
// ============================================================================

/// Feedforward network configuration
#[derive(Debug, Clone, Copy)]
pub struct FeedforwardConfig {
    pub seq_len: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub activation: Activation,
}

/// Feedforward weights (up and down projections)
#[derive(Debug)]
pub struct FeedforwardWeights<'a> {
    pub up: &'a [f32],
    pub down: &'a [f32],
}

/// Feedforward network layer
///
/// Implements: Dense(hidden) → Activation → Dense(hidden)
/// Typically: hidden_size → 4*hidden_size → hidden_size
///
/// # Arguments
/// * `input`: Sequence of tokens (shape: seq_len × hidden_size)
/// * `weights`: Up and down projection weights
/// * `config`: Feedforward configuration
///
/// # Returns
/// Feedforward output (shape: seq_len × hidden_size)
pub fn feedforward(
    input: &[f32],
    weights: &FeedforwardWeights,
    config: &FeedforwardConfig,
) -> MinervaResult<Vec<f32>> {
    let w_up = weights.up;
    let w_down = weights.down;
    let FeedforwardConfig {
        seq_len,
        hidden_size,
        intermediate_size,
        activation,
    } = config;

    let seq_len_val = *seq_len;
    let hidden_size_val = *hidden_size;
    let intermediate_size_val = *intermediate_size;
    let activation_val = *activation;

    if input.len() != seq_len_val * hidden_size_val {
        return Err(MinervaError::InferenceError(
            "Input shape mismatch".to_string(),
        ));
    }

    if w_up.len() != hidden_size_val * intermediate_size_val {
        return Err(MinervaError::InferenceError(
            "Up weight shape mismatch".to_string(),
        ));
    }

    if w_down.len() != intermediate_size_val * hidden_size_val {
        return Err(MinervaError::InferenceError(
            "Down weight shape mismatch".to_string(),
        ));
    }

    // Step 1: Up projection: (seq_len, hidden_size) @ (hidden_size, intermediate_size)
    let mut up_output = vec![0.0; seq_len_val * intermediate_size_val];

    for i in 0..seq_len_val {
        for j in 0..intermediate_size_val {
            let mut sum = 0.0;
            for k in 0..hidden_size_val {
                sum += input[i * hidden_size_val + k] * w_up[k * intermediate_size_val + j];
            }
            up_output[i * intermediate_size_val + j] = sum;
        }
    }

    // Step 2: Apply activation
    let activated = apply_activation(&up_output, activation_val);

    // Step 3: Down projection: (seq_len, intermediate_size) @ (intermediate_size, hidden_size)
    let mut output = vec![0.0; seq_len_val * hidden_size_val];

    for i in 0..seq_len_val {
        for j in 0..hidden_size_val {
            let mut sum = 0.0;
            for k in 0..intermediate_size_val {
                sum += activated[i * intermediate_size_val + k] * w_down[k * hidden_size_val + j];
            }
            output[i * hidden_size_val + j] = sum;
        }
    }

    Ok(output)
}

// ============================================================================
// Transformer Block
// ============================================================================

/// Transformer block configuration
#[derive(Debug, Clone, Copy)]
pub struct TransformerBlockConfig {
    pub seq_len: usize,
    pub hidden_size: usize,
    pub num_heads: usize,
    pub intermediate_size: usize,
    pub activation: Activation,
    pub causal: bool,
    pub eps: f32,
}

/// Transformer block weights
#[derive(Debug)]
pub struct TransformerBlockWeights<'a> {
    pub attn_scale: Option<&'a [f32]>,
    pub ff_up: &'a [f32],
    pub ff_down: &'a [f32],
}

/// Complete transformer block
///
/// Implements:
/// 1. LayerNorm → MultiHeadAttention → Residual
/// 2. LayerNorm → Feedforward → Residual
///
/// # Arguments
/// * `input`: Input tensor (seq_len, hidden_size)
/// * `weights`: Block weights (attention scale, FFN up/down)
/// * `config`: Block configuration
///
/// # Returns
/// Block output (seq_len, hidden_size)
pub fn transformer_block(
    input: &[f32],
    weights: &TransformerBlockWeights,
    config: &TransformerBlockConfig,
) -> MinervaResult<Vec<f32>> {
    let attn_scale = weights.attn_scale;
    let ff_up = weights.ff_up;
    let ff_down = weights.ff_down;
    let TransformerBlockConfig {
        seq_len,
        hidden_size,
        num_heads,
        intermediate_size,
        activation,
        causal,
        eps,
    } = config;

    let seq_len_val = *seq_len;
    let hidden_size_val = *hidden_size;
    let num_heads_val = *num_heads;
    let intermediate_size_val = *intermediate_size;
    let activation_val = *activation;
    let causal_val = *causal;
    let eps_val = *eps;

    if input.len() != seq_len_val * hidden_size_val {
        return Err(MinervaError::InferenceError(
            "Input shape mismatch".to_string(),
        ));
    }

    // ---- Attention Block ----
    // Step 1: LayerNorm
    let ln1_config = LNCfg {
        seq_len: seq_len_val,
        hidden_size: hidden_size_val,
        scale: attn_scale.map(|s| s.to_vec()),
        eps: eps_val,
    };
    let normed = layer_norm(input, &ln1_config)?;

    // Step 2: MultiHeadAttention (simplified: input is Q, K, V)
    let mha_config = MultiHeadConfig {
        seq_len: seq_len_val,
        hidden_size: hidden_size_val,
        num_heads: num_heads_val,
        causal: causal_val,
    };
    let attn_out = multi_head_attention(&normed, &mha_config)?;

    // Step 3: Residual connection
    let mut after_attn = input.to_vec();
    for (i, val) in attn_out.iter().enumerate() {
        after_attn[i] += val;
    }

    // ---- Feedforward Block ----
    // Step 4: LayerNorm
    let ln2_config = LNCfg {
        seq_len: seq_len_val,
        hidden_size: hidden_size_val,
        scale: None,
        eps: eps_val,
    };
    let normed2 = layer_norm(&after_attn, &ln2_config)?;

    // Step 5: Feedforward
    let ff_config = FeedforwardConfig {
        seq_len: seq_len_val,
        hidden_size: hidden_size_val,
        intermediate_size: intermediate_size_val,
        activation: activation_val,
    };
    let ff_weights = FeedforwardWeights {
        up: ff_up,
        down: ff_down,
    };
    let ff_out = feedforward(&normed2, &ff_weights, &ff_config)?;

    // Step 6: Residual connection
    let mut output = after_attn.to_vec();
    for (i, val) in ff_out.iter().enumerate() {
        output[i] += val;
    }

    Ok(output)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gelu_activation() {
        let x = 1.0;
        let y = gelu(x);
        // GELU(1.0) ≈ 0.841
        assert!((y - 0.841).abs() < 0.01);
    }

    #[test]
    fn test_silu_activation() {
        let x = 0.0;
        let y = silu(x);
        // SiLU(0) = 0
        assert_eq!(y, 0.0);
    }

    #[test]
    fn test_relu_activation() {
        assert_eq!(relu(-1.0), 0.0);
        assert_eq!(relu(0.0), 0.0);
        assert_eq!(relu(1.0), 1.0);
    }

    #[test]
    fn test_apply_activation_gelu() {
        let input = vec![0.0, 1.0, 2.0];
        let output = apply_activation(&input, Activation::GELU);
        assert_eq!(output.len(), 3);
        assert!(output[0] < 0.1); // GELU(0) ≈ 0
        assert!(output[1] > 0.8); // GELU(1) ≈ 0.84
    }

    #[test]
    fn test_embed_tokens_shapes() {
        let vocab_size = 10;
        let hidden_size = 4;
        let embeddings = vec![0.1; vocab_size * hidden_size];
        let tokens = vec![0, 1, 2];

        let config = EmbeddingConfig {
            vocab_size,
            hidden_size,
        };
        let output = embed_tokens(&tokens, &embeddings, &config).unwrap();

        assert_eq!(output.len(), tokens.len() * hidden_size);
    }

    #[test]
    fn test_embed_tokens_out_of_vocab() {
        let vocab_size = 10;
        let hidden_size = 4;
        let embeddings = vec![0.1; vocab_size * hidden_size];
        let tokens = vec![0, 999]; // 999 is out of vocab

        let config = EmbeddingConfig {
            vocab_size,
            hidden_size,
        };
        let result = embed_tokens(&tokens, &embeddings, &config);
        assert!(result.is_err());
    }

    #[test]
    fn test_position_encoding_shapes() {
        let config = PositionConfig {
            seq_len: 4,
            hidden_size: 8,
            base: 10000.0,
        };
        let pe = create_position_encoding(&config);
        assert_eq!(pe.len(), 4 * 8);
    }

    #[test]
    fn test_position_encoding_symmetry() {
        let config = PositionConfig {
            seq_len: 3,
            hidden_size: 4,
            base: 10000.0,
        };
        let pe = create_position_encoding(&config);

        // Check that even/odd pairs follow sin/cos pattern
        // Position 0, dimension 0 should be sin
        let pos0_dim0 = pe[0];
        let pos0_dim1 = pe[1];

        // Both should be close to their expected values
        // sin(0) = 0, cos(0) = 1
        assert!(pos0_dim0.abs() < 0.01);
        assert!((pos0_dim1 - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_add_position_encoding() {
        let embeddings = vec![0.5; 12]; // 3 tokens × 4 hidden
        let pe = vec![0.1; 12];

        let result = add_position_encoding(&embeddings, &pe).unwrap();
        assert_eq!(result.len(), 12);
        assert!((result[0] - 0.6).abs() < 0.01); // 0.5 + 0.1
    }

    #[test]
    fn test_add_position_encoding_shape_mismatch() {
        let embeddings = vec![0.5; 12];
        let pe = vec![0.1; 8]; // Wrong shape

        let result = add_position_encoding(&embeddings, &pe);
        assert!(result.is_err());
    }

    #[test]
    fn test_feedforward_shapes() {
        let seq_len = 2;
        let hidden_size = 4;
        let intermediate_size = 8;

        let input = vec![0.1; seq_len * hidden_size];
        let w_up = vec![0.2; hidden_size * intermediate_size];
        let w_down = vec![0.1; intermediate_size * hidden_size];

        let config = FeedforwardConfig {
            seq_len,
            hidden_size,
            intermediate_size,
            activation: Activation::GELU,
        };

        let weights = FeedforwardWeights {
            up: &w_up,
            down: &w_down,
        };
        let output = feedforward(&input, &weights, &config).unwrap();
        assert_eq!(output.len(), seq_len * hidden_size);
    }

    #[test]
    fn test_feedforward_all_activations() {
        let seq_len = 1;
        let hidden_size = 2;
        let intermediate_size = 4;

        let input = vec![1.0; seq_len * hidden_size];
        let w_up = vec![0.5; hidden_size * intermediate_size];
        let w_down = vec![0.5; intermediate_size * hidden_size];

        let weights = FeedforwardWeights {
            up: &w_up,
            down: &w_down,
        };

        for activation in [Activation::GELU, Activation::SiLU, Activation::ReLU] {
            let config = FeedforwardConfig {
                seq_len,
                hidden_size,
                intermediate_size,
                activation,
            };
            let result = feedforward(&input, &weights, &config);
            assert!(result.is_ok());
        }
    }

    #[test]
    fn test_transformer_block_shapes() {
        let seq_len = 2;
        let hidden_size = 8;
        let num_heads = 2;
        let intermediate_size = 16;

        let input = vec![0.1; seq_len * hidden_size];
        let attn_scale = vec![1.0; hidden_size];
        let ff_up = vec![0.2; hidden_size * intermediate_size];
        let ff_down = vec![0.1; intermediate_size * hidden_size];

        let config = TransformerBlockConfig {
            seq_len,
            hidden_size,
            num_heads,
            intermediate_size,
            activation: Activation::GELU,
            causal: true,
            eps: 1e-6,
        };

        let weights = TransformerBlockWeights {
            attn_scale: Some(&attn_scale),
            ff_up: &ff_up,
            ff_down: &ff_down,
        };

        let output = transformer_block(&input, &weights, &config).unwrap();
        assert_eq!(output.len(), seq_len * hidden_size);
    }

    #[test]
    fn test_transformer_block_residuals() {
        let seq_len = 1;
        let hidden_size = 4;
        let num_heads = 1;
        let intermediate_size = 8;

        // Small inputs to trace through
        let input = vec![1.0; seq_len * hidden_size];
        let attn_scale = vec![1.0; hidden_size];
        let ff_up = vec![0.1; hidden_size * intermediate_size];
        let ff_down = vec![0.1; intermediate_size * hidden_size];

        let config = TransformerBlockConfig {
            seq_len,
            hidden_size,
            num_heads,
            intermediate_size,
            activation: Activation::ReLU,
            causal: false,
            eps: 1e-6,
        };

        let weights = TransformerBlockWeights {
            attn_scale: Some(&attn_scale),
            ff_up: &ff_up,
            ff_down: &ff_down,
        };

        let output = transformer_block(&input, &weights, &config).unwrap();

        // Output should be close to input (with small residual additions)
        assert_eq!(output.len(), input.len());
        for val in output.iter() {
            assert!(val.is_finite());
        }
    }
}
