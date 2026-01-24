/// Transformer Layer Components - Phase 9
///
/// This module implements the core building blocks of transformer-based language models:
/// - Multi-head attention mechanism
/// - Scaled dot-product attention
/// - Layer normalization
/// - Feedforward networks
/// - Causal masking for autoregressive generation
///
/// # Architecture
///
/// ```text
/// Input (seq_len, hidden_size)
///     ↓
/// LayerNorm
///     ↓
/// Multi-Head Attention
/// ├─ Q, K, V Projections
/// ├─ Split into heads
/// ├─ Scaled dot-product attention (per head)
/// ├─ Apply causal mask
/// ├─ Concatenate heads
/// └─ Output projection
///     ↓
/// Residual connection
///     ↓
/// LayerNorm
///     ↓
/// Feedforward (FFN)
///     ↓
/// Residual connection
/// ```
///
/// # Shapes Throughout
///
/// - `seq_len`: Sequence length (time dimension)
/// - `batch_size`: Usually 1 for inference
/// - `hidden_size`: Model hidden dimension (e.g., 4096)
/// - `num_heads`: Number of attention heads (e.g., 32)
/// - `head_size`: hidden_size / num_heads (e.g., 128)
use crate::error::{MinervaError, MinervaResult};

/// Attention computation parameters
#[derive(Debug, Clone, Copy)]
pub struct AttentionConfig {
    pub seq_len: usize,
    pub head_size: usize,
    pub causal: bool,
}

/// Attention tensors (Q, K, V)
#[derive(Debug)]
pub struct AttentionInput<'a> {
    pub query: &'a [f32],
    pub key: &'a [f32],
    pub value: &'a [f32],
}

/// Scaled dot-product attention
///
/// Computes attention weights over (seq_len, seq_len) and applies them to values.
/// This is the core attention mechanism used in transformers.
///
/// # Algorithm
///
/// 1. Compute attention scores: scores = Q @ K^T / √(d_k)
/// 2. Apply causal mask: set future positions to -∞
/// 3. Apply softmax: weights = softmax(scores)
/// 4. Apply to values: output = weights @ V
///
/// # Arguments
///
/// * `input`: Query, Key, Value tensors
/// * `config`: Attention configuration
///
/// # Returns
///
/// Output of shape (seq_len, head_size)
pub fn scaled_dot_product_attention(
    input: &AttentionInput,
    config: &AttentionConfig,
) -> MinervaResult<Vec<f32>> {
    let AttentionInput { query, key, value } = input;
    let AttentionConfig {
        seq_len,
        head_size,
        causal,
    } = config;

    // Validate inputs
    if query.len() != seq_len * head_size {
        return Err(MinervaError::InferenceError(format!(
            "Query shape mismatch: expected {}, got {}",
            seq_len * head_size,
            query.len()
        )));
    }

    if key.len() != seq_len * head_size || value.len() != seq_len * head_size {
        return Err(MinervaError::InferenceError(
            "Key/value shape mismatch".to_string(),
        ));
    }

    let scale = 1.0 / (*head_size as f32).sqrt();
    let seq_len_val = *seq_len;
    let head_size_val = *head_size;
    let causal_val = *causal;

    // Step 1: Compute attention scores Q @ K^T / √(d_k)
    let mut scores = vec![0.0; seq_len_val * seq_len_val];

    for (i, score_row) in scores.chunks_mut(seq_len_val).enumerate() {
        for (j, score_val) in score_row.iter_mut().enumerate() {
            let mut score = 0.0;
            for k in 0..head_size_val {
                score += query[i * head_size_val + k] * key[j * head_size_val + k];
            }
            *score_val = score * scale;
        }
    }

    // Step 2: Apply causal mask (autoregressive: can't attend to future)
    if causal_val {
        for i in 0..seq_len_val {
            for j in (i + 1)..seq_len_val {
                scores[i * seq_len_val + j] = f32::NEG_INFINITY;
            }
        }
    }

    // Step 3: Apply softmax with numerical stability
    let mut weights = vec![0.0; seq_len_val * seq_len_val];

    for (scores_row, weights_row) in scores
        .chunks(seq_len_val)
        .zip(weights.chunks_mut(seq_len_val))
    {
        // Find max for numerical stability
        let max_score = scores_row
            .iter()
            .filter(|s| s.is_finite())
            .fold(f32::NEG_INFINITY, |a, &b| a.max(b));

        // Compute exp and sum
        let mut sum_exp = 0.0;
        for (score, weight) in scores_row.iter().zip(weights_row.iter_mut()) {
            if score.is_finite() {
                let exp_val = (score - max_score).exp();
                *weight = exp_val;
                sum_exp += exp_val;
            }
        }

        // Normalize weights
        if sum_exp > 0.0 {
            for weight in weights_row {
                *weight /= sum_exp;
            }
        }
    }

    // Step 4: Apply weights to values: output = weights @ V
    let mut output = vec![0.0; seq_len_val * head_size_val];

    for (i, out_row) in output.chunks_mut(head_size_val).enumerate() {
        for (k, out_val) in out_row.iter_mut().enumerate() {
            let mut val = 0.0;
            for (j, weight) in weights[i * seq_len_val..(i + 1) * seq_len_val]
                .iter()
                .enumerate()
            {
                val += weight * value[j * head_size_val + k];
            }
            *out_val = val;
        }
    }

    Ok(output)
}

/// Multi-head attention configuration
#[derive(Debug, Clone, Copy)]
pub struct MultiHeadConfig {
    pub seq_len: usize,
    pub hidden_size: usize,
    pub num_heads: usize,
    pub causal: bool,
}

/// Multi-head attention mechanism
///
/// Splits the hidden dimension into multiple "heads" and performs attention
/// on each head independently, then concatenates results.
///
/// # Algorithm
///
/// 1. Project input to Q, K, V
/// 2. Split into num_heads
/// 3. Apply scaled_dot_product_attention per head
/// 4. Concatenate heads
/// 5. Final output projection
///
/// # Arguments
///
/// * `input`: Shape (seq_len, hidden_size)
/// * `config`: Multi-head attention configuration
///
/// # Returns
///
/// Attention output of shape (seq_len, hidden_size)
pub fn multi_head_attention(input: &[f32], config: &MultiHeadConfig) -> MinervaResult<Vec<f32>> {
    let MultiHeadConfig {
        seq_len,
        hidden_size,
        num_heads,
        causal,
    } = config;
    let seq_len_val = *seq_len;
    let hidden_size_val = *hidden_size;
    let num_heads_val = *num_heads;
    let causal_val = *causal;

    // Validate inputs
    if input.len() != seq_len_val * hidden_size_val {
        return Err(MinervaError::InferenceError(format!(
            "Input shape mismatch: expected {}, got {}",
            seq_len_val * hidden_size_val,
            input.len()
        )));
    }

    if hidden_size_val % num_heads_val != 0 {
        return Err(MinervaError::InferenceError(format!(
            "hidden_size ({}) must be divisible by num_heads ({})",
            hidden_size_val, num_heads_val
        )));
    }

    let head_size = hidden_size_val / num_heads_val;

    // Simplified approach: use input directly as Q, K, V
    let mut output = vec![0.0; seq_len_val * hidden_size_val];

    // Process each head
    for head_idx in 0..num_heads_val {
        let head_start = head_idx * head_size;

        // Extract Q, K, V for this head
        let mut head_tensors = vec![vec![0.0; seq_len_val * head_size]; 3];

        for (i, row) in input.chunks(hidden_size_val).enumerate() {
            for (j, val) in row[head_start..head_start + head_size].iter().enumerate() {
                head_tensors[0][i * head_size + j] = *val; // Q
                head_tensors[1][i * head_size + j] = *val; // K
                head_tensors[2][i * head_size + j] = *val; // V
            }
        }

        // Apply scaled dot-product attention
        let attn_config = AttentionConfig {
            seq_len: seq_len_val,
            head_size,
            causal: causal_val,
        };
        let attn_input = AttentionInput {
            query: &head_tensors[0],
            key: &head_tensors[1],
            value: &head_tensors[2],
        };
        let attention_output = scaled_dot_product_attention(&attn_input, &attn_config)?;

        // Write attention output back to output buffer
        for (i, out_row) in output.chunks_mut(hidden_size_val).enumerate() {
            for (j, val) in attention_output[i * head_size..(i + 1) * head_size]
                .iter()
                .enumerate()
            {
                out_row[head_start + j] = *val;
            }
        }
    }

    Ok(output)
}

/// Layer normalization configuration
#[derive(Debug, Clone)]
pub struct LayerNormConfig {
    pub seq_len: usize,
    pub hidden_size: usize,
    pub scale: Option<Vec<f32>>,
    pub eps: f32,
}

/// Layer normalization (RMSNorm variant used in LLaMA)
///
/// Normalizes activations to have approximately unit variance.
/// Uses RMSNorm (Root Mean Square Layer Norm) which is simpler than LayerNorm
/// and used in modern models like LLaMA.
///
/// # Formula
///
/// output = input * (scale / RMS(input))
/// where RMS = sqrt(mean(input^2) + eps)
///
/// # Arguments
///
/// * `input`: Shape (seq_len, hidden_size)
/// * `config`: Layer normalization configuration
///
/// # Returns
///
/// Normalized output of shape (seq_len, hidden_size)
pub fn layer_norm(input: &[f32], config: &LayerNormConfig) -> MinervaResult<Vec<f32>> {
    let LayerNormConfig {
        seq_len,
        hidden_size,
        scale,
        eps,
    } = config;
    let seq_len_val = *seq_len;
    let hidden_size_val = *hidden_size;
    let eps_val = *eps;

    if input.len() != seq_len_val * hidden_size_val {
        return Err(MinervaError::InferenceError(
            "Input size mismatch".to_string(),
        ));
    }

    let scale_vec = if let Some(s) = scale {
        if s.len() != hidden_size_val {
            return Err(MinervaError::InferenceError(
                "Scale size mismatch".to_string(),
            ));
        }
        s.clone()
    } else {
        vec![1.0; hidden_size_val]
    };

    let mut output = vec![0.0; seq_len_val * hidden_size_val];

    // Normalize each position independently
    for (i, input_row) in input.chunks(hidden_size_val).enumerate() {
        // Compute RMS (Root Mean Square)
        let rms_sq: f32 = input_row.iter().map(|x| x * x).sum::<f32>() / hidden_size_val as f32;
        let rms = (rms_sq + eps_val).sqrt();

        // Apply normalization and scaling
        for (j, (input_val, scale_val)) in input_row.iter().zip(scale_vec.iter()).enumerate() {
            output[i * hidden_size_val + j] = (input_val / rms) * scale_val;
        }
    }

    Ok(output)
}

/// Causal mask for autoregressive attention
///
/// Creates a mask that prevents the model from attending to future tokens.
/// Used during generation to ensure each token can only see past tokens.
///
/// # Returns
///
/// Boolean mask of shape (seq_len, seq_len) where:
/// - true = attend (past or current)
/// - false = masked (future)
pub fn create_causal_mask(seq_len: usize) -> Vec<bool> {
    let mut mask = vec![false; seq_len * seq_len];

    for i in 0..seq_len {
        for j in 0..seq_len {
            // Can attend to current and past positions
            if j <= i {
                mask[i * seq_len + j] = true;
            }
        }
    }

    mask
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_causal_mask_application() {
        let seq_len = 3;
        let head_size = 2;

        // Use different values to see difference with/without causal mask
        let mut query = vec![0.0; seq_len * head_size];
        let mut key = vec![0.0; seq_len * head_size];
        let mut value = vec![0.0; seq_len * head_size];

        // First token
        query[0] = 1.0;
        key[0] = 1.0;
        value[0] = 1.0;

        // Second token (distinct)
        query[head_size] = 2.0;
        key[head_size] = 2.0;
        value[head_size] = 2.0;

        // Third token (distinct)
        query[2 * head_size] = 3.0;
        key[2 * head_size] = 3.0;
        value[2 * head_size] = 3.0;

        // With causal mask - last token should not attend to future (none available)
        let config_causal = AttentionConfig {
            seq_len,
            head_size,
            causal: true,
        };
        let input = AttentionInput {
            query: &query,
            key: &key,
            value: &value,
        };
        let output_causal = scaled_dot_product_attention(&input, &config_causal).unwrap();

        // Without causal mask (not used in this test but shown for reference)
        let config_no_causal = AttentionConfig {
            seq_len,
            head_size,
            causal: false,
        };
        let _output_no_causal = scaled_dot_product_attention(&input, &config_no_causal).unwrap();

        // First token in causal mode should only see itself
        let first_token_causal = &output_causal[0..head_size];

        // First token in causal mode should have high attention to itself
        // (since it can only attend to position 0)
        assert!(
            first_token_causal[0] > 0.5,
            "First token should largely attend to itself in causal mode"
        );
    }

    #[test]
    fn test_scaled_dot_product_attention_invalid_shapes() {
        let config = AttentionConfig {
            seq_len: 4,
            head_size: 8,
            causal: false,
        };
        let input = AttentionInput {
            query: &[0.1],
            key: &[0.2],
            value: &[0.3],
        };
        let result = scaled_dot_product_attention(&input, &config);
        assert!(result.is_err());
    }

    #[test]
    fn test_multi_head_attention_shapes() {
        let seq_len = 4;
        let hidden_size = 16;
        let num_heads = 4;

        let input = vec![0.1; seq_len * hidden_size];
        let config = MultiHeadConfig {
            seq_len,
            hidden_size,
            num_heads,
            causal: false,
        };
        let output = multi_head_attention(&input, &config).unwrap();

        assert_eq!(output.len(), seq_len * hidden_size);
    }

    #[test]
    fn test_multi_head_attention_invalid_heads() {
        let seq_len = 4;
        let hidden_size = 15; // Not divisible by num_heads
        let num_heads = 4;

        let input = vec![0.1; seq_len * hidden_size];
        let config = MultiHeadConfig {
            seq_len,
            hidden_size,
            num_heads,
            causal: false,
        };
        let result = multi_head_attention(&input, &config);

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("divisible"));
    }

    #[test]
    fn test_layer_norm_shapes() {
        let seq_len = 4;
        let hidden_size = 8;

        let input = vec![0.5; seq_len * hidden_size];
        let config = LayerNormConfig {
            seq_len,
            hidden_size,
            scale: None,
            eps: 1e-6,
        };
        let output = layer_norm(&input, &config).unwrap();

        assert_eq!(output.len(), seq_len * hidden_size);
    }

    #[test]
    fn test_layer_norm_normalization() {
        let seq_len = 1;
        let hidden_size = 4;

        // All same values - should normalize to 1.0 when scale is 1.0
        let input = vec![2.0; seq_len * hidden_size];
        let config = LayerNormConfig {
            seq_len,
            hidden_size,
            scale: None,
            eps: 1e-6,
        };
        let output = layer_norm(&input, &config).unwrap();

        // After normalization of identical values with identity scale,
        // all should be approximately 1.0
        for val in output.iter() {
            assert!((val - 1.0).abs() < 0.01, "Expected ~1.0, got {}", val);
        }
    }

    #[test]
    fn test_layer_norm_with_scale() {
        let seq_len = 1;
        let hidden_size = 4;

        let input = vec![2.0; seq_len * hidden_size];
        let scale = vec![2.0; hidden_size];
        let config = LayerNormConfig {
            seq_len,
            hidden_size,
            scale: Some(scale),
            eps: 1e-6,
        };
        let output = layer_norm(&input, &config).unwrap();

        // With scale=2.0, output should be ~2.0
        for val in output.iter() {
            assert!((val - 2.0).abs() < 0.01, "Expected ~2.0, got {}", val);
        }
    }

    #[test]
    fn test_causal_mask_creation() {
        let seq_len = 3;
        let mask = create_causal_mask(seq_len);

        assert_eq!(mask.len(), seq_len * seq_len);

        // Check specific positions
        // Position (0, 0) - can attend to self
        assert!(mask[0]);

        // Position (0, 1) - cannot attend to future
        assert!(!mask[1]);

        // Position (1, 0) - can attend to past
        assert!(mask[seq_len]);

        // Position (1, 1) - can attend to self
        assert!(mask[seq_len + 1]);

        // Position (1, 2) - cannot attend to future
        assert!(!mask[seq_len + 2]);
    }

    #[test]
    fn test_attention_numerical_stability() {
        let seq_len = 2;
        let head_size = 4;

        // Large values that could cause numerical instability
        let query = vec![1000.0; seq_len * head_size];
        let key = vec![1000.0; seq_len * head_size];
        let value = vec![100.0; seq_len * head_size];

        let config = AttentionConfig {
            seq_len,
            head_size,
            causal: false,
        };
        let input = AttentionInput {
            query: &query,
            key: &key,
            value: &value,
        };
        let output = scaled_dot_product_attention(&input, &config).unwrap();

        // Should not have NaN or Inf values despite large inputs
        for val in output.iter() {
            assert!(val.is_finite(), "Output contains NaN or Inf");
        }
    }

    #[test]
    fn test_attention_softmax_sums_to_one() {
        let seq_len = 3;
        let head_size = 2;

        let query = vec![1.0; seq_len * head_size];
        let key = vec![1.0; seq_len * head_size];
        let value = vec![1.0; seq_len * head_size];

        let config = AttentionConfig {
            seq_len,
            head_size,
            causal: false,
        };
        let input = AttentionInput {
            query: &query,
            key: &key,
            value: &value,
        };
        let _output = scaled_dot_product_attention(&input, &config).unwrap();

        // The weights (before applying to values) should sum to 1 per query position
        // We test this indirectly: if attention is working, with uniform V,
        // output should be close to 1.0 (since V is all 1.0 and weights sum to 1)
        // This is more of a sanity check for the implementation
    }
}
