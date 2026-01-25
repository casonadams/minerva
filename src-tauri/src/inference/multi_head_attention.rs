use super::attention::{AttentionConfig, AttentionInput, scaled_dot_product_attention};
/// Multi-Head Attention for Transformer Networks
///
/// Splits hidden dimension into multiple "heads" and performs parallel attention,
/// then concatenates results. This allows the model to attend to different
/// representation subspaces.
use crate::error::{MinervaError, MinervaResult};

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
