/// Scaled Dot-Product Attention for Transformer Networks
///
/// Implements the core attention mechanism: Attention(Q,K,V) = softmax(QK^T/√d)V
/// This is the fundamental building block of transformer networks.
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
