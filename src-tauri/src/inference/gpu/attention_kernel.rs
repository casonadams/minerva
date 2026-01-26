/// High-Performance Attention Kernel
///
/// Optimized implementations for:
/// - GQA (Grouped Query Attention)
/// - Flash Attention approximation
/// - Batched operations
use ndarray::{Array1, Array2, Array3, Axis, s};
use std::f32;

/// GQA Attention - optimized for GPT-OSS
///
/// Input shapes:
///   q: (seq_len, num_query_heads, head_dim)
///   k: (seq_len, num_kv_heads, head_dim)
///   v: (seq_len, num_kv_heads, head_dim)
///   
/// Output:
///   (seq_len, num_query_heads, head_dim)
pub fn gqa_attention(
    q: &Array3<f32>,
    k: &Array3<f32>,
    v: &Array3<f32>,
    head_dim: usize,
) -> Array3<f32> {
    let (seq_len, num_query_heads, _) = q.dim();
    let (_, num_kv_heads, _) = k.dim();

    let mut output = Array3::zeros((seq_len, num_query_heads, head_dim));

    // For each query head
    for q_head in 0..num_query_heads {
        // Map query head to KV head (GQA: 8 query heads → 1 KV head in GPT-OSS)
        let kv_head = q_head / (num_query_heads / num_kv_heads);

        // Get slices for this head
        let q_head_seq = q.slice(s![.., q_head, ..]); // (seq_len, head_dim)
        let k_head_seq = k.slice(s![.., kv_head, ..]); // (seq_len, head_dim)
        let v_head_seq = v.slice(s![.., kv_head, ..]); // (seq_len, head_dim)

        // Compute attention scores: Q @ K^T
        // (seq_len, head_dim) @ (head_dim, seq_len) = (seq_len, seq_len)
        let scores = q_head_seq.dot(&k_head_seq.t()) * (1.0 / (head_dim as f32).sqrt());

        // Apply softmax
        let attn_weights = softmax_2d(&scores);

        // Apply attention: softmax(QK^T) @ V
        // (seq_len, seq_len) @ (seq_len, head_dim) = (seq_len, head_dim)
        let head_output = attn_weights.dot(&v_head_seq);

        output.slice_mut(s![.., q_head, ..]).assign(&head_output);
    }

    output
}

/// Flash Attention approximation - compute attention in blocks
///
/// Reduces memory bandwidth by computing attention in cache-friendly blocks
/// instead of all-at-once
pub fn flash_attention_approx(
    q: &Array3<f32>,
    k: &Array3<f32>,
    v: &Array3<f32>,
    head_dim: usize,
    block_size: usize,
) -> Array3<f32> {
    let (seq_len, num_heads, _) = q.dim();
    let mut output = Array3::zeros((seq_len, num_heads, head_dim));

    // Process in blocks to fit in cache
    for block_start in (0..seq_len).step_by(block_size) {
        let block_end = (block_start + block_size).min(seq_len);
        let _current_block_size = block_end - block_start;

        // Get block slices - convert to owned arrays for function call
        let q_block = q.slice(s![block_start..block_end, .., ..]).to_owned();
        let k_block = k.slice(s![..block_end, .., ..]).to_owned(); // Include all prev tokens
        let v_block = v.slice(s![..block_end, .., ..]).to_owned();

        // Compute attention for this block
        let block_output = gqa_attention(&q_block, &k_block, &v_block, head_dim);

        // Store block output
        output
            .slice_mut(s![block_start..block_end, .., ..])
            .assign(&block_output);
    }

    output
}

/// Softmax for 2D matrix (row-wise)
/// Input: (seq_len, seq_len)
/// Output: (seq_len, seq_len) - each row sums to 1
#[inline]
fn softmax_2d(input: &Array2<f32>) -> Array2<f32> {
    let mut output = input.clone();

    for mut row in output.rows_mut() {
        // Find max for numerical stability
        let max_val = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        // Apply exp(x - max)
        for elem in row.iter_mut() {
            *elem = (*elem - max_val).exp();
        }

        // Normalize (sum to 1)
        let sum: f32 = row.sum();
        if sum > 0.0 {
            for elem in row.iter_mut() {
                *elem /= sum;
            }
        }
    }

    output
}

/// Softmax for 1D vector
#[inline]
pub fn softmax_1d(input: &Array1<f32>) -> Array1<f32> {
    let mut output = input.clone();

    let max_val = input.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    for elem in output.iter_mut() {
        *elem = (*elem - max_val).exp();
    }

    let sum: f32 = output.sum();
    if sum > 0.0 {
        output /= sum;
    }

    output
}

/// Causal mask - prevent attention to future tokens
/// Returns mask matrix: 0 = attend, -inf = don't attend
pub fn causal_mask(seq_len: usize) -> Array2<f32> {
    let mut mask = Array2::zeros((seq_len, seq_len));

    for i in 0..seq_len {
        for j in (i + 1)..seq_len {
            mask[[i, j]] = f32::NEG_INFINITY;
        }
    }

    mask
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_softmax_1d() {
        let input = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let output = softmax_1d(&input);

        // Should sum to 1
        assert!((output.sum() - 1.0).abs() < 1e-5);

        // Each element should be between 0 and 1
        for elem in output.iter() {
            assert!(*elem >= 0.0 && *elem <= 1.0);
        }
    }

    #[test]
    fn test_causal_mask() {
        let mask = causal_mask(3);

        // Diagonal and lower should be 0 (attend)
        assert_eq!(mask[[0, 0]], 0.0);
        assert_eq!(mask[[1, 0]], 0.0);
        assert_eq!(mask[[2, 0]], 0.0);

        // Upper should be -inf (don't attend)
        assert!(mask[[0, 1]].is_infinite());
        assert!(mask[[0, 2]].is_infinite());
    }
}
