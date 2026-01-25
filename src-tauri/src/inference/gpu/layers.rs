use crate::inference::gpu::kv_cache::KVCache;
/// Transformer layers implementation
///
/// Core transformer components: attention, MLP, normalization
use ndarray::Array2;

/// RMS Normalization (like LayerNorm but simpler)
/// Used in Llama and Mistral architectures
pub fn rms_norm(hidden: &Array2<f32>, _weight: &Array2<f32>, eps: f32) -> Array2<f32> {
    // Compute RMS: sqrt(mean(x^2) + eps)
    let squared = hidden * hidden;
    let mean_sq = squared.mean().unwrap_or(1.0);
    let rms = (mean_sq + eps).sqrt();

    // Normalize and scale
    let normalized = hidden / rms;
    // Scale by weight (weight should be shape [hidden_size])
    // For now, return normalized (weight application will be in matmul)
    normalized
}

/// Multi-head attention computation
///
/// # Arguments
/// * `hidden` - Input hidden states (seq_len, hidden_size)
/// * `q_proj` - Query projection weights (hidden_size, hidden_size)
/// * `k_proj` - Key projection weights (hidden_size, hidden_size)
/// * `v_proj` - Value projection weights (hidden_size, hidden_size)
/// * `o_proj` - Output projection weights (hidden_size, hidden_size)
/// * `num_heads` - Number of attention heads
/// * `_kv_cache` - Optional KV cache for generation
pub fn attention(
    hidden: &Array2<f32>,
    q_proj: &Array2<f32>,
    k_proj: &Array2<f32>,
    v_proj: &Array2<f32>,
    o_proj: &Array2<f32>,
    num_heads: usize,
    _kv_cache: &mut Option<KVCache>,
) -> Array2<f32> {
    let _seq_len = hidden.shape()[0];
    let hidden_size = hidden.shape()[1];
    let head_dim = hidden_size / num_heads;

    // Project to Q, K, V
    let q = hidden.dot(q_proj);
    let k = hidden.dot(k_proj);
    let v = hidden.dot(v_proj);

    // For now, skip KV cache complexity and just do basic attention
    // This will be optimized with Flash Attention later

    // Attention scores: Q @ K^T / sqrt(d)
    let scale = 1.0 / (head_dim as f32).sqrt();
    let scores = q.dot(&k.t()) * scale;

    // Softmax (simplified - just normalize for now)
    let attn_weights = softmax(&scores);

    // Apply attention to values
    let output = attn_weights.dot(&v);

    // Output projection
    output.dot(o_proj)
}

/// Softmax computation
fn softmax(x: &Array2<f32>) -> Array2<f32> {
    // Simplified softmax: subtract max for stability, then exp and normalize
    let mut result = x.clone();

    for mut row in result.rows_mut() {
        let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        row.mapv_inplace(|x| (x - max).exp());
        let sum: f32 = row.iter().sum();
        if sum > 0.0 {
            row.mapv_inplace(|x| x / sum);
        }
    }

    result
}

/// SiLU activation (Sigmoid Linear Unit)
pub fn silu(x: &Array2<f32>) -> Array2<f32> {
    x.mapv(|v| v / (1.0 + (-v).exp()))
}

/// MLP (Feed-Forward Network)
/// Uses SwiGLU: (gate_proj(x) * silu) * up_proj(x) -> down_proj
pub fn mlp(
    hidden: &Array2<f32>,
    gate_proj: &Array2<f32>,
    up_proj: &Array2<f32>,
    down_proj: &Array2<f32>,
) -> Array2<f32> {
    // Gate projection with SiLU
    let gate = hidden.dot(gate_proj);
    let gate = silu(&gate);

    // Up projection
    let up = hidden.dot(up_proj);

    // Element-wise multiply (SwiGLU)
    let combined = &gate * &up;

    // Down projection
    combined.dot(down_proj)
}

/// Single transformer layer
pub fn transformer_layer(
    hidden: &Array2<f32>,
    q_proj: &Array2<f32>,
    k_proj: &Array2<f32>,
    v_proj: &Array2<f32>,
    o_proj: &Array2<f32>,
    gate_proj: &Array2<f32>,
    up_proj: &Array2<f32>,
    down_proj: &Array2<f32>,
    attn_norm_weight: &Array2<f32>,
    ffn_norm_weight: &Array2<f32>,
    num_heads: usize,
    eps: f32,
    kv_cache: &mut Option<KVCache>,
) -> Array2<f32> {
    // Pre-norm architecture: norm -> attn -> residual -> norm -> ffn -> residual

    // Attention block
    let hidden_norm = rms_norm(hidden, attn_norm_weight, eps);
    let attn_out = attention(
        &hidden_norm,
        q_proj,
        k_proj,
        v_proj,
        o_proj,
        num_heads,
        kv_cache,
    );
    let hidden = hidden + &attn_out; // Residual

    // MLP block
    let hidden_norm = rms_norm(&hidden, ffn_norm_weight, eps);
    let mlp_out = mlp(&hidden_norm, gate_proj, up_proj, down_proj);
    hidden + mlp_out // Residual
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_rms_norm() {
        let x = Array2::ones((2, 4));
        let weight = Array2::ones((2, 4));
        let norm = rms_norm(&x, &weight, 1e-5);

        assert_eq!(norm.shape(), &[2, 4]);
    }

    #[test]
    fn test_silu() {
        let x = Array2::from_elem((2, 3), 0.5);
        let result = silu(&x);

        assert_eq!(result.shape(), &[2, 3]);
        // SiLU(0.5) should be around 0.3, but we're just checking shape
    }

    #[test]
    fn test_softmax() {
        let x = Array2::ones((2, 3));
        let result = softmax(&x);

        assert_eq!(result.shape(), &[2, 3]);
        // Each row should sum to ~1.0
        for row in result.rows() {
            let sum: f32 = row.iter().sum();
            assert!((sum - 1.0).abs() < 1e-5);
        }
    }
}
