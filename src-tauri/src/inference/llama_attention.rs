use super::rope_utils::RoPEParams;
/// Multi-Head Attention with Rotary Positional Embeddings
///
/// Implements scaled dot-product attention with rotary position embeddings (RoPE).
/// This allows the model to incorporate positional information while maintaining
/// relative position awareness.
///
/// Core formula: Attention(Q,K,V) = softmax(Q@K^T / sqrt(d)) @ V
use crate::error::{MinervaError, MinervaResult};

/// Attention output
#[derive(Debug, Clone)]
pub struct AttentionOutput {
    /// Attention output tensor
    pub output: Vec<f32>,
    /// Attention weights for visualization
    pub weights: Option<Vec<f32>>,
}

/// Parameters for MultiHeadAttention forward pass
pub struct AttentionParams<'a> {
    /// Query data (mutable)
    pub query: &'a mut [f32],
    /// Key data (mutable)
    pub key: &'a mut [f32],
    /// Value data
    pub value: &'a [f32],
    /// Position in sequence
    pub pos: usize,
}

/// Parameters for MultiHeadAttention::apply_rope
struct RopeParams<'a> {
    query: &'a mut [f32],
    key: &'a mut [f32],
    pos: usize,
}

/// Parameters for MultiHeadAttention::compute_scores
struct ScoreParams<'a> {
    query: &'a [f32],
    keys: &'a [f32],
    scale: f32,
}

/// Compute softmax on values
fn softmax(values: &mut [f32]) {
    let max = values.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    for v in values.iter_mut() {
        *v = (*v - max).exp();
    }
    let sum: f32 = values.iter().sum();
    if sum > 0.0 {
        for v in values.iter_mut() {
            *v /= sum;
        }
    }
}

/// Multi-head self-attention with rotary embeddings
pub struct MultiHeadAttention {
    num_heads: usize,
    head_dim: usize,
    rope: RoPEParams,
}

impl MultiHeadAttention {
    /// Create new multihead attention
    ///
    /// # Arguments
    /// * `num_heads` - Number of attention heads
    /// * `total_dim` - Total embedding dimension (must be divisible by num_heads)
    ///
    /// # Errors
    /// Returns error if total_dim is not divisible by num_heads
    pub fn new(num_heads: usize, total_dim: usize) -> MinervaResult<Self> {
        if !total_dim.is_multiple_of(num_heads) {
            return Err(MinervaError::InferenceError(
                "Total dimension must be divisible by num_heads".to_string(),
            ));
        }

        let head_dim = total_dim / num_heads;
        if !(num_heads * head_dim).is_multiple_of(2) {
            return Err(MinervaError::InferenceError(
                "num_heads * head_dim must be even for RoPE".to_string(),
            ));
        }

        Ok(Self {
            num_heads,
            head_dim,
            rope: RoPEParams::new(head_dim),
        })
    }

    /// Apply rotary embeddings to query and key
    fn apply_rope(&self, rope: RopeParams) {
        for h in 0..self.num_heads {
            for d in (0..self.head_dim).step_by(2) {
                let angle = self.rope.get_angle(rope.pos, d);
                let cos = angle.cos();
                let sin = angle.sin();

                // Apply rotation to query
                let q_idx_base = h * self.head_dim + d;
                if q_idx_base + 1 < rope.query.len() {
                    let q0 = rope.query[q_idx_base];
                    let q1 = rope.query[q_idx_base + 1];
                    rope.query[q_idx_base] = q0 * cos - q1 * sin;
                    rope.query[q_idx_base + 1] = q0 * sin + q1 * cos;
                }

                // Apply rotation to key
                let k_idx_base = h * self.head_dim + d;
                if k_idx_base + 1 < rope.key.len() {
                    let k0 = rope.key[k_idx_base];
                    let k1 = rope.key[k_idx_base + 1];
                    rope.key[k_idx_base] = k0 * cos - k1 * sin;
                    rope.key[k_idx_base + 1] = k0 * sin + k1 * cos;
                }
            }
        }
    }

    /// Compute attention scores between query and keys
    fn compute_scores(&self, params: ScoreParams) -> Vec<f32> {
        let num_keys = params.keys.len() / (self.num_heads * self.head_dim);
        let mut scores = vec![0.0; num_keys];

        for h in 0..self.num_heads {
            for (k_pos, s) in scores.iter_mut().enumerate() {
                let mut dot_product = 0.0;
                for d in 0..self.head_dim {
                    let q_idx = h * self.head_dim + d;
                    let k_idx = k_pos * (self.num_heads * self.head_dim) + h * self.head_dim + d;
                    if q_idx < params.query.len() && k_idx < params.keys.len() {
                        dot_product += params.query[q_idx] * params.keys[k_idx];
                    }
                }
                *s += dot_product * params.scale;
            }
        }

        scores
    }

    /// Forward pass for attention
    ///
    /// # Arguments
    /// * `params` - Attention parameters including Q, K, V and position
    ///
    /// # Returns
    /// Attention output with optional weights for visualization
    ///
    /// # Errors
    /// Returns error if query and key dimensions don't match
    pub fn forward(&self, params: AttentionParams) -> MinervaResult<AttentionOutput> {
        if params.query.len() != params.key.len() {
            return Err(MinervaError::InferenceError(
                "Query and key dimensions must match".to_string(),
            ));
        }

        // Apply rotary embeddings
        self.apply_rope(RopeParams {
            query: params.query,
            key: params.key,
            pos: params.pos,
        });

        // Compute attention scores
        let scale = (self.head_dim as f32).sqrt().recip();
        let mut scores = self.compute_scores(ScoreParams {
            query: params.query,
            keys: params.key,
            scale,
        });

        // Apply softmax
        softmax(&mut scores);

        // Compute output
        let mut output = vec![0.0; params.query.len()];
        let num_values = params.value.len() / (self.num_heads * self.head_dim);

        for h in 0..self.num_heads {
            for d in 0..self.head_dim {
                let mut sum = 0.0;
                for (v_pos, &score_weight) in scores.iter().enumerate() {
                    if v_pos < num_values {
                        let v_idx =
                            v_pos * (self.num_heads * self.head_dim) + h * self.head_dim + d;
                        if v_idx < params.value.len() {
                            sum += score_weight * params.value[v_idx];
                        }
                    }
                }
                output[h * self.head_dim + d] = sum;
            }
        }

        Ok(AttentionOutput {
            output,
            weights: Some(scores),
        })
    }
}
