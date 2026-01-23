/// LLaMA Inference Engine - Phase 6 Step 3
///
/// This module implements the core LLaMA inference algorithm with:
/// - Multi-head self-attention with rotary positional embeddings
/// - Feed-forward networks with SiLU activation
/// - Token generation with sampling and temperature control
/// - KV cache for efficient inference
use crate::error::{MinervaError, MinervaResult};

/// Rotary positional embeddings parameters
#[derive(Debug, Clone, Copy)]
struct RoPEParams {
    /// Head dimension
    head_dim: usize,
    /// Theta base for rotation
    theta_base: f32,
}

impl RoPEParams {
    /// Create new RoPE parameters
    fn new(head_dim: usize) -> Self {
        Self {
            head_dim,
            theta_base: 10_000.0,
        }
    }

    /// Calculate rotary angle for position and dimension
    fn get_angle(&self, pos: usize, dim: usize) -> f32 {
        let freq = self
            .theta_base
            .powf(-2.0 * (dim as f32) / (self.head_dim as f32));
        (pos as f32) * freq
    }
}

/// Parameters for KV cache initialization
#[derive(Debug, Clone, Copy)]
pub struct KVCacheConfig {
    /// Number of layers
    pub num_layers: usize,
    /// Maximum sequence length
    pub max_seq_len: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// Head dimension
    pub head_dim: usize,
}

/// KV Cache for efficient inference
#[derive(Debug, Clone)]
pub struct KVCache {
    /// Key cache: [layer][seq_len][num_heads][head_dim]
    keys: Vec<Vec<Vec<Vec<f32>>>>,
    /// Value cache: [layer][seq_len][num_heads][head_dim]
    values: Vec<Vec<Vec<Vec<f32>>>>,
}

impl KVCache {
    /// Create new KV cache
    pub fn new(config: KVCacheConfig) -> Self {
        Self {
            keys: vec![
                vec![vec![vec![0.0; config.head_dim]; config.num_heads]; config.max_seq_len];
                config.num_layers
            ],
            values: vec![
                vec![
                    vec![vec![0.0; config.head_dim]; config.num_heads];
                    config.max_seq_len
                ];
                config.num_layers
            ],
        }
    }

    /// Store key and value for a position
    pub fn store(&mut self, layer: usize, pos: usize, kv: (&[f32], &[f32])) -> MinervaResult<()> {
        let (k, v) = kv;
        if layer >= self.keys.len() {
            return Err(MinervaError::InferenceError(format!(
                "Layer index {} out of bounds",
                layer
            )));
        }
        if pos >= self.keys[layer].len() {
            return Err(MinervaError::InferenceError(format!(
                "Position {} out of bounds",
                pos
            )));
        }

        // Flatten head dimension
        let num_heads = self.keys[layer][pos].len();
        let head_dim = self.keys[layer][pos][0].len();

        for h in 0..num_heads {
            let start = h * head_dim;
            let end = start + head_dim;
            if start < k.len() && end <= k.len() {
                self.keys[layer][pos][h].copy_from_slice(&k[start..end]);
                self.values[layer][pos][h].copy_from_slice(&v[start..end]);
            }
        }

        Ok(())
    }

    /// Get key and value for a position
    pub fn get(&self, layer: usize, pos: usize) -> MinervaResult<(Vec<f32>, Vec<f32>)> {
        if layer >= self.keys.len() {
            return Err(MinervaError::InferenceError(format!(
                "Layer index {} out of bounds",
                layer
            )));
        }
        if pos >= self.keys[layer].len() {
            return Err(MinervaError::InferenceError(format!(
                "Position {} out of bounds",
                pos
            )));
        }

        let mut k = Vec::new();
        let mut v = Vec::new();

        for head in &self.keys[layer][pos] {
            k.extend_from_slice(head);
        }
        for head in &self.values[layer][pos] {
            v.extend_from_slice(head);
        }

        Ok((k, v))
    }

    /// Clear cache
    pub fn clear(&mut self) {
        for layer in &mut self.keys {
            for pos in layer {
                for head in pos {
                    head.fill(0.0);
                }
            }
        }
        for layer in &mut self.values {
            for pos in layer {
                for head in pos {
                    head.fill(0.0);
                }
            }
        }
    }
}

/// Attention output
#[derive(Debug, Clone)]
pub struct AttentionOutput {
    /// Attention output tensor
    pub output: Vec<f32>,
    /// Attention weights for visualization
    pub weights: Option<Vec<f32>>,
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

/// Compute RMSNorm (Root Mean Square Layer Normalization)
pub fn rmsnorm(x: &[f32], weight: &[f32], eps: f32) -> MinervaResult<Vec<f32>> {
    if x.len() != weight.len() {
        return Err(MinervaError::InferenceError(format!(
            "Input size {} != weight size {}",
            x.len(),
            weight.len()
        )));
    }

    let rms = (x.iter().map(|v| v * v).sum::<f32>() / (x.len() as f32) + eps).sqrt();
    Ok(x.iter().zip(weight).map(|(a, b)| (a / rms) * b).collect())
}

/// Compute SiLU activation (Swish)
pub fn silu(x: &[f32]) -> Vec<f32> {
    x.iter().map(|v| v / (1.0 + (-v).exp())).collect()
}

/// Multi-head self-attention with rotary embeddings
pub struct MultiHeadAttention {
    num_heads: usize,
    head_dim: usize,
    rope: RoPEParams,
}

impl MultiHeadAttention {
    /// Create new attention module
    pub fn new(num_heads: usize, hidden_dim: usize) -> MinervaResult<Self> {
        if !hidden_dim.is_multiple_of(num_heads) {
            return Err(MinervaError::InferenceError(
                "Hidden dimension must be divisible by num_heads".to_string(),
            ));
        }

        let head_dim = hidden_dim / num_heads;
        Ok(Self {
            num_heads,
            head_dim,
            rope: RoPEParams::new(head_dim),
        })
    }

    /// Apply rotary embeddings to query and key
    fn apply_rope(&self, query: &mut [f32], key: &mut [f32], pos: usize) {
        for h in 0..self.num_heads {
            for d in (0..self.head_dim).step_by(2) {
                let angle = self.rope.get_angle(pos, d);
                let cos = angle.cos();
                let sin = angle.sin();

                // Apply rotation to query
                let q_idx_base = h * self.head_dim + d;
                if q_idx_base + 1 < query.len() {
                    let q0 = query[q_idx_base];
                    let q1 = query[q_idx_base + 1];
                    query[q_idx_base] = q0 * cos - q1 * sin;
                    query[q_idx_base + 1] = q0 * sin + q1 * cos;
                }

                // Apply rotation to key
                let k_idx_base = h * self.head_dim + d;
                if k_idx_base + 1 < key.len() {
                    let k0 = key[k_idx_base];
                    let k1 = key[k_idx_base + 1];
                    key[k_idx_base] = k0 * cos - k1 * sin;
                    key[k_idx_base + 1] = k0 * sin + k1 * cos;
                }
            }
        }
    }

    /// Compute attention scores between query and keys
    fn compute_scores(&self, query: &[f32], keys: &[f32], scale: f32) -> Vec<f32> {
        let num_keys = keys.len() / (self.num_heads * self.head_dim);
        let mut scores = vec![0.0; num_keys];

        for h in 0..self.num_heads {
            for (k_pos, score) in scores.iter_mut().enumerate() {
                let mut dot_product = 0.0;
                for d in 0..self.head_dim {
                    let q_idx = h * self.head_dim + d;
                    let k_idx = k_pos * (self.num_heads * self.head_dim) + h * self.head_dim + d;
                    if q_idx < query.len() && k_idx < keys.len() {
                        dot_product += query[q_idx] * keys[k_idx];
                    }
                }
                *score += dot_product * scale;
            }
        }

        scores
    }

    /// Forward pass for attention
    pub fn forward(
        &self,
        query: &mut [f32],
        key: &mut [f32],
        value: &[f32],
        pos: usize,
    ) -> MinervaResult<AttentionOutput> {
        if query.len() != key.len() {
            return Err(MinervaError::InferenceError(
                "Query and key dimensions must match".to_string(),
            ));
        }

        // Apply rotary embeddings
        self.apply_rope(query, key, pos);

        // Compute attention scores
        let scale = (self.head_dim as f32).sqrt().recip();
        let mut scores = self.compute_scores(query, key, scale);

        // Apply softmax
        softmax(&mut scores);

        // Compute output
        let mut output = vec![0.0; query.len()];
        let num_values = value.len() / (self.num_heads * self.head_dim);

        for h in 0..self.num_heads {
            for d in 0..self.head_dim {
                let mut sum = 0.0;
                for (v_pos, &score_weight) in scores.iter().enumerate() {
                    if v_pos < num_values {
                        let v_idx =
                            v_pos * (self.num_heads * self.head_dim) + h * self.head_dim + d;
                        if v_idx < value.len() {
                            sum += score_weight * value[v_idx];
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

/// Feed-forward network
pub struct FeedForward {
    hidden_size: usize,
    intermediate_size: usize,
}

impl FeedForward {
    /// Create new feed-forward layer
    pub fn new(hidden_size: usize, intermediate_size: usize) -> Self {
        Self {
            hidden_size,
            intermediate_size,
        }
    }

    /// Forward pass: hidden -> up -> activate -> down -> hidden
    pub fn forward(
        &self,
        x: &[f32],
        up_weight: &[f32],
        down_weight: &[f32],
    ) -> MinervaResult<Vec<f32>> {
        if x.len() != self.hidden_size {
            return Err(MinervaError::InferenceError(format!(
                "Input size {} != hidden size {}",
                x.len(),
                self.hidden_size
            )));
        }

        if up_weight.len() != self.hidden_size * self.intermediate_size {
            return Err(MinervaError::InferenceError(
                "Up weight dimension mismatch".to_string(),
            ));
        }

        // Up projection: x @ up_weight
        let mut hidden = vec![0.0; self.intermediate_size];
        for i in 0..self.intermediate_size {
            for j in 0..self.hidden_size {
                hidden[i] += x[j] * up_weight[i * self.hidden_size + j];
            }
        }

        // Apply SiLU activation
        hidden = silu(&hidden);

        // Down projection: hidden @ down_weight
        let mut output = vec![0.0; self.hidden_size];
        if down_weight.len() == self.intermediate_size * self.hidden_size {
            for i in 0..self.hidden_size {
                for j in 0..self.intermediate_size {
                    output[i] += hidden[j] * down_weight[j * self.hidden_size + i];
                }
            }
        }

        Ok(output)
    }
}

/// Token sampling strategy
#[derive(Debug, Clone, Copy)]
pub enum SamplingStrategy {
    /// Greedy decoding - always pick highest probability token
    Greedy,
    /// Top-k sampling - sample from k most likely tokens
    TopK(usize),
    /// Top-p (nucleus) sampling - sample from tokens with cumulative probability p
    TopP(f32),
}

/// Decoder for token generation
pub struct Decoder {
    vocab_size: usize,
    max_seq_len: usize,
}

impl Decoder {
    /// Create new decoder
    pub fn new(vocab_size: usize, max_seq_len: usize) -> Self {
        Self {
            vocab_size,
            max_seq_len,
        }
    }

    /// Sample next token from logits
    pub fn sample_token(
        &self,
        logits: &[f32],
        temperature: f32,
        strategy: SamplingStrategy,
    ) -> MinervaResult<usize> {
        if logits.len() != self.vocab_size {
            return Err(MinervaError::InferenceError(format!(
                "Logits size {} != vocab size {}",
                logits.len(),
                self.vocab_size
            )));
        }

        if temperature <= 0.0 {
            return Err(MinervaError::InferenceError(
                "Temperature must be positive".to_string(),
            ));
        }

        // Apply temperature scaling
        let probs = logits.iter().map(|l| l / temperature).collect::<Vec<_>>();

        // Apply softmax
        let max = probs.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut probs: Vec<f32> = probs.iter().map(|p| (p - max).exp()).collect();
        let sum: f32 = probs.iter().sum();
        for p in &mut probs {
            *p /= sum;
        }

        // Apply sampling strategy
        let token = match strategy {
            SamplingStrategy::Greedy => probs
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx)
                .ok_or_else(|| MinervaError::InferenceError("No valid token found".to_string()))?,

            SamplingStrategy::TopK(k) => {
                if k == 0 {
                    return Err(MinervaError::InferenceError("k must be > 0".to_string()));
                }
                let k = k.min(self.vocab_size);
                let mut indices: Vec<_> = (0..self.vocab_size).collect();
                indices.sort_by(|a, b| {
                    probs[*b]
                        .partial_cmp(&probs[*a])
                        .unwrap_or(std::cmp::Ordering::Equal)
                });

                // Zero out probabilities outside top-k
                for i in k..self.vocab_size {
                    probs[indices[i]] = 0.0;
                }

                // Renormalize
                let sum: f32 = probs.iter().sum();
                if sum > 0.0 {
                    for p in &mut probs {
                        *p /= sum;
                    }
                }

                // Sample from top-k
                self.sample_categorical(&probs)?
            }

            SamplingStrategy::TopP(p) => {
                if p <= 0.0 || p > 1.0 {
                    return Err(MinervaError::InferenceError(
                        "p must be in (0, 1]".to_string(),
                    ));
                }

                let mut indices: Vec<_> = (0..self.vocab_size).collect();
                indices.sort_by(|a, b| {
                    probs[*b]
                        .partial_cmp(&probs[*a])
                        .unwrap_or(std::cmp::Ordering::Equal)
                });

                let mut cumsum = 0.0;
                for i in indices.iter().cloned() {
                    cumsum += probs[i];
                    if cumsum < p {
                        // Keep this token
                    } else {
                        probs[i] = 0.0;
                    }
                }

                // Renormalize
                let sum: f32 = probs.iter().sum();
                if sum > 0.0 {
                    for p in &mut probs {
                        *p /= sum;
                    }
                }

                self.sample_categorical(&probs)?
            }
        };

        Ok(token)
    }

    /// Sample from categorical distribution
    fn sample_categorical(&self, probs: &[f32]) -> MinervaResult<usize> {
        let mut cumsum = 0.0;
        let rand = (std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as f32)
            / 1e9;
        let rand = rand.fract();

        for (i, &p) in probs.iter().enumerate() {
            cumsum += p;
            if rand < cumsum {
                return Ok(i);
            }
        }

        // Return last token if rounding errors
        Ok(probs.len().saturating_sub(1))
    }

    /// Generate tokens
    pub fn generate(
        &self,
        initial_tokens: &[usize],
        num_tokens: usize,
        temperature: f32,
        strategy: SamplingStrategy,
        mut forward: impl FnMut(&[usize]) -> MinervaResult<Vec<f32>>,
    ) -> MinervaResult<Vec<usize>> {
        if initial_tokens.is_empty() {
            return Err(MinervaError::InferenceError(
                "Initial tokens cannot be empty".to_string(),
            ));
        }

        if initial_tokens.len() + num_tokens > self.max_seq_len {
            return Err(MinervaError::InferenceError(
                "Sequence too long for max_seq_len".to_string(),
            ));
        }

        let mut tokens = initial_tokens.to_vec();
        let mut sequence = initial_tokens.to_vec();

        for _ in 0..num_tokens {
            let logits = forward(&tokens)?;
            let next_token = self.sample_token(&logits, temperature, strategy)?;
            tokens.push(next_token);
            sequence.push(next_token);
        }

        Ok(sequence)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rope_params_creation() {
        let rope = RoPEParams::new(64);
        assert_eq!(rope.head_dim, 64);
    }

    #[test]
    fn test_rope_angle_calculation() {
        let rope = RoPEParams::new(64);
        let angle = rope.get_angle(0, 0);
        assert_eq!(angle, 0.0);

        let angle = rope.get_angle(1, 0);
        assert!(angle > 0.0);
    }

    #[test]
    fn test_kv_cache_creation() {
        let config = KVCacheConfig {
            num_layers: 4,
            max_seq_len: 512,
            num_heads: 8,
            head_dim: 64,
        };
        let cache = KVCache::new(config);
        assert_eq!(cache.keys.len(), 4);
        assert_eq!(cache.keys[0].len(), 512);
    }

    #[test]
    fn test_kv_cache_store_and_get() {
        let config = KVCacheConfig {
            num_layers: 1,
            max_seq_len: 10,
            num_heads: 2,
            head_dim: 4,
        };
        let mut cache = KVCache::new(config);
        let k = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
        let v = vec![0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];

        assert!(cache.store(0, 0, (&k, &v)).is_ok());
        let (k_retrieved, v_retrieved) = cache.get(0, 0).unwrap();
        assert_eq!(k_retrieved.len(), 8);
        assert_eq!(v_retrieved.len(), 8);
    }

    #[test]
    fn test_rmsnorm() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let weight = vec![0.5, 0.5, 0.5, 0.5];
        let result = rmsnorm(&x, &weight, 1e-6).unwrap();
        assert_eq!(result.len(), 4);
        assert!(result.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_silu_activation() {
        let x = vec![0.0, 1.0, -1.0, 2.0];
        let result = silu(&x);
        assert_eq!(result.len(), 4);
        assert!(result.iter().all(|v| v.is_finite()));
        assert!(result[0] >= 0.0 && result[0] <= 0.1);
        assert!(result[1] > 0.7);
    }

    #[test]
    fn test_multihead_attention_creation() {
        let attn = MultiHeadAttention::new(8, 512).unwrap();
        assert_eq!(attn.num_heads, 8);
        assert_eq!(attn.head_dim, 64);
    }

    #[test]
    fn test_multihead_attention_invalid_dims() {
        let result = MultiHeadAttention::new(8, 510);
        assert!(result.is_err());
    }

    #[test]
    fn test_feedforward_creation() {
        let ff = FeedForward::new(512, 2048);
        assert_eq!(ff.hidden_size, 512);
        assert_eq!(ff.intermediate_size, 2048);
    }

    #[test]
    fn test_feedforward_forward() {
        let ff = FeedForward::new(4, 8);
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let up_w = vec![0.1; 32];
        let down_w = vec![0.1; 32];

        let result = ff.forward(&x, &up_w, &down_w).unwrap();
        assert_eq!(result.len(), 4);
    }

    #[test]
    fn test_decoder_creation() {
        let decoder = Decoder::new(32000, 2048);
        assert_eq!(decoder.vocab_size, 32000);
    }

    #[test]
    fn test_decoder_sample_greedy() {
        let decoder = Decoder::new(100, 512);
        let logits = vec![0.1; 100];
        let token = decoder
            .sample_token(&logits, 1.0, SamplingStrategy::Greedy)
            .unwrap();
        assert!(token < 100);
    }

    #[test]
    fn test_decoder_sample_topk() {
        let decoder = Decoder::new(100, 512);
        let mut logits = vec![0.1; 100];
        logits[0] = 1.0;
        logits[1] = 0.8;

        let token = decoder
            .sample_token(&logits, 1.0, SamplingStrategy::TopK(5))
            .unwrap();
        assert!(token < 100);
    }

    #[test]
    fn test_decoder_sample_topp() {
        let decoder = Decoder::new(100, 512);
        let mut logits = vec![0.1; 100];
        logits[0] = 1.0;
        logits[1] = 0.9;

        let token = decoder
            .sample_token(&logits, 1.0, SamplingStrategy::TopP(0.9))
            .unwrap();
        assert!(token < 100);
    }

    #[test]
    fn test_decoder_invalid_temperature() {
        let decoder = Decoder::new(100, 512);
        let logits = vec![0.1; 100];
        let result = decoder.sample_token(&logits, -1.0, SamplingStrategy::Greedy);
        assert!(result.is_err());
    }

    #[test]
    fn test_decoder_invalid_topk() {
        let decoder = Decoder::new(100, 512);
        let logits = vec![0.1; 100];
        let result = decoder.sample_token(&logits, 1.0, SamplingStrategy::TopK(0));
        assert!(result.is_err());
    }

    #[test]
    fn test_decoder_invalid_topp() {
        let decoder = Decoder::new(100, 512);
        let logits = vec![0.1; 100];
        let result = decoder.sample_token(&logits, 1.0, SamplingStrategy::TopP(0.0));
        assert!(result.is_err());
    }

    #[test]
    fn test_softmax() {
        let mut values = vec![1.0, 2.0, 3.0];
        softmax(&mut values);
        let sum: f32 = values.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
        assert!(values.iter().all(|v| *v >= 0.0 && *v <= 1.0));
    }

    #[test]
    fn test_sampling_strategy_greedy() {
        let strategy = SamplingStrategy::Greedy;
        match strategy {
            SamplingStrategy::Greedy => assert!(true),
            _ => assert!(false),
        }
    }

    #[test]
    fn test_sampling_strategy_topk() {
        let strategy = SamplingStrategy::TopK(10);
        match strategy {
            SamplingStrategy::TopK(k) => assert_eq!(k, 10),
            _ => assert!(false),
        }
    }

    #[test]
    fn test_sampling_strategy_topp() {
        let strategy = SamplingStrategy::TopP(0.9);
        match strategy {
            SamplingStrategy::TopP(p) => assert!((p - 0.9).abs() < 1e-5),
            _ => assert!(false),
        }
    }

    #[test]
    fn test_attention_output() {
        let output = AttentionOutput {
            output: vec![0.1, 0.2, 0.3],
            weights: Some(vec![0.5, 0.5]),
        };
        assert_eq!(output.output.len(), 3);
        assert!(output.weights.is_some());
    }

    #[test]
    fn test_rope_params_theta() {
        let rope = RoPEParams::new(128);
        assert_eq!(rope.theta_base, 10_000.0);
    }

    #[test]
    fn test_kv_cache_clear() {
        let config = KVCacheConfig {
            num_layers: 2,
            max_seq_len: 100,
            num_heads: 4,
            head_dim: 32,
        };
        let mut cache = KVCache::new(config);
        let k = vec![0.5; 128];
        let v = vec![0.5; 128];
        cache.store(0, 0, (&k, &v)).unwrap();

        cache.clear();
        let (k_cleared, v_cleared) = cache.get(0, 0).unwrap();
        assert!(k_cleared.iter().all(|&v| v == 0.0));
        assert!(v_cleared.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_decoder_generate_sequence() {
        let decoder = Decoder::new(100, 512);
        let initial = vec![1];
        let mut call_count = 0;

        let result = decoder
            .generate(&initial, 5, 1.0, SamplingStrategy::Greedy, |_tokens| {
                call_count += 1;
                Ok(vec![0.1; 100])
            })
            .unwrap();

        assert_eq!(result.len(), 6); // initial + 5 generated
    }

    #[test]
    fn test_rmsnorm_size_mismatch() {
        let x = vec![1.0, 2.0];
        let weight = vec![0.5, 0.5, 0.5];
        let result = rmsnorm(&x, &weight, 1e-6);
        assert!(result.is_err());
    }

    #[test]
    fn test_feedforward_size_mismatch() {
        let ff = FeedForward::new(4, 8);
        let x = vec![1.0, 2.0]; // Wrong size
        let up_w = vec![0.1; 32];
        let down_w = vec![0.1; 32];

        let result = ff.forward(&x, &up_w, &down_w);
        assert!(result.is_err());
    }
}
