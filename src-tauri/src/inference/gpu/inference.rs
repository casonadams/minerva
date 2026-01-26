/// High-Performance GPU Inference Engine
///
/// Optimized for maximum throughput (tokens/second)
/// Focuses on: GQA attention, KV caching, batching
use crate::error::MinervaResult;
use ndarray::{Array1, Array2, Array3, s};
use std::time::Instant;

/// Lightweight inference engine optimized for speed
pub struct FastInferenceEngine {
    /// Cached KV pairs from previous tokens
    pub kv_cache: KVCacheOptimized,
    /// Benchmark metrics
    pub metrics: InferenceMetrics,
}

/// High-performance KV cache with batch support
pub struct KVCacheOptimized {
    /// K cache: (layer, seq_len, num_kv_heads, head_dim)
    k_caches: Vec<Vec<Array2<f32>>>,
    /// V cache: (layer, seq_len, num_kv_heads, head_dim)
    v_caches: Vec<Vec<Array2<f32>>>,
    /// Current sequence length
    pub seq_len: usize,
    /// Max sequence length
    pub max_seq_len: usize,
    /// Number of layers
    pub num_layers: usize,
    /// Number of KV heads (8 for GPT-OSS)
    pub num_kv_heads: usize,
    /// Head dimension
    pub head_dim: usize,
}

/// Metrics for performance tracking
#[derive(Clone, Default, Debug)]
pub struct InferenceMetrics {
    pub load_time_ms: u64,
    pub forward_pass_ms: u64,
    pub attention_time_ms: u64,
    pub mlp_time_ms: u64,
    pub tokens_processed: usize,
    pub tokens_per_second: f32,
}

impl FastInferenceEngine {
    /// Create new inference engine
    pub fn new(
        num_layers: usize,
        num_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
    ) -> Self {
        let kv_cache = KVCacheOptimized::new(num_layers, num_kv_heads, head_dim, max_seq_len);

        Self {
            kv_cache,
            metrics: InferenceMetrics::default(),
        }
    }

    /// Single forward pass with timing
    pub fn forward_timed(
        &mut self,
        hidden_states: &Array2<f32>,
    ) -> MinervaResult<(Array2<f32>, u64)> {
        let start = Instant::now();

        // Placeholder: just return input for now
        // Will be replaced with actual forward pass
        let output = hidden_states.clone();

        let elapsed_ms = start.elapsed().as_millis() as u64;
        self.metrics.forward_pass_ms = elapsed_ms;

        Ok((output, elapsed_ms))
    }

    /// Get throughput metrics
    pub fn get_metrics(&self) -> InferenceMetrics {
        self.metrics.clone()
    }
}

impl KVCacheOptimized {
    /// Create new KV cache
    pub fn new(
        num_layers: usize,
        num_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
    ) -> Self {
        // Pre-allocate cache for all layers
        // Each layer has K and V for all tokens up to max_seq_len
        let k_caches = vec![Vec::with_capacity(max_seq_len); num_layers];
        let v_caches = vec![Vec::with_capacity(max_seq_len); num_layers];

        Self {
            k_caches,
            v_caches,
            seq_len: 0,
            max_seq_len,
            num_layers,
            num_kv_heads,
            head_dim,
        }
    }

    /// Append new K, V to cache (called after each token)
    pub fn append(
        &mut self,
        layer_idx: usize,
        k_new: &Array2<f32>,
        v_new: &Array2<f32>,
    ) -> MinervaResult<()> {
        if layer_idx >= self.num_layers {
            return Err(crate::error::MinervaError::InferenceError(
                "Layer index out of bounds".to_string(),
            ));
        }

        // Check if we're at capacity (before incrementing for this token)
        if self.k_caches[layer_idx].len() >= self.max_seq_len {
            return Err(crate::error::MinervaError::ContextLimitExceeded {
                max: self.max_seq_len,
                required: self.k_caches[layer_idx].len() + 1,
            });
        }

        // Append new K, V
        self.k_caches[layer_idx].push(k_new.clone());
        self.v_caches[layer_idx].push(v_new.clone());

        Ok(())
    }

    /// Get cached K, V for a layer
    pub fn get(&self, layer_idx: usize) -> MinervaResult<(Vec<&Array2<f32>>, Vec<&Array2<f32>>)> {
        if layer_idx >= self.num_layers {
            return Err(crate::error::MinervaError::InferenceError(
                "Layer index out of bounds".to_string(),
            ));
        }

        let k_refs: Vec<&Array2<f32>> = self.k_caches[layer_idx].iter().collect();
        let v_refs: Vec<&Array2<f32>> = self.v_caches[layer_idx].iter().collect();

        Ok((k_refs, v_refs))
    }

    /// Reset cache
    pub fn reset(&mut self) {
        self.seq_len = 0;
        for layer in 0..self.num_layers {
            self.k_caches[layer].clear();
            self.v_caches[layer].clear();
        }
    }

    /// Increment sequence length
    pub fn next_token(&mut self) {
        self.seq_len += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kv_cache_creation() {
        let cache = KVCacheOptimized::new(24, 8, 360, 4096);
        assert_eq!(cache.num_layers, 24);
        assert_eq!(cache.num_kv_heads, 8);
        assert_eq!(cache.head_dim, 360);
        assert_eq!(cache.seq_len, 0);
    }

    #[test]
    fn test_kv_cache_append() {
        let mut cache = KVCacheOptimized::new(24, 8, 360, 4096);
        let k_new = Array2::zeros((8, 360));
        let v_new = Array2::zeros((8, 360));

        assert!(cache.append(0, &k_new, &v_new).is_ok());
        assert!(cache.get(0).is_ok());
    }

    #[test]
    fn test_kv_cache_bounds() {
        let mut cache = KVCacheOptimized::new(24, 8, 360, 2);
        let k = Array2::zeros((8, 360));
        let v = Array2::zeros((8, 360));

        assert!(cache.append(0, &k, &v).is_ok());
        assert!(cache.append(0, &k, &v).is_ok());
        assert!(cache.append(0, &k, &v).is_err()); // Should fail - exceeded max_seq_len
    }

    #[test]
    fn test_inference_engine() {
        let engine = FastInferenceEngine::new(24, 8, 360, 4096);
        assert_eq!(engine.kv_cache.num_layers, 24);
        let metrics = engine.get_metrics();
        assert_eq!(metrics.tokens_processed, 0);
    }
}
