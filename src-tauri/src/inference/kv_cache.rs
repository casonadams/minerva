/// Key-Value Cache for Efficient Inference
///
/// During token generation, we reuse previously computed key and value tensors
/// to avoid recomputing attention for all past tokens. This cache stores
/// key-value pairs for each layer and position in the sequence.
///
/// Structure: keys[layer][pos][head][head_dim]
use crate::error::{MinervaError, MinervaResult};

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

/// Parameters for KV cache store operation
#[derive(Debug, Clone)]
pub struct KVStoreParams {
    /// Layer index
    pub layer: usize,
    /// Position index
    pub pos: usize,
    /// Key data
    pub k: Vec<f32>,
    /// Value data
    pub v: Vec<f32>,
}

impl KVStoreParams {
    /// Create builder for KV store params
    pub fn builder(k: Vec<f32>, v: Vec<f32>) -> KVStoreParamsBuilder {
        KVStoreParamsBuilder {
            layer: 0,
            pos: 0,
            k,
            v,
        }
    }
}

/// Builder for KVStoreParams to reduce function parameters
pub struct KVStoreParamsBuilder {
    layer: usize,
    pos: usize,
    k: Vec<f32>,
    v: Vec<f32>,
}

impl KVStoreParamsBuilder {
    /// Set layer index
    pub fn layer(mut self, layer: usize) -> Self {
        self.layer = layer;
        self
    }

    /// Set position index
    pub fn pos(mut self, pos: usize) -> Self {
        self.pos = pos;
        self
    }

    /// Build KVStoreParams
    pub fn build(self) -> KVStoreParams {
        KVStoreParams {
            layer: self.layer,
            pos: self.pos,
            k: self.k,
            v: self.v,
        }
    }
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
    pub fn store(&mut self, params: KVStoreParams) -> MinervaResult<()> {
        if params.layer >= self.keys.len() {
            return Err(MinervaError::InferenceError(format!(
                "Layer index {} out of bounds",
                params.layer
            )));
        }
        if params.pos >= self.keys[params.layer].len() {
            return Err(MinervaError::InferenceError(format!(
                "Position {} out of bounds",
                params.pos
            )));
        }

        // Flatten head dimension
        let num_heads = self.keys[params.layer][params.pos].len();
        let head_dim = self.keys[params.layer][params.pos][0].len();

        for h in 0..num_heads {
            let start = h * head_dim;
            let end = start + head_dim;
            if start < params.k.len() && end <= params.k.len() {
                self.keys[params.layer][params.pos][h].copy_from_slice(&params.k[start..end]);
                self.values[params.layer][params.pos][h].copy_from_slice(&params.v[start..end]);
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
