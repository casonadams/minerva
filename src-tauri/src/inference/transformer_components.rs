pub use super::activation::{Activation, apply_activation, gelu, relu, silu};
pub use super::embedding::{EmbeddingConfig, embed_tokens};
pub use super::feedforward::{FeedforwardConfig, FeedforwardWeights, feedforward};
pub use super::position_encoding::{
    PositionConfig, add_position_encoding, create_position_encoding,
};
use super::transformer_layers::{LayerNormConfig as LNCfg, MultiHeadConfig};
/// Transformer Components - Phase 9 Day 3 (Refactored Phase 13)
/// Re-exports components from focused modules: activation, embedding, position_encoding, feedforward
use super::transformer_layers::{layer_norm, multi_head_attention};
use crate::error::MinervaResult;

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

struct AttentionParams {
    seq_len: usize,
    hidden_size: usize,
    num_heads: usize,
    causal: bool,
    eps: f32,
}

fn apply_attention_layer(
    input: &[f32],
    attn_scale: Option<&[f32]>,
    params: &AttentionParams,
) -> MinervaResult<Vec<f32>> {
    let ln_cfg = LNCfg {
        seq_len: params.seq_len,
        hidden_size: params.hidden_size,
        scale: attn_scale.map(|s| s.to_vec()),
        eps: params.eps,
    };
    let normed = layer_norm(input, &ln_cfg)?;
    let mha_cfg = MultiHeadConfig {
        seq_len: params.seq_len,
        hidden_size: params.hidden_size,
        num_heads: params.num_heads,
        causal: params.causal,
    };
    let attn_out = multi_head_attention(&normed, &mha_cfg)?;
    let mut result = input.to_vec();
    for (i, val) in attn_out.iter().enumerate() {
        result[i] += val;
    }
    Ok(result)
}

struct FeedforwardLayerInput<'a> {
    ff_up: &'a [f32],
    ff_down: &'a [f32],
    seq_len: usize,
    hidden_size: usize,
    intermediate_size: usize,
    activation: Activation,
    eps: f32,
}

fn apply_feedforward_layer(
    input: &[f32],
    ff_input: &FeedforwardLayerInput,
) -> MinervaResult<Vec<f32>> {
    let ln_cfg = LNCfg {
        seq_len: ff_input.seq_len,
        hidden_size: ff_input.hidden_size,
        scale: None,
        eps: ff_input.eps,
    };
    let normed = layer_norm(input, &ln_cfg)?;
    let ff_cfg = FeedforwardConfig {
        seq_len: ff_input.seq_len,
        hidden_size: ff_input.hidden_size,
        intermediate_size: ff_input.intermediate_size,
        activation: ff_input.activation,
    };
    let ff_weights = FeedforwardWeights {
        up: ff_input.ff_up,
        down: ff_input.ff_down,
    };
    let ff_out = feedforward(&normed, &ff_weights, &ff_cfg)?;
    let mut result = input.to_vec();
    for (i, val) in ff_out.iter().enumerate() {
        result[i] += val;
    }
    Ok(result)
}

/// Complete transformer block: LayerNorm → Attention → Residual → LayerNorm → FF → Residual
pub fn transformer_block(
    input: &[f32],
    weights: &TransformerBlockWeights,
    config: &TransformerBlockConfig,
) -> MinervaResult<Vec<f32>> {
    let TransformerBlockConfig {
        seq_len,
        hidden_size,
        num_heads,
        intermediate_size,
        activation,
        causal,
        eps,
    } = config;
    if input.len() != seq_len * hidden_size {
        return Err(crate::error::MinervaError::InferenceError(
            "Input shape mismatch".to_string(),
        ));
    }
    let attn_params = AttentionParams {
        seq_len: *seq_len,
        hidden_size: *hidden_size,
        num_heads: *num_heads,
        causal: *causal,
        eps: *eps,
    };
    let after_attn = apply_attention_layer(input, weights.attn_scale, &attn_params)?;
    let ff_input = FeedforwardLayerInput {
        ff_up: weights.ff_up,
        ff_down: weights.ff_down,
        seq_len: *seq_len,
        hidden_size: *hidden_size,
        intermediate_size: *intermediate_size,
        activation: *activation,
        eps: *eps,
    };
    apply_feedforward_layer(&after_attn, &ff_input)
}
