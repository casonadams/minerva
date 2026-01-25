use super::activation::{Activation, apply_activation};
/// Feedforward Network Layer for Transformer Networks
///
/// Implements the feed-forward network (FFN) component of transformer blocks.
/// Typically applies: Dense(hidden → intermediate) → Activation → Dense(intermediate → hidden)
use crate::error::{MinervaError, MinervaResult};

/// Feedforward network configuration
#[derive(Debug, Clone, Copy)]
pub struct FeedforwardConfig {
    pub seq_len: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub activation: Activation,
}

/// Feedforward weights (up and down projections)
#[derive(Debug)]
pub struct FeedforwardWeights<'a> {
    pub up: &'a [f32],
    pub down: &'a [f32],
}

/// Projection dimensions
struct ProjectionDims {
    seq_len: usize,
    hidden_size: usize,
    intermediate_size: usize,
}

/// Validate feedforward input shapes
fn validate_shapes(
    weights: &FeedforwardWeights,
    dims: &ProjectionDims,
    input_len: usize,
) -> MinervaResult<()> {
    if input_len != dims.seq_len * dims.hidden_size {
        return Err(MinervaError::InferenceError(
            "Input shape mismatch".to_string(),
        ));
    }
    if weights.up.len() != dims.hidden_size * dims.intermediate_size {
        return Err(MinervaError::InferenceError(
            "Up weight shape mismatch".to_string(),
        ));
    }
    if weights.down.len() != dims.intermediate_size * dims.hidden_size {
        return Err(MinervaError::InferenceError(
            "Down weight shape mismatch".to_string(),
        ));
    }
    Ok(())
}

/// Compute up projection: (seq_len, hidden_size) @ (hidden_size, intermediate_size)
fn compute_up_projection(input: &[f32], w_up: &[f32], dims: &ProjectionDims) -> Vec<f32> {
    let mut up_output = vec![0.0; dims.seq_len * dims.intermediate_size];
    for i in 0..dims.seq_len {
        for j in 0..dims.intermediate_size {
            let mut sum = 0.0;
            for k in 0..dims.hidden_size {
                sum += input[i * dims.hidden_size + k] * w_up[k * dims.intermediate_size + j];
            }
            up_output[i * dims.intermediate_size + j] = sum;
        }
    }
    up_output
}

/// Compute down projection: (seq_len, intermediate_size) @ (intermediate_size, hidden_size)
fn compute_down_projection(activated: &[f32], w_down: &[f32], dims: &ProjectionDims) -> Vec<f32> {
    let mut output = vec![0.0; dims.seq_len * dims.hidden_size];
    for i in 0..dims.seq_len {
        for j in 0..dims.hidden_size {
            let mut sum = 0.0;
            for k in 0..dims.intermediate_size {
                sum += activated[i * dims.intermediate_size + k] * w_down[k * dims.hidden_size + j];
            }
            output[i * dims.hidden_size + j] = sum;
        }
    }
    output
}

/// Feedforward network layer
///
/// Implements: Dense(hidden) → Activation → Dense(hidden)
/// Typically: hidden_size → 4*hidden_size → hidden_size
///
/// # Arguments
/// * `input`: Sequence of tokens (shape: seq_len × hidden_size)
/// * `weights`: Up and down projection weights
/// * `config`: Feedforward configuration
///
/// # Returns
/// Feedforward output (shape: seq_len × hidden_size)
pub fn feedforward(
    input: &[f32],
    weights: &FeedforwardWeights,
    config: &FeedforwardConfig,
) -> MinervaResult<Vec<f32>> {
    let FeedforwardConfig {
        seq_len,
        hidden_size,
        intermediate_size,
        activation,
    } = config;

    let dims = ProjectionDims {
        seq_len: *seq_len,
        hidden_size: *hidden_size,
        intermediate_size: *intermediate_size,
    };

    validate_shapes(weights, &dims, input.len())?;

    let up_output = compute_up_projection(input, weights.up, &dims);
    let activated = apply_activation(&up_output, *activation);
    let output = compute_down_projection(&activated, weights.down, &dims);

    Ok(output)
}
