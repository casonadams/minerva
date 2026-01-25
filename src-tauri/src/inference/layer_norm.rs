/// Layer Normalization for Transformer Networks
///
/// Implements RMSNorm (Root Mean Square Layer Norm) which is simpler
/// and used in modern models like LLaMA instead of standard LayerNorm.
use crate::error::{MinervaError, MinervaResult};

/// Layer normalization configuration
#[derive(Debug, Clone)]
pub struct LayerNormConfig {
    pub seq_len: usize,
    pub hidden_size: usize,
    pub scale: Option<Vec<f32>>,
    pub eps: f32,
}

/// Layer normalization (RMSNorm variant used in LLaMA)
///
/// Normalizes activations to have approximately unit variance.
/// Uses RMSNorm (Root Mean Square Layer Norm) which is simpler than LayerNorm
/// and used in modern models like LLaMA.
///
/// # Formula
///
/// output = input * (scale / RMS(input))
/// where RMS = sqrt(mean(input^2) + eps)
///
/// # Arguments
///
/// * `input`: Shape (seq_len, hidden_size)
/// * `config`: Layer normalization configuration
///
/// # Returns
///
/// Normalized output of shape (seq_len, hidden_size)
pub fn layer_norm(input: &[f32], config: &LayerNormConfig) -> MinervaResult<Vec<f32>> {
    let LayerNormConfig {
        seq_len,
        hidden_size,
        scale,
        eps,
    } = config;
    let seq_len_val = *seq_len;
    let hidden_size_val = *hidden_size;
    let eps_val = *eps;

    if input.len() != seq_len_val * hidden_size_val {
        return Err(MinervaError::InferenceError(
            "Input size mismatch".to_string(),
        ));
    }

    let scale_vec = if let Some(s) = scale {
        if s.len() != hidden_size_val {
            return Err(MinervaError::InferenceError(
                "Scale size mismatch".to_string(),
            ));
        }
        s.clone()
    } else {
        vec![1.0; hidden_size_val]
    };

    let mut output = vec![0.0; seq_len_val * hidden_size_val];

    // Normalize each position independently
    for (i, input_row) in input.chunks(hidden_size_val).enumerate() {
        // Compute RMS (Root Mean Square)
        let rms_sq: f32 = input_row.iter().map(|x| x * x).sum::<f32>() / hidden_size_val as f32;
        let rms = (rms_sq + eps_val).sqrt();

        // Apply normalization and scaling
        for (j, (input_val, scale_val)) in input_row.iter().zip(scale_vec.iter()).enumerate() {
            output[i * hidden_size_val + j] = (input_val / rms) * scale_val;
        }
    }

    Ok(output)
}
