/// Positional Encoding for Transformer Networks
///
/// Provides absolute positional encoding using sine and cosine functions.
/// This enables the transformer to understand token positions in sequences,
/// since the base attention mechanism has no inherent positional awareness.
use crate::error::MinervaResult;

/// Positional encoding configuration
#[derive(Debug, Clone, Copy)]
pub struct PositionConfig {
    pub seq_len: usize,
    pub hidden_size: usize,
    pub base: f32, // Usually 10000 for absolute, 100000 for RoPE
}

/// Absolute positional encoding
///
/// Creates sine/cosine positional embeddings (like in original Transformer)
/// Formula: PE(pos, 2i) = sin(pos / 10000^(2i / d))
///          PE(pos, 2i+1) = cos(pos / 10000^(2i / d))
///
/// # Arguments
/// * `seq_len`: Sequence length
/// * `hidden_size`: Hidden dimension
/// * `base`: Base for exponential (typically 10000)
///
/// # Returns
/// Positional encodings (shape: seq_len × hidden_size)
pub fn create_position_encoding(config: &PositionConfig) -> Vec<f32> {
    let PositionConfig {
        seq_len,
        hidden_size,
        base,
    } = config;

    let seq_len_val = *seq_len;
    let hidden_size_val = *hidden_size;
    let base_val = *base;

    let mut pe = vec![0.0; seq_len_val * hidden_size_val];

    for pos in 0..seq_len_val {
        for i in 0..hidden_size_val {
            let div_term = (base_val).powf(2.0 * (i as f32 / 2.0) / hidden_size_val as f32);
            let arg = pos as f32 / div_term;

            if i % 2 == 0 {
                // Even indices: sin
                pe[pos * hidden_size_val + i] = arg.sin();
            } else {
                // Odd indices: cos
                pe[pos * hidden_size_val + i] = arg.cos();
            }
        }
    }

    pe
}

/// Add positional encoding to embeddings
pub fn add_position_encoding(
    embeddings: &[f32],
    position_encoding: &[f32],
) -> MinervaResult<Vec<f32>> {
    if embeddings.len() != position_encoding.len() {
        return Err(crate::error::MinervaError::InferenceError(format!(
            "Shape mismatch: embeddings {} vs positions {}",
            embeddings.len(),
            position_encoding.len()
        )));
    }

    let mut result = embeddings.to_vec();
    for (i, &pe) in position_encoding.iter().enumerate() {
        result[i] += pe;
    }

    Ok(result)
}
