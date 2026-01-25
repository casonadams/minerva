/// Token Embedding Layer for Transformer Networks
///
/// Maps token IDs to dense embeddings using a pre-trained embedding matrix.
/// This layer is the first step in the transformer pipeline, converting discrete
/// token IDs into continuous vectors suitable for neural processing.
use crate::error::{MinervaError, MinervaResult};

/// Embedding configuration
#[derive(Debug, Clone)]
pub struct EmbeddingConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
}

/// Token embedding layer
///
/// Maps token IDs to dense embeddings
///
/// # Arguments
/// * `tokens`: Token IDs (shape: seq_len)
/// * `embeddings`: Full embedding matrix (shape: vocab_size × hidden_size)
/// * `config`: Embedding configuration
///
/// # Returns
/// Embedded representations (shape: seq_len × hidden_size)
pub fn embed_tokens(
    tokens: &[usize],
    embeddings: &[f32],
    config: &EmbeddingConfig,
) -> MinervaResult<Vec<f32>> {
    let EmbeddingConfig {
        vocab_size,
        hidden_size,
    } = config;

    if embeddings.len() != vocab_size * hidden_size {
        return Err(MinervaError::InferenceError(format!(
            "Embedding matrix shape mismatch: expected {}, got {}",
            vocab_size * hidden_size,
            embeddings.len()
        )));
    }

    let mut output = vec![0.0; tokens.len() * hidden_size];

    for (i, &token_id) in tokens.iter().enumerate() {
        if token_id >= *vocab_size {
            return Err(MinervaError::InferenceError(format!(
                "Token ID {} out of vocabulary size {}",
                token_id, vocab_size
            )));
        }

        // Copy embedding row for this token
        let start = token_id * hidden_size;
        let end = start + hidden_size;
        output[i * hidden_size..(i + 1) * hidden_size].copy_from_slice(&embeddings[start..end]);
    }

    Ok(output)
}
