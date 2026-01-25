use super::softmax_utils::{simple_random, softmax};
use super::temperature::apply_temperature;
/// Top-K Token Sampling
///
/// Restricts sampling to the K most probable tokens, eliminating very low
/// probability (nonsense) tokens while maintaining diversity.
use crate::error::{MinervaError, MinervaResult};
use std::cmp::Ordering;

/// Top-K sampling configuration
#[derive(Debug, Clone, Copy)]
pub struct TopKConfig {
    pub k: usize,
    pub temperature: f32,
}

impl TopKConfig {
    /// Standard top-K (50 tokens, standard temperature)
    pub fn standard() -> Self {
        Self {
            k: 50,
            temperature: 1.0,
        }
    }
}

/// Top-K sampling
///
/// Selects uniformly at random from the K most probable tokens.
/// Prevents sampling of very low-probability (nonsense) tokens.
///
/// # Arguments
/// * `logits`: Raw model outputs
/// * `config`: Top-K configuration
/// * `seed`: Random seed for sampling
///
/// # Returns
/// Sampled token ID from top K
pub fn sample_top_k(logits: &[f32], config: &TopKConfig, seed: u64) -> MinervaResult<usize> {
    if logits.is_empty() {
        return Err(MinervaError::InferenceError("Empty logits".to_string()));
    }

    if config.k == 0 {
        return Err(MinervaError::InferenceError("K must be > 0".to_string()));
    }

    let k = config.k.min(logits.len());

    // Apply temperature
    let scaled_logits = apply_temperature(logits, config.temperature);

    // Get indices sorted by logits (descending)
    let mut indices: Vec<usize> = (0..scaled_logits.len()).collect();
    indices.sort_by(|&a, &b| {
        scaled_logits[b]
            .partial_cmp(&scaled_logits[a])
            .unwrap_or(Ordering::Equal)
    });

    // Keep only top K
    let top_k_indices = &indices[0..k];

    // Compute softmax over top K
    let top_k_logits: Vec<f32> = top_k_indices.iter().map(|&i| scaled_logits[i]).collect();
    let top_k_probs = softmax(&top_k_logits);

    // Sample uniformly from top K
    let random_val = simple_random(seed) % 1000000;
    let uniform = random_val as f32 / 1000000.0;

    let mut cumsum = 0.0;
    for (idx, &prob) in top_k_probs.iter().enumerate() {
        cumsum += prob;
        if uniform < cumsum {
            return Ok(top_k_indices[idx]);
        }
    }

    // Fallback: return top-1
    Ok(top_k_indices[0])
}
