use super::softmax_utils::{simple_random, softmax};
use super::temperature::apply_temperature;
/// Top-P (Nucleus) Token Sampling
///
/// Samples from the smallest set of tokens whose cumulative probability
/// exceeds threshold P. Provides more natural diversity than top-K.
use crate::error::{MinervaError, MinervaResult};
use std::cmp::Ordering;

/// Top-P (nucleus) sampling configuration
#[derive(Debug, Clone, Copy)]
pub struct TopPConfig {
    pub p: f32,
    pub temperature: f32,
}

impl TopPConfig {
    /// Standard nucleus sampling (95% probability, standard temperature)
    pub fn standard() -> Self {
        Self {
            p: 0.95,
            temperature: 1.0,
        }
    }
}

/// Top-P (nucleus) sampling
///
/// Keeps the smallest set of tokens with cumulative probability >= P.
/// More natural than fixed top-K, adapts to probability distribution.
///
/// # Arguments
/// * `logits`: Raw model outputs
/// * `config`: Top-P configuration
/// * `seed`: Random seed for sampling
///
/// # Returns
/// Sampled token ID from nucleus
pub fn sample_top_p(logits: &[f32], config: &TopPConfig, seed: u64) -> MinervaResult<usize> {
    if logits.is_empty() {
        return Err(MinervaError::InferenceError("Empty logits".to_string()));
    }

    if !(0.0..=1.0).contains(&config.p) {
        return Err(MinervaError::InferenceError(
            "P must be in [0, 1]".to_string(),
        ));
    }

    // Apply temperature
    let scaled_logits = apply_temperature(logits, config.temperature);

    // Compute softmax
    let probs = softmax(&scaled_logits);

    // Sort probabilities (descending) and keep track of indices
    let mut indexed_probs: Vec<(usize, f32)> =
        probs.iter().enumerate().map(|(i, &p)| (i, p)).collect();
    indexed_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));

    // Find cutoff where cumulative probability >= p
    let mut cumsum = 0.0;
    let cutoff_idx = indexed_probs
        .iter()
        .position(|(_, prob)| {
            cumsum += prob;
            cumsum >= config.p
        })
        .unwrap_or(indexed_probs.len() - 1)
        + 1;

    // Keep top P tokens
    let nucleus_probs: Vec<(usize, f32)> = indexed_probs[0..cutoff_idx].to_vec();

    // Normalize nucleus probabilities
    let sum: f32 = nucleus_probs.iter().map(|(_, p)| p).sum();
    let normalized_nucleus: Vec<(usize, f32)> = nucleus_probs
        .iter()
        .map(|(token, prob)| (*token, prob / sum))
        .collect();

    // Sample from nucleus
    let random_val = simple_random(seed) % 1000000;
    let uniform = random_val as f32 / 1000000.0;

    let mut cumsum = 0.0;
    for (token, prob) in normalized_nucleus {
        cumsum += prob;
        if uniform < cumsum {
            return Ok(token);
        }
    }

    // Fallback
    Ok(nucleus_probs[0].0)
}
