use super::softmax_utils::{argmax, simple_random, softmax};
use super::temperature::{apply_temperature, TemperatureConfig};
/// Temperature-Based Token Sampling
///
/// Applies temperature scaling to adjust probability distribution sharpness,
/// then samples proportionally to probabilities.
use crate::error::{MinervaError, MinervaResult};

/// Temperature-based sampling
///
/// Applies temperature scaling before computing probabilities.
/// This provides fine-grained control over diversity.
///
/// # Arguments
/// * `logits`: Raw model outputs
/// * `config`: Temperature configuration
/// * `seed`: Random seed for sampling
///
/// # Returns
/// Sampled token ID
pub fn sample_temperature(
    logits: &[f32],
    config: &TemperatureConfig,
    seed: u64,
) -> MinervaResult<usize> {
    if logits.is_empty() {
        return Err(MinervaError::InferenceError("Empty logits".to_string()));
    }

    // Apply temperature
    let scaled_logits = apply_temperature(logits, config.temperature);

    // Compute softmax
    let probs = softmax(&scaled_logits);

    // Sample using simple LCG (Linear Congruential Generator)
    let random_val = simple_random(seed) % 1000000;
    let uniform = random_val as f32 / 1000000.0;

    let mut cumsum = 0.0;
    for (token, &prob) in probs.iter().enumerate() {
        cumsum += prob;
        if uniform < cumsum {
            return Ok(token);
        }
    }

    // Fallback: return argmax
    argmax(&scaled_logits)
        .ok_or_else(|| MinervaError::InferenceError("No valid tokens".to_string()))
}
