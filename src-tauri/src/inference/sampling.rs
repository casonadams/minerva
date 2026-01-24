/// Advanced Sampling Strategies - Phase 9 Day 5
///
/// This module implements production-ready token sampling strategies for
/// generating text from language model logits. It provides multiple methods
/// to control diversity and quality of generated output.
///
/// # Sampling Strategies
///
/// 1. **Greedy**: Always pick the highest probability token
///    - Deterministic
///    - Best for factual tasks
///    - Can produce repetitive text
///
/// 2. **Temperature**: Scale logits before softmax to control sharpness
///    - T < 1.0: Sharper (less diverse)
///    - T = 1.0: Original
///    - T > 1.0: Softer (more diverse)
///    - Useful control for all strategies
///
/// 3. **Top-K**: Keep only top K tokens by probability
///    - Eliminates low-probability tokens
///    - Prevents nonsense tokens
///    - Common in production (K=50 or K=100)
///
/// 4. **Top-P (Nucleus)**: Keep tokens until cumulative probability reaches P
///    - Adaptive token count
///    - More natural than fixed top-K
///    - Standard in modern models (P=0.95)
///
/// # Example
///
/// ```rust
/// let logits = vec![1.0, 2.0, 0.5, 3.0];  // Raw model outputs
/// let sample = sample_nucleus(&logits, 0.95, 42)?;
/// println!("Sampled token: {}", sample);
/// ```
use crate::error::{MinervaError, MinervaResult};
use std::cmp::Ordering;

// ============================================================================
// Sampling Configuration
// ============================================================================

/// Temperature for softmax scaling
///
/// Temperature controls how "sharp" or "smooth" the probability distribution is:
/// - Below 1.0 (e.g., 0.5): Distribution becomes sharper, less diverse
/// - Above 1.0 (e.g., 2.0): Distribution becomes softer, more diverse
#[derive(Debug, Clone, Copy)]
pub struct TemperatureConfig {
    pub temperature: f32,
}

impl TemperatureConfig {
    /// Create standard temperature (1.0 = no change)
    pub fn standard() -> Self {
        Self { temperature: 1.0 }
    }

    /// Create sharp temperature (less diversity)
    pub fn sharp() -> Self {
        Self { temperature: 0.7 }
    }

    /// Create soft temperature (more diversity)
    pub fn soft() -> Self {
        Self { temperature: 1.5 }
    }
}

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

// ============================================================================
// Softmax and Temperature
// ============================================================================

/// Apply temperature scaling to logits
///
/// Temperature scales the logits before softmax:
/// - Lower temperature: sharper probability distribution
/// - Higher temperature: softer probability distribution
fn apply_temperature(logits: &[f32], temperature: f32) -> Vec<f32> {
    if (temperature - 1.0).abs() < 0.001 {
        // Skip scaling if temperature is 1.0
        logits.to_vec()
    } else {
        logits.iter().map(|&x| x / temperature).collect()
    }
}

/// Compute softmax probabilities from logits
fn softmax(logits: &[f32]) -> Vec<f32> {
    let mut probs = logits.to_vec();

    // Find max for numerical stability
    let max_val = probs.iter().copied().fold(f32::NEG_INFINITY, f32::max);

    // Compute exp and sum
    let mut sum_exp = 0.0;
    for prob in &mut probs {
        *prob = (*prob - max_val).exp();
        sum_exp += *prob;
    }

    // Normalize
    if sum_exp > 0.0 {
        for prob in &mut probs {
            *prob /= sum_exp;
        }
    }

    probs
}

// ============================================================================
// Sampling Functions
// ============================================================================

/// Greedy sampling: always pick the highest probability token
///
/// This is deterministic and selects the most likely token at each step.
/// Good for tasks where consistency is important.
///
/// # Arguments
/// * `logits`: Raw model outputs
///
/// # Returns
/// Token ID with highest probability
pub fn sample_greedy(logits: &[f32]) -> MinervaResult<usize> {
    if logits.is_empty() {
        return Err(MinervaError::InferenceError("Empty logits".to_string()));
    }

    let (token, _prob) = logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Equal))
        .ok_or_else(|| MinervaError::InferenceError("No valid tokens".to_string()))?;

    Ok(token)
}

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
    sample_greedy(&scaled_logits)
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
    for (i, &prob) in top_k_probs.iter().enumerate() {
        cumsum += prob;
        if uniform < cumsum {
            return Ok(top_k_indices[i]);
        }
    }

    // Fallback
    Ok(top_k_indices[0])
}

/// Top-P (nucleus) sampling
///
/// Selects tokens in order of probability until the cumulative
/// probability reaches P. This is more adaptive than top-K.
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

/// Simple pseudo-random number generator (Linear Congruential Generator)
///
/// This is deterministic given a seed, making it reproducible.
/// Not cryptographically secure, but sufficient for sampling.
fn simple_random(seed: u64) -> u64 {
    const A: u64 = 1103515245;
    const C: u64 = 12345;
    const M: u64 = 2u64.pow(31);

    (A.wrapping_mul(seed).wrapping_add(C)) % M
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_temperature_config_creation() {
        let standard = TemperatureConfig::standard();
        assert_eq!(standard.temperature, 1.0);

        let sharp = TemperatureConfig::sharp();
        assert!(sharp.temperature < 1.0);

        let soft = TemperatureConfig::soft();
        assert!(soft.temperature > 1.0);
    }

    #[test]
    fn test_apply_temperature_unchanged() {
        let logits = vec![1.0, 2.0, 3.0];
        let scaled = apply_temperature(&logits, 1.0);
        assert_eq!(scaled, logits);
    }

    #[test]
    fn test_apply_temperature_scaling() {
        let logits = vec![1.0, 2.0, 3.0];

        // Temperature < 1 should amplify differences
        let sharp = apply_temperature(&logits, 0.5);
        assert!(sharp[2] > logits[2]); // Larger logits amplified more

        // Temperature > 1 should reduce differences
        let soft = apply_temperature(&logits, 2.0);
        assert!(soft[2] < logits[2]); // Larger logits reduced more
    }

    #[test]
    fn test_softmax_properties() {
        let logits = vec![1.0, 2.0, 3.0];
        let probs = softmax(&logits);

        // All probabilities should be positive
        for &p in &probs {
            assert!(p > 0.0);
            assert!(p <= 1.0);
        }

        // Should sum to approximately 1
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 0.001);

        // Higher logits should have higher probability
        assert!(probs[2] > probs[1]);
        assert!(probs[1] > probs[0]);
    }

    #[test]
    fn test_softmax_numerical_stability() {
        // Very large logits that could cause overflow
        let large_logits = vec![1000.0, 1001.0, 1002.0];
        let probs = softmax(&large_logits);

        // Should still sum to 1
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 0.001);

        // All should be finite
        for &p in &probs {
            assert!(p.is_finite());
        }
    }

    #[test]
    fn test_sample_greedy() {
        let logits = vec![0.1, 0.5, 0.3, 0.8];
        let token = sample_greedy(&logits).unwrap();
        assert_eq!(token, 3); // Index of max value
    }

    #[test]
    fn test_sample_greedy_empty() {
        let result = sample_greedy(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_sample_temperature_deterministic() {
        let logits = vec![1.0, 2.0, 3.0];
        let config = TemperatureConfig::standard();

        // Same seed should give same result
        let token1 = sample_temperature(&logits, &config, 42).unwrap();
        let token2 = sample_temperature(&logits, &config, 42).unwrap();
        assert_eq!(token1, token2);
    }

    #[test]
    fn test_sample_temperature_empty() {
        let config = TemperatureConfig::standard();
        let result = sample_temperature(&[], &config, 42);
        assert!(result.is_err());
    }

    #[test]
    fn test_sample_top_k_basic() {
        let logits = vec![0.1, 0.5, 0.3, 0.8, 0.2];
        let config = TopKConfig {
            k: 2,
            temperature: 1.0,
        };

        let token = sample_top_k(&logits, &config, 42).unwrap();
        // Should be either index 3 (0.8) or index 1 (0.5)
        assert!(token == 3 || token == 1);
    }

    #[test]
    fn test_sample_top_k_k_larger_than_vocab() {
        let logits = vec![0.1, 0.5, 0.3];
        let config = TopKConfig {
            k: 100, // Larger than vocab size
            temperature: 1.0,
        };

        let result = sample_top_k(&logits, &config, 42);
        assert!(result.is_ok());
    }

    #[test]
    fn test_sample_top_k_k_zero() {
        let logits = vec![0.1, 0.5, 0.3];
        let config = TopKConfig {
            k: 0,
            temperature: 1.0,
        };

        let result = sample_top_k(&logits, &config, 42);
        assert!(result.is_err());
    }

    #[test]
    fn test_sample_top_p_basic() {
        let logits = vec![0.1, 0.5, 0.3, 0.8, 0.2];
        let config = TopPConfig {
            p: 0.95,
            temperature: 1.0,
        };

        let token = sample_top_p(&logits, &config, 42).unwrap();
        assert!(token < logits.len());
    }

    #[test]
    fn test_sample_top_p_p_boundary() {
        let logits = vec![1.0, 1.0, 1.0];

        let config_zero = TopPConfig {
            p: 0.0,
            temperature: 1.0,
        };
        let result_zero = sample_top_p(&logits, &config_zero, 42);
        assert!(result_zero.is_ok()); // Should include at least one token

        let config_one = TopPConfig {
            p: 1.0,
            temperature: 1.0,
        };
        let result_one = sample_top_p(&logits, &config_one, 42);
        assert!(result_one.is_ok()); // Should include all tokens
    }

    #[test]
    fn test_sample_top_p_invalid_p() {
        let logits = vec![0.1, 0.5, 0.3];
        let config_invalid = TopPConfig {
            p: 1.5, // Invalid
            temperature: 1.0,
        };

        let result = sample_top_p(&logits, &config_invalid, 42);
        assert!(result.is_err());
    }

    #[test]
    fn test_sample_top_k_vs_top_p() {
        let logits = vec![10.0, 8.0, 6.0, 1.0, 0.1];

        let topk = sample_top_k(
            &logits,
            &TopKConfig {
                k: 3,
                temperature: 1.0,
            },
            42,
        )
        .unwrap();

        let topp = sample_top_p(
            &logits,
            &TopPConfig {
                p: 0.95,
                temperature: 1.0,
            },
            42,
        )
        .unwrap();

        // Both should return valid indices
        assert!(topk < logits.len());
        assert!(topp < logits.len());
    }

    #[test]
    fn test_simple_random_deterministic() {
        let r1 = simple_random(42);
        let r2 = simple_random(42);
        assert_eq!(r1, r2);
    }

    #[test]
    fn test_simple_random_different_seeds() {
        let r1 = simple_random(42);
        let r2 = simple_random(43);
        assert_ne!(r1, r2);
    }

    #[test]
    fn test_top_k_config_standard() {
        let config = TopKConfig::standard();
        assert_eq!(config.k, 50);
        assert_eq!(config.temperature, 1.0);
    }

    #[test]
    fn test_top_p_config_standard() {
        let config = TopPConfig::standard();
        assert_eq!(config.p, 0.95);
        assert_eq!(config.temperature, 1.0);
    }
}
