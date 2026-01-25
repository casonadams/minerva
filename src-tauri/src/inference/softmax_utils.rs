/// Softmax and Random Number Generation Utilities
///
/// Provides numerically stable softmax computation and simple random number
/// generation for token sampling.
use std::cmp::Ordering;

/// Compute softmax probabilities from logits
pub fn softmax(logits: &[f32]) -> Vec<f32> {
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

/// Simple linear congruential generator (LCG) for random numbers
pub fn simple_random(seed: u64) -> u64 {
    let a: u64 = 1103515245;
    let c: u64 = 12345;
    let m: u64 = 2u64.pow(31);
    (a.wrapping_mul(seed).wrapping_add(c)) % m
}

/// Find the index of maximum value in slice
pub fn argmax(logits: &[f32]) -> Option<usize> {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Equal))
        .map(|(idx, _)| idx)
}
