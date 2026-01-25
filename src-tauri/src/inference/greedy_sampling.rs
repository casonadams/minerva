use super::softmax_utils::argmax;
/// Greedy Token Sampling
///
/// Selects the token with the highest probability deterministically.
/// Useful for tasks where consistency and factuality are more important
/// than diversity.
use crate::error::{MinervaError, MinervaResult};

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

    argmax(logits).ok_or_else(|| MinervaError::InferenceError("No valid tokens".to_string()))
}
