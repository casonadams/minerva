use super::llama_utils::silu;
/// Feed-Forward Network
///
/// Implements the feed-forward sub-layer of the transformer architecture.
/// Uses the architecture: hidden -> up_project -> activate (SiLU) -> down_project -> output
///
/// This expands the hidden representation to an intermediate size, applies a non-linear
/// activation, then projects back to the original hidden size.
use crate::error::{MinervaError, MinervaResult};

/// Parameters for FeedForward forward pass
pub struct FFParams<'a> {
    /// Input data
    pub x: &'a [f32],
    /// Up weight
    pub up_weight: &'a [f32],
    /// Down weight
    pub down_weight: &'a [f32],
}

/// Feed-forward network
pub struct FeedForward {
    hidden_size: usize,
    intermediate_size: usize,
}

impl FeedForward {
    /// Create new feed-forward layer
    ///
    /// # Arguments
    /// * `hidden_size` - Size of input/output
    /// * `intermediate_size` - Size of hidden layer
    ///
    /// # Example
    /// ```
    /// # use inference::llama_feedforward::FeedForward;
    /// let ff = FeedForward::new(512, 2048);
    /// assert_eq!(ff.hidden_size, 512);
    /// ```
    pub fn new(hidden_size: usize, intermediate_size: usize) -> Self {
        Self {
            hidden_size,
            intermediate_size,
        }
    }

    /// Forward pass: hidden -> up -> activate -> down -> hidden
    ///
    /// # Arguments
    /// * `params` - Feed-forward parameters including input and weights
    ///
    /// # Returns
    /// Output vector of size hidden_size
    ///
    /// # Errors
    /// Returns error if input size or weight dimensions don't match
    ///
    /// # Algorithm
    /// 1. Up projection: x @ up_weight (hidden_size -> intermediate_size)
    /// 2. Activation: Apply SiLU to expanded representation
    /// 3. Down projection: hidden @ down_weight (intermediate_size -> hidden_size)
    pub fn forward(&self, params: FFParams) -> MinervaResult<Vec<f32>> {
        if params.x.len() != self.hidden_size {
            return Err(MinervaError::InferenceError(format!(
                "Input size {} != hidden size {}",
                params.x.len(),
                self.hidden_size
            )));
        }

        if params.up_weight.len() != self.hidden_size * self.intermediate_size {
            return Err(MinervaError::InferenceError(
                "Up weight dimension mismatch".to_string(),
            ));
        }

        // Up projection: x @ up_weight
        let mut hidden = vec![0.0; self.intermediate_size];
        for (i, h) in hidden.iter_mut().enumerate() {
            for (j, &x) in params.x.iter().enumerate() {
                *h += x * params.up_weight[i * self.hidden_size + j];
            }
        }

        // Apply SiLU activation
        hidden = silu(&hidden);

        // Down projection: hidden @ down_weight
        let mut output = vec![0.0; self.hidden_size];
        if params.down_weight.len() == self.intermediate_size * self.hidden_size {
            for (i, o) in output.iter_mut().enumerate() {
                for (j, &h) in hidden.iter().enumerate() {
                    *o += h * params.down_weight[j * self.hidden_size + i];
                }
            }
        }

        Ok(output)
    }
}
