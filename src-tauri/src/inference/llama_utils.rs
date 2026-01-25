/// LLaMA Utility Functions
///
/// Core mathematical operations used in the LLaMA inference pipeline:
/// - RMSNorm: Layer normalization using root mean square
/// - SiLU: Sigmoid Linear Unit activation function
use crate::error::{MinervaError, MinervaResult};

/// Compute RMSNorm (Root Mean Square Layer Normalization)
///
/// RMSNorm is a simplification of LayerNorm that uses only the root mean square
/// for normalization: RMSNorm(x) = (x / RMS(x)) * weight
///
/// # Arguments
/// * `x` - Input vector
/// * `weight` - Scaling weights (affine parameters)
/// * `eps` - Small epsilon value for numerical stability
///
/// # Returns
/// Normalized and scaled output vector
///
/// # Errors
/// Returns error if input and weight vectors have different lengths
///
/// # Example
/// ```ignore
/// let x = vec![1.0, 2.0, 3.0, 4.0];
/// let weight = vec![0.5, 0.5, 0.5, 0.5];
/// let result = rmsnorm(&x, &weight, 1e-6).unwrap();
/// assert_eq!(result.len(), 4);
/// ```
pub fn rmsnorm(x: &[f32], weight: &[f32], eps: f32) -> MinervaResult<Vec<f32>> {
    if x.len() != weight.len() {
        return Err(MinervaError::InferenceError(format!(
            "Input size {} != weight size {}",
            x.len(),
            weight.len()
        )));
    }

    let rms = (x.iter().map(|v| v * v).sum::<f32>() / (x.len() as f32) + eps).sqrt();
    Ok(x.iter().zip(weight).map(|(a, b)| (a / rms) * b).collect())
}

/// Compute SiLU activation (Sigmoid Linear Unit, also known as Swish)
///
/// SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
///
/// This activation function combines linear and sigmoid components, providing
/// smooth gradients and good performance in deep networks.
///
/// # Arguments
/// * `x` - Input vector
///
/// # Returns
/// Activated output vector
///
/// # Example
/// ```ignore
/// let x = vec![0.0, 1.0, -1.0, 2.0];
/// let result = silu(&x);
/// assert_eq!(result.len(), 4);
/// ```
pub fn silu(x: &[f32]) -> Vec<f32> {
    x.iter().map(|v| v / (1.0 + (-v).exp())).collect()
}
