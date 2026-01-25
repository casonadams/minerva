/// Activation Functions for Transformer Networks
///
/// Provides GELU, SiLU, and ReLU activation functions used in transformer
/// architectures. These functions are applied element-wise to intermediate
/// layer outputs.
///
/// GELU activation function
///
/// Gaussian Error Linear Unit: a smooth approximation of ReLU
/// Formula: x * Φ(x) where Φ is the cumulative normal distribution
/// Approximation: x * 0.5 * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
///
/// Used in most modern transformers (BERT, GPT, LLaMA)
#[inline]
pub fn gelu(x: f32) -> f32 {
    const COEFF: f32 = 0.044_715;
    const SQRT_2_PI: f32 = 0.797_885; // sqrt(2/π)

    let x_cube = x * x * x;
    let inner = SQRT_2_PI * (x + COEFF * x_cube);
    x * 0.5 * (1.0 + inner.tanh())
}

/// SiLU (Swish) activation function
///
/// Self-Gated Linear Unit: x * sigmoid(x)
/// Used in modern models like LLaMA
#[inline]
pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// ReLU activation function
///
/// Rectified Linear Unit: max(0, x)
/// Classic activation, still used in some FFN layers
#[inline]
pub fn relu(x: f32) -> f32 {
    x.max(0.0)
}

/// Activation function selector
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Activation {
    GELU,
    SiLU,
    ReLU,
}

/// Apply activation function to entire vector
pub fn apply_activation(input: &[f32], activation: Activation) -> Vec<f32> {
    match activation {
        Activation::GELU => input.iter().map(|&x| gelu(x)).collect(),
        Activation::SiLU => input.iter().map(|&x| silu(x)).collect(),
        Activation::ReLU => input.iter().map(|&x| relu(x)).collect(),
    }
}
