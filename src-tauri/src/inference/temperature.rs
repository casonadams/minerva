/// Temperature Scaling for Token Sampling
///
/// Temperature controls the sharpness of probability distributions before sampling.
/// Lower values make the distribution sharper (less diverse), higher values make it
/// softer (more diverse).
///
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

/// Apply temperature scaling to logits
///
/// Temperature scales the logits before softmax:
/// - Lower temperature: sharper probability distribution
/// - Higher temperature: softer probability distribution
pub fn apply_temperature(logits: &[f32], temperature: f32) -> Vec<f32> {
    if (temperature - 1.0).abs() < 0.001 {
        // Skip scaling if temperature is 1.0
        logits.to_vec()
    } else {
        logits.iter().map(|&x| x / temperature).collect()
    }
}
