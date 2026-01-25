/// Rotary Positional Embeddings (RoPE) Implementation
///
/// RoPE applies a rotation matrix to query and key vectors based on their position
/// in the sequence. This allows the model to incorporate positional information
/// while maintaining relative position awareness.
///
/// The rotation angle is calculated as: θⱼ = base^(-2j/d)
/// where base=10000, j is the dimension index, and d is the head dimension.
/// Rotary positional embeddings parameters
#[derive(Debug, Clone, Copy)]
pub struct RoPEParams {
    /// Head dimension (d in the formula)
    head_dim: usize,
    /// Theta base for rotation (typically 10,000 in LLaMA)
    theta_base: f32,
}

impl RoPEParams {
    /// Create new RoPE parameters with given head dimension
    ///
    /// # Arguments
    /// * `head_dim` - The dimension of each attention head
    ///
    /// # Example
    /// ```
    /// # use inference::rope_utils::RoPEParams;
    /// let rope = RoPEParams::new(64);
    /// assert_eq!(rope.head_dim, 64);
    /// ```
    pub fn new(head_dim: usize) -> Self {
        Self {
            head_dim,
            theta_base: 10_000.0,
        }
    }

    /// Calculate rotary angle for position and dimension
    ///
    /// Computes θⱼ,ₘ = m * θⱼ where θⱼ = base^(-2j/d)
    /// and m is the position in the sequence
    ///
    /// # Arguments
    /// * `pos` - Position in the sequence
    /// * `dim` - Dimension index within the head
    pub fn get_angle(&self, pos: usize, dim: usize) -> f32 {
        let freq = self
            .theta_base
            .powf(-2.0 * (dim as f32) / (self.head_dim as f32));
        (pos as f32) * freq
    }
}
