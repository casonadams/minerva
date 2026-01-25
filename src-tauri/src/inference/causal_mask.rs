/// Causal Masking for Autoregressive Transformer Networks
///
/// Implements causal masks that prevent the model from attending to future tokens,
/// essential for autoregressive generation.
///
/// Causal mask for autoregressive attention
///
/// Creates a mask that prevents the model from attending to future tokens.
/// Used during generation to ensure each token can only see past tokens.
///
/// # Returns
///
/// Boolean mask of shape (seq_len, seq_len) where:
/// - true = attend (past or current)
/// - false = masked (future)
pub fn create_causal_mask(seq_len: usize) -> Vec<bool> {
    let mut mask = vec![false; seq_len * seq_len];

    for i in 0..seq_len {
        for j in 0..seq_len {
            // Can attend to current and past positions
            if j <= i {
                mask[i * seq_len + j] = true;
            }
        }
    }

    mask
}
