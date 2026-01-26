use serde::{Deserialize, Serialize};

/// GPT-OSS 20B Model Configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GPTOSSConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub max_position_embeddings: usize,
    pub rope_theta: f32,
    pub initializer_range: f32,
    pub rms_norm_eps: f32,
    pub use_cache: bool,
}

impl Default for GPTOSSConfig {
    fn default() -> Self {
        Self {
            vocab_size: 201088,
            hidden_size: 2880,
            intermediate_size: 7168,
            num_hidden_layers: 24,
            num_attention_heads: 64,
            num_key_value_heads: 8,
            max_position_embeddings: 4096,
            rope_theta: 10000.0,
            initializer_range: 0.02,
            rms_norm_eps: 1e-6,
            use_cache: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = GPTOSSConfig::default();
        assert_eq!(config.vocab_size, 201088);
        assert_eq!(config.num_hidden_layers, 24);
        assert_eq!(config.hidden_size, 2880);
    }
}
