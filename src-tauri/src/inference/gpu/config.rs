use crate::error::{MinervaError, MinervaResult};
/// Model configuration structures
///
/// Loads model configuration from config.json files and validates
/// that the model architecture is supported.
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub architectures: Vec<String>,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: Option<usize>,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub hidden_act: String,
    pub rms_norm_eps: f32,
    pub rope_theta: Option<f32>,
    pub attention_dropout: Option<f32>,
    pub layer_norm_eps: Option<f32>,
}

impl ModelConfig {
    /// Load configuration from JSON file
    pub fn from_file(path: &Path) -> MinervaResult<Self> {
        let json_str = std::fs::read_to_string(path)?;
        let config: ModelConfig = serde_json::from_str(&json_str)?;
        config.validate()?;
        Ok(config)
    }

    /// Validate configuration
    fn validate(&self) -> MinervaResult<()> {
        // Check required fields
        if self.hidden_size == 0 {
            return Err(MinervaError::ValidationError(
                "hidden_size must be > 0".to_string(),
            ));
        }
        if self.num_hidden_layers == 0 {
            return Err(MinervaError::ValidationError(
                "num_hidden_layers must be > 0".to_string(),
            ));
        }
        if self.num_attention_heads == 0 {
            return Err(MinervaError::ValidationError(
                "num_attention_heads must be > 0".to_string(),
            ));
        }
        if self.vocab_size == 0 {
            return Err(MinervaError::ValidationError(
                "vocab_size must be > 0".to_string(),
            ));
        }

        // Check architecture is supported
        let supported = vec!["LlamaForCausalLM", "MistralForCausalLM", "PhiForCausalLM"];
        if !supported
            .iter()
            .any(|arch| self.architectures.contains(&arch.to_string()))
        {
            eprintln!(
                "Warning: Model architecture may not be fully supported. Found: {:?}",
                self.architectures
            );
        }

        Ok(())
    }

    /// Get head dimension
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    /// Get number of KV heads (for multi-query attention)
    pub fn num_kv_heads(&self) -> usize {
        self.num_key_value_heads.unwrap_or(self.num_attention_heads)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_config_head_dim() {
        let config = ModelConfig {
            architectures: vec!["LlamaForCausalLM".to_string()],
            hidden_size: 4096,
            intermediate_size: 11008,
            num_hidden_layers: 32,
            num_attention_heads: 32,
            num_key_value_heads: None,
            vocab_size: 32000,
            max_position_embeddings: 2048,
            hidden_act: "silu".to_string(),
            rms_norm_eps: 1e-5,
            rope_theta: Some(10000.0),
            attention_dropout: None,
            layer_norm_eps: None,
        };

        assert_eq!(config.head_dim(), 128);
        assert_eq!(config.num_kv_heads(), 32);
    }

    #[test]
    fn test_model_config_validation() {
        let mut config = ModelConfig {
            architectures: vec!["LlamaForCausalLM".to_string()],
            hidden_size: 4096,
            intermediate_size: 11008,
            num_hidden_layers: 32,
            num_attention_heads: 32,
            num_key_value_heads: None,
            vocab_size: 32000,
            max_position_embeddings: 2048,
            hidden_act: "silu".to_string(),
            rms_norm_eps: 1e-5,
            rope_theta: Some(10000.0),
            attention_dropout: None,
            layer_norm_eps: None,
        };

        assert!(config.validate().is_ok());

        // Test invalid config
        config.hidden_size = 0;
        assert!(config.validate().is_err());
    }
}
