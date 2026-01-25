/// Unified format loader abstraction
///
/// Provides a common interface for loading models in different formats:
/// - GGUF (llama.cpp quantized)
/// - SafeTensors (standard huggingface format)
/// - MLX (Apple Silicon optimized)
use crate::error::MinervaResult;
use ndarray::Array2;
use std::path::Path;

/// Format type enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelFormat {
    GGUF,
    SafeTensors,
    MLX,
}

impl ModelFormat {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::GGUF => "GGUF",
            Self::SafeTensors => "SafeTensors",
            Self::MLX => "MLX",
        }
    }
}

/// Model loading result with metadata
#[derive(Debug, Clone)]
pub struct LoadResult {
    pub format: ModelFormat,
    pub load_time_ms: u64,
    pub memory_bytes: usize,
    pub num_tensors: usize,
}

/// Unified loader trait - all formats implement this
pub trait FormatLoader: Send + Sync {
    /// Load model from path
    fn load(&self, path: &Path) -> MinervaResult<LoadedModel>;

    /// Get format name
    fn format(&self) -> ModelFormat;

    /// Detect if path is this format
    fn detect(&self, path: &Path) -> bool;
}

/// Loaded model with format-agnostic interface
pub struct LoadedModel {
    pub format: ModelFormat,
    pub config: ModelConfig,
    pub weights: ModelWeights,
    pub metadata: LoadMetadata,
}

/// Configuration that works across all formats
#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub model_name: String,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub num_attention_heads: usize,
    pub num_kv_heads: Option<usize>,
    pub vocab_size: usize,
    pub intermediate_size: usize,
    pub max_sequence_length: usize,
    pub architectures: Vec<String>,
}

/// Model weights abstraction
pub struct ModelWeights {
    pub embedding: Array2<f32>,
    pub lm_head: Array2<f32>,
    pub layers: Vec<TransformerLayer>,
    pub final_norm: Array2<f32>,
}

/// Single transformer layer weights
pub struct TransformerLayer {
    pub attn_norm: Array2<f32>,
    pub attn_q: Array2<f32>,
    pub attn_k: Array2<f32>,
    pub attn_v: Array2<f32>,
    pub attn_o: Array2<f32>,

    pub ffn_norm: Array2<f32>,
    pub ffn_gate: Array2<f32>,
    pub ffn_up: Array2<f32>,
    pub ffn_down: Array2<f32>,
}

/// Metadata about the load
#[derive(Debug, Clone)]
pub struct LoadMetadata {
    pub format: ModelFormat,
    pub load_time_ms: u64,
    pub memory_bytes: usize,
    pub num_tensors: usize,
    pub quantization: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_format_str() {
        assert_eq!(ModelFormat::GGUF.as_str(), "GGUF");
        assert_eq!(ModelFormat::SafeTensors.as_str(), "SafeTensors");
        assert_eq!(ModelFormat::MLX.as_str(), "MLX");
    }

    #[test]
    fn test_model_config_creation() {
        let config = ModelConfig {
            model_name: "test".to_string(),
            hidden_size: 2048,
            num_layers: 22,
            num_attention_heads: 32,
            num_kv_heads: Some(8),
            vocab_size: 32000,
            intermediate_size: 5632,
            max_sequence_length: 4096,
            architectures: vec!["LlamaForCausalLM".to_string()],
        };

        assert_eq!(config.hidden_size, 2048);
        assert_eq!(config.num_layers, 22);
        assert_eq!(config.num_kv_heads, Some(8));
    }
}
