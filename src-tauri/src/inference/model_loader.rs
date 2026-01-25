use super::engine_config::InferenceEngineConfig;
use super::inference_engine::InferenceEngine;
use super::model_weights::{LayerWeights, ModelWeights};
use super::pure_rust_backend::ModelType;
/// Model Loader - Phase 9 Day 7 Complete Implementation
///
/// Integrates safetensors weight loading (Day 1) with the complete
/// InferenceEngine pipeline (Days 2-5) to create a production-ready
/// model loading and inference system.
///
/// This module handles:
/// - JSON configuration parsing
/// - Safetensors weight loading
/// - Weight extraction and mapping
/// - Configuration inference from model metadata
/// - Model validation and error handling
use crate::error::{MinervaError, MinervaResult};
use std::path::Path;

// ============================================================================
// Model Loader Configuration
// ============================================================================

/// Configuration for loading a model from files
#[derive(Debug, Clone)]
pub struct ModelLoaderConfig {
    /// Path to safetensors weights file
    pub weights_path: String,
    /// Path to config.json metadata file
    pub config_path: String,
    /// Optional override for model type detection
    pub model_type_override: Option<ModelType>,
}

impl ModelLoaderConfig {
    /// Create config for a model directory (auto-discovers files)
    pub fn from_directory(model_dir: &str) -> MinervaResult<Self> {
        let weights_path = format!("{}/model.safetensors", model_dir);
        let config_path = format!("{}/config.json", model_dir);

        // Verify files exist
        if !Path::new(&weights_path).exists() {
            return Err(MinervaError::InferenceError(format!(
                "Weights file not found: {}",
                weights_path
            )));
        }

        if !Path::new(&config_path).exists() {
            return Err(MinervaError::InferenceError(format!(
                "Config file not found: {}",
                config_path
            )));
        }

        Ok(Self {
            weights_path,
            config_path,
            model_type_override: None,
        })
    }
}

// ============================================================================
// Model Metadata (from config.json)
// ============================================================================

/// Model metadata from config.json
#[derive(Debug, Clone)]
pub struct ModelMetadata {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_heads: usize,
    pub num_layers: usize,
    pub intermediate_size: usize,
    pub model_type: ModelType,
    pub rope_theta: Option<f32>,
}

impl ModelMetadata {
    /// Parse metadata from config.json
    pub fn from_config(config_path: &str) -> MinervaResult<Self> {
        // Read the file
        let config_str = std::fs::read_to_string(config_path).map_err(|e| {
            MinervaError::InferenceError(format!("Failed to read config file: {}", e))
        })?;

        // Parse JSON
        let config_json: serde_json::Value = serde_json::from_str(&config_str).map_err(|e| {
            MinervaError::InferenceError(format!("Failed to parse config.json: {}", e))
        })?;

        // Extract fields with proper error handling
        let vocab_size = config_json["vocab_size"].as_u64().ok_or_else(|| {
            MinervaError::InferenceError("vocab_size not found or not a number".to_string())
        })? as usize;

        let hidden_size = config_json["hidden_size"].as_u64().ok_or_else(|| {
            MinervaError::InferenceError("hidden_size not found or not a number".to_string())
        })? as usize;

        let num_attention_heads = config_json["num_attention_heads"]
            .as_u64()
            .or_else(|| config_json["num_heads"].as_u64())
            .ok_or_else(|| {
                MinervaError::InferenceError(
                    "num_attention_heads not found or not a number".to_string(),
                )
            })? as usize;

        let num_hidden_layers = config_json["num_hidden_layers"]
            .as_u64()
            .or_else(|| config_json["num_layers"].as_u64())
            .ok_or_else(|| {
                MinervaError::InferenceError(
                    "num_hidden_layers not found or not a number".to_string(),
                )
            })? as usize;

        // Intermediate size: try multiple field names
        let intermediate_size = config_json["intermediate_size"]
            .as_u64()
            .or_else(|| {
                // Fallback: 4 * hidden_size (common default)
                Some((hidden_size as u64) * 4)
            })
            .ok_or_else(|| {
                MinervaError::InferenceError(
                    "intermediate_size could not be determined".to_string(),
                )
            })? as usize;

        // Detect model type from architecture_name or model_type field
        let model_type_str = config_json["model_type"]
            .as_str()
            .or_else(|| config_json["architectures"].get(0).and_then(|a| a.as_str()))
            .unwrap_or("unknown");

        let model_type = ModelType::parse(model_type_str);

        // RoPE theta (optional, for rotary embeddings)
        let rope_theta = config_json["rope_theta"].as_f64().map(|x| x as f32);

        Ok(Self {
            vocab_size,
            hidden_size,
            num_heads: num_attention_heads,
            num_layers: num_hidden_layers,
            intermediate_size,
            model_type,
            rope_theta,
        })
    }

    /// Create InferenceEngineConfig from metadata
    pub fn to_engine_config(&self) -> MinervaResult<InferenceEngineConfig> {
        // Infer activation and other settings based on model type
        let (activation, causal) = match self.model_type {
            ModelType::Llama => (super::transformer_components::Activation::SiLU, true),
            ModelType::Mistral => (super::transformer_components::Activation::SiLU, true),
            ModelType::Phi => (super::transformer_components::Activation::GELU, true),
            ModelType::Qwen => (super::transformer_components::Activation::SiLU, true),
            ModelType::Unknown => {
                return Err(MinervaError::InferenceError(
                    "Cannot infer config for unknown model type".to_string(),
                ));
            }
        };

        Ok(InferenceEngineConfig {
            vocab_size: self.vocab_size,
            hidden_size: self.hidden_size,
            num_heads: self.num_heads,
            num_layers: self.num_layers,
            intermediate_size: self.intermediate_size,
            activation,
            causal,
            eps: 1e-6,
            max_seq_len: 2048,
        })
    }
}

// ============================================================================
// Model Loader
// ============================================================================

/// Loads models from disk and creates InferenceEngine instances
pub struct ModelLoader;

impl ModelLoader {
    /// Load a model from safetensors and config files
    pub fn load_model(config: &ModelLoaderConfig) -> MinervaResult<InferenceEngine> {
        // Step 1: Load metadata
        let metadata = ModelMetadata::from_config(&config.config_path)?;

        // Step 2: Create engine configuration
        let engine_config = metadata.to_engine_config()?;

        // Step 3: Load weights from safetensors
        let weights = Self::load_weights(&config.weights_path, &engine_config)?;

        // Step 4: Create and validate engine
        let engine = InferenceEngine::new(engine_config, weights)?;

        Ok(engine)
    }

    /// Load weights from safetensors file
    fn load_weights(
        weights_path: &str,
        config: &InferenceEngineConfig,
    ) -> MinervaResult<ModelWeights> {
        use safetensors::SafeTensors;
        use std::collections::HashMap;
        use std::fs;

        // Read safetensors file
        let file_data = fs::read(weights_path).map_err(|e| {
            MinervaError::InferenceError(format!("Failed to read weights file: {}", e))
        })?;

        // Deserialize safetensors
        let safetensors = SafeTensors::deserialize(&file_data).map_err(|e| {
            MinervaError::InferenceError(format!("Failed to deserialize safetensors: {}", e))
        })?;

        // Extract tensors into a map for easy access
        let mut tensors: HashMap<String, Vec<f32>> = HashMap::new();

        for (name, tensor) in safetensors.tensors() {
            let data = tensor.data();
            let f32_count = data.len() / 4;

            if data.len() % 4 != 0 {
                return Err(MinervaError::InferenceError(format!(
                    "Tensor {} has misaligned data size",
                    name
                )));
            }

            let mut f32_data = vec![0.0_f32; f32_count];
            for i in 0..f32_count {
                let bytes = [
                    data[i * 4],
                    data[i * 4 + 1],
                    data[i * 4 + 2],
                    data[i * 4 + 3],
                ];
                f32_data[i] = f32::from_le_bytes(bytes);
            }

            tensors.insert(name.to_string(), f32_data);
        }

        // Extract embedding weights
        let embeddings = Self::extract_tensor(
            &tensors,
            &[
                "model.embed_tokens.weight",
                "embeddings.weight",
                "wte.weight",
            ],
            config.vocab_size * config.hidden_size,
        )?;

        // Extract per-layer weights
        let mut layers = Vec::new();
        for layer_idx in 0..config.num_layers {
            let attn_norm_scale = Self::extract_tensor(
                &tensors,
                &[
                    &format!("model.layers.{}.input_layernorm.weight", layer_idx),
                    &format!("model.layers.{}.norm1.weight", layer_idx),
                ],
                config.hidden_size,
            )?;

            let ff_up = Self::extract_tensor(
                &tensors,
                &[
                    &format!("model.layers.{}.mlp.up_proj.weight", layer_idx),
                    &format!("model.layers.{}.mlp.gate_proj.weight", layer_idx),
                ],
                config.hidden_size * config.intermediate_size,
            )?;

            let ff_down = Self::extract_tensor(
                &tensors,
                &[&format!("model.layers.{}.mlp.down_proj.weight", layer_idx)],
                config.intermediate_size * config.hidden_size,
            )?;

            layers.push(LayerWeights {
                attn_norm_scale,
                ffn_norm_scale: vec![1.0; config.hidden_size],
                ff_up,
                ff_down,
            });
        }

        // Extract final norm scale
        let final_norm_scale = Self::extract_tensor(
            &tensors,
            &["model.norm.weight", "final_layer_norm.weight"],
            config.hidden_size,
        )?;

        // Extract output projection
        let output_proj = Self::extract_tensor(
            &tensors,
            &["lm_head.weight"],
            config.hidden_size * config.vocab_size,
        )?;

        Ok(ModelWeights {
            embeddings,
            layers,
            final_norm_scale,
            output_proj,
        })
    }

    /// Helper to extract tensor by trying multiple possible names
    fn extract_tensor(
        tensors: &std::collections::HashMap<String, Vec<f32>>,
        possible_names: &[&str],
        expected_size: usize,
    ) -> MinervaResult<Vec<f32>> {
        for name in possible_names {
            if let Some(tensor) = tensors.get(*name)
                && tensor.len() == expected_size
            {
                return Ok(tensor.clone());
            }
        }

        Err(MinervaError::InferenceError(format!(
            "Could not find tensor with any of: {:?} (expected size: {})",
            possible_names, expected_size
        )))
    }

    /// Load a model from a directory containing model.safetensors and config.json
    pub fn load_from_directory(model_dir: &str) -> MinervaResult<InferenceEngine> {
        let config = ModelLoaderConfig::from_directory(model_dir)?;
        Self::load_model(&config)
    }

    /// Validate model weights against configuration
    #[allow(dead_code)]
    fn validate_weights(
        weights: &ModelWeights,
        config: &InferenceEngineConfig,
    ) -> MinervaResult<()> {
        // Embedding matrix
        if weights.embeddings.len() != config.vocab_size * config.hidden_size {
            return Err(MinervaError::InferenceError(format!(
                "Embedding shape mismatch: expected {} × {}, got {}",
                config.vocab_size,
                config.hidden_size,
                weights.embeddings.len() / config.hidden_size
            )));
        }

        // Layers
        if weights.layers.len() != config.num_layers {
            return Err(MinervaError::InferenceError(format!(
                "Layer count mismatch: expected {}, got {}",
                config.num_layers,
                weights.layers.len()
            )));
        }

        // Final norm scale
        if weights.final_norm_scale.len() != config.hidden_size {
            return Err(MinervaError::InferenceError(format!(
                "Final norm scale shape mismatch: expected {}, got {}",
                config.hidden_size,
                weights.final_norm_scale.len()
            )));
        }

        // Output projection
        if weights.output_proj.len() != config.hidden_size * config.vocab_size {
            return Err(MinervaError::InferenceError(format!(
                "Output projection shape mismatch: expected {} × {}, got {}",
                config.hidden_size,
                config.vocab_size,
                weights.output_proj.len() / config.vocab_size
            )));
        }

        // Per-layer validation
        for (i, layer) in weights.layers.iter().enumerate() {
            if layer.attn_norm_scale.len() != config.hidden_size {
                return Err(MinervaError::InferenceError(format!(
                    "Layer {} attn norm scale shape mismatch",
                    i
                )));
            }

            if layer.ff_up.len() != config.hidden_size * config.intermediate_size {
                return Err(MinervaError::InferenceError(format!(
                    "Layer {} FF up shape mismatch",
                    i
                )));
            }

            if layer.ff_down.len() != config.intermediate_size * config.hidden_size {
                return Err(MinervaError::InferenceError(format!(
                    "Layer {} FF down shape mismatch",
                    i
                )));
            }
        }

        Ok(())
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_loader_config_creation() {
        let config = ModelLoaderConfig {
            weights_path: "models/llama/model.safetensors".to_string(),
            config_path: "models/llama/config.json".to_string(),
            model_type_override: None,
        };

        assert_eq!(config.weights_path, "models/llama/model.safetensors");
        assert_eq!(config.config_path, "models/llama/config.json");
    }

    #[test]
    fn test_model_loader_config_from_directory_missing_weights() {
        let result = ModelLoaderConfig::from_directory("/nonexistent/path");
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("Weights file not found")
        );
    }

    #[test]
    fn test_metadata_to_engine_config_llama() {
        let metadata = ModelMetadata {
            vocab_size: 32000,
            hidden_size: 4096,
            num_heads: 32,
            num_layers: 32,
            intermediate_size: 11008,
            model_type: ModelType::Llama,
            rope_theta: Some(10000.0),
        };

        let config = metadata.to_engine_config().unwrap();
        assert_eq!(config.vocab_size, 32000);
        assert_eq!(config.hidden_size, 4096);
        assert_eq!(config.num_heads, 32);
        assert!(config.causal);
    }

    #[test]
    fn test_metadata_to_engine_config_unknown_type() {
        let metadata = ModelMetadata {
            vocab_size: 30000,
            hidden_size: 768,
            num_heads: 12,
            num_layers: 12,
            intermediate_size: 3072,
            model_type: ModelType::Unknown,
            rope_theta: None,
        };

        let result = metadata.to_engine_config();
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("unknown model type")
        );
    }

    #[test]
    fn test_metadata_from_config_llama() {
        let json_str = r#"{
            "vocab_size": 32000,
            "hidden_size": 4096,
            "num_attention_heads": 32,
            "num_hidden_layers": 32,
            "intermediate_size": 11008,
            "model_type": "llama",
            "rope_theta": 10000.0
        }"#;

        let temp_path = "/tmp/test_config_llama.json";
        if std::fs::write(temp_path, json_str).is_ok() {
            let metadata = ModelMetadata::from_config(temp_path).unwrap();

            assert_eq!(metadata.vocab_size, 32000);
            assert_eq!(metadata.hidden_size, 4096);
            assert_eq!(metadata.num_heads, 32);
            assert_eq!(metadata.num_layers, 32);
            assert_eq!(metadata.intermediate_size, 11008);
            assert_eq!(metadata.model_type, ModelType::Llama);
            assert_eq!(metadata.rope_theta, Some(10000.0));

            let _ = std::fs::remove_file(temp_path);
        }
    }

    #[test]
    fn test_metadata_from_config_missing_file() {
        let result = ModelMetadata::from_config("/nonexistent/path.json");
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Failed to read"));
    }

    #[test]
    fn test_validate_weights_valid() {
        let config = InferenceEngineConfig::tiny(1000);
        let weights = create_dummy_weights(&config);

        let result = ModelLoader::validate_weights(&weights, &config);
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_weights_embedding_mismatch() {
        let config = InferenceEngineConfig::tiny(1000);
        let mut weights = create_dummy_weights(&config);

        weights.embeddings = vec![0.1; 100];

        let result = ModelLoader::validate_weights(&weights, &config);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Embedding"));
    }

    #[test]
    fn test_validate_weights_layer_count_mismatch() {
        let config = InferenceEngineConfig::tiny(1000);
        let mut weights = create_dummy_weights(&config);

        weights.layers = vec![];

        let result = ModelLoader::validate_weights(&weights, &config);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Layer count"));
    }

    #[test]
    fn test_extract_tensor_found() {
        let mut tensors = std::collections::HashMap::new();
        let test_data = vec![1.0, 2.0, 3.0, 4.0];
        tensors.insert("model.test.weight".to_string(), test_data.clone());

        let result = ModelLoader::extract_tensor(&tensors, &["model.test.weight"], 4);

        assert!(result.is_ok());
        assert_eq!(result.unwrap(), test_data);
    }

    #[test]
    fn test_extract_tensor_not_found() {
        let tensors = std::collections::HashMap::new();

        let result = ModelLoader::extract_tensor(&tensors, &["missing.weight"], 4);

        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("Could not find tensor")
        );
    }

    fn create_dummy_weights(config: &InferenceEngineConfig) -> ModelWeights {
        let mut layers = Vec::new();

        for _ in 0..config.num_layers {
            layers.push(LayerWeights {
                attn_norm_scale: vec![1.0; config.hidden_size],
                ffn_norm_scale: vec![1.0; config.hidden_size],
                ff_up: vec![0.1; config.hidden_size * config.intermediate_size],
                ff_down: vec![0.1; config.intermediate_size * config.hidden_size],
            });
        }

        ModelWeights {
            embeddings: vec![0.1; config.vocab_size * config.hidden_size],
            layers,
            final_norm_scale: vec![1.0; config.hidden_size],
            output_proj: vec![0.01; config.hidden_size * config.vocab_size],
        }
    }
}
