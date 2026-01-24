/// Phase 10 Day 1b: MLX Model Type Support
///
/// Extends the model type system to include MLX-compatible models,
/// enabling the inference system to automatically detect and load models
/// optimized for Apple Silicon via MLX framework.
///
/// This module:
/// - Detects MLX-compatible models from HuggingFace Hub
/// - Provides model metadata for MLX models
/// - Integrates with MlxBackend for inference
/// - Supports automatic model type detection
///
/// # Supported MLX Models
///
/// MLX framework supports any HuggingFace transformer model, including:
/// - LLaMA, LLaMA-2, LLaMA-3 variants
/// - Mistral, Mixtral models
/// - Phi-1, Phi-2, Phi-3 (Microsoft)
/// - Qwen models (Alibaba)
/// - Gemma (Google)
/// - And any other HuggingFace text model
///
/// # Usage
///
/// ```rust,ignore
/// use crate::inference::mlx_model_support::{MlxModelInfo, detect_mlx_model};
///
/// // Detect if a model can be loaded with MLX
/// let model_info = detect_mlx_model("meta-llama/Llama-2-7b")?;
/// assert!(model_info.supports_mlx);
///
/// // Get MLX-specific configuration
/// let config = model_info.get_mlx_config()?;
/// ```
use crate::error::{MinervaError, MinervaResult};
use crate::inference::pure_rust_backend::ModelType;
use serde::{Deserialize, Serialize};

// ============================================================================
// MLX Model Types
// ============================================================================

/// MLX-specific model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MlxModelConfig {
    /// HuggingFace model ID (e.g., "meta-llama/Llama-2-7b")
    pub model_id: String,
    /// Base model type as string (for inference strategy)
    pub base_type: String,
    /// Whether the model is quantized
    pub is_quantized: bool,
    /// Quantization format (e.g., "int4", "int8", "float16")
    pub quantization_format: Option<String>,
    /// Parameters in billions
    pub param_count_billions: f32,
    /// Recommended context length
    pub context_length: usize,
    /// Whether model is verified to work with MLX
    pub mlx_verified: bool,
    /// MLX-specific optimizations available
    pub mlx_features: Vec<String>,
}

/// Information about an MLX-compatible model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MlxModelInfo {
    /// Model identifier
    pub model_id: String,
    /// Human-readable name
    pub name: String,
    /// Base architecture (Llama, Mistral, Phi, etc.)
    pub architecture: String,
    /// Model size category (tiny, small, medium, large, xlarge)
    pub size: String,
    /// Whether this model supports MLX inference
    pub supports_mlx: bool,
    /// MLX configuration if available
    pub mlx_config: Option<MlxModelConfig>,
    /// Whether model has official MLX weights
    pub has_mlx_weights: bool,
    /// Download size in GB
    pub size_gb: f32,
    /// Recommended RAM in GB
    pub recommended_ram_gb: f32,
}

impl MlxModelInfo {
    /// Get MLX configuration, creating one if not present
    pub fn get_mlx_config(&mut self) -> MinervaResult<&MlxModelConfig> {
        if self.mlx_config.is_none() {
            self.mlx_config = Some(create_mlx_config(&self.model_id)?);
        }
        self.mlx_config
            .as_ref()
            .ok_or_else(|| MinervaError::InferenceError("Failed to create MLX config".to_string()))
    }
}

// ============================================================================
// MLX Model Detection
// ============================================================================

/// Common MLX-compatible models from HuggingFace
const MLX_MODEL_REGISTRY: &[(&str, &str, &str)] = &[
    // Meta LLaMA
    ("meta-llama/Llama-2-7b", "Llama 2 7B", "llama"),
    ("meta-llama/Llama-2-13b", "Llama 2 13B", "llama"),
    ("meta-llama/Llama-2-70b", "Llama 2 70B", "llama"),
    ("meta-llama/Llama-3-8b", "Llama 3 8B", "llama"),
    ("meta-llama/Llama-3-70b", "Llama 3 70B", "llama"),
    // Mistral
    ("mistralai/Mistral-7B", "Mistral 7B", "mistral"),
    (
        "mistralai/Mistral-7B-Instruct-v0.1",
        "Mistral 7B Instruct",
        "mistral",
    ),
    (
        "mistralai/Mistral-7B-Instruct-v0.2",
        "Mistral 7B Instruct v0.2",
        "mistral",
    ),
    ("mistralai/Mixtral-8x7B", "Mixtral 8x7B", "mistral"),
    // Microsoft Phi
    ("microsoft/phi-1", "Phi 1", "phi"),
    ("microsoft/phi-1_5", "Phi 1.5", "phi"),
    ("microsoft/phi-2", "Phi 2", "phi"),
    ("microsoft/phi-3", "Phi 3", "phi"),
    // Alibaba Qwen
    ("Qwen/Qwen-7B", "Qwen 7B", "qwen"),
    ("Qwen/Qwen-14B", "Qwen 14B", "qwen"),
    ("Qwen/Qwen2-7B", "Qwen 2 7B", "qwen"),
    // Google Gemma
    ("google/gemma-7b", "Gemma 7B", "gemma"),
    ("google/gemma-2b", "Gemma 2B", "gemma"),
];

/// Detect if a model is MLX-compatible
pub fn detect_mlx_model(model_id: &str) -> MinervaResult<MlxModelInfo> {
    // Normalize model ID
    let normalized_id = model_id.to_lowercase();

    // Check registry
    for (registry_id, name, arch) in MLX_MODEL_REGISTRY {
        if normalized_id.contains(&registry_id.to_lowercase()) {
            return Ok(MlxModelInfo {
                model_id: model_id.to_string(),
                name: name.to_string(),
                architecture: arch.to_string(),
                size: estimate_size_category(arch),
                supports_mlx: true,
                mlx_config: Some(create_mlx_config(model_id)?),
                has_mlx_weights: true,
                size_gb: estimate_model_size(arch),
                recommended_ram_gb: estimate_ram_requirement(arch),
            });
        }
    }

    // Generic MLX support for HuggingFace models
    // (most transformer models can run on MLX)
    if is_likely_transformer_model(model_id) {
        let arch = detect_architecture(model_id);
        return Ok(MlxModelInfo {
            model_id: model_id.to_string(),
            name: model_id.to_string(),
            architecture: arch.clone(),
            size: estimate_size_category(&arch),
            supports_mlx: true,
            mlx_config: Some(create_mlx_config(model_id)?),
            has_mlx_weights: false,
            size_gb: estimate_model_size(&arch),
            recommended_ram_gb: estimate_ram_requirement(&arch),
        });
    }

    Err(MinervaError::InferenceError(format!(
        "Model '{}' does not appear to be compatible with MLX. \
         MLX supports most HuggingFace transformer models.",
        model_id
    )))
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Create MLX configuration for a model
fn create_mlx_config(model_id: &str) -> MinervaResult<MlxModelConfig> {
    let base_type = detect_base_type(model_id);
    let (param_count, is_quantized, quant_format) = extract_model_params(model_id);

    Ok(MlxModelConfig {
        model_id: model_id.to_string(),
        base_type: base_type.as_str().to_string(),
        is_quantized,
        quantization_format: quant_format,
        param_count_billions: param_count,
        context_length: 2048, // Default, can be refined per model
        mlx_verified: is_verified_mlx_model(model_id),
        mlx_features: get_mlx_features(&base_type),
    })
}

/// Extract parameter count from model ID
fn extract_param_count(lower: &str) -> f32 {
    // Split by common separators and check each part
    for part in lower.split(|c: char| c == '-' || c == '_' || c == '/' || c == ' ') {
        if part.ends_with('b') && part.len() > 1 {
            // Try to parse the number before 'b'
            let num_str = &part[..part.len() - 1];
            if let Ok(count) = num_str.parse::<f32>() {
                if count > 0.5 && count < 1000.0 {
                    // Reasonable range for parameter counts
                    return count;
                }
            }
        }
    }
    3.0 // Default estimate
}

/// Detect architecture from model ID
fn detect_architecture(model_id: &str) -> String {
    let lower = model_id.to_lowercase();

    if lower.contains("llama") {
        "llama".to_string()
    } else if lower.contains("mistral") || lower.contains("mixtral") {
        "mistral".to_string()
    } else if lower.contains("phi") {
        "phi".to_string()
    } else if lower.contains("qwen") {
        "qwen".to_string()
    } else if lower.contains("gemma") {
        "gemma".to_string()
    } else {
        "unknown".to_string()
    }
}

/// Detect base ModelType from model ID
fn detect_base_type(model_id: &str) -> ModelType {
    let arch = detect_architecture(model_id);
    match arch.as_str() {
        "llama" => ModelType::Llama,
        "mistral" => ModelType::Mistral,
        "phi" => ModelType::Phi,
        "qwen" => ModelType::Qwen,
        _ => ModelType::Unknown,
    }
}

/// Estimate model size category
fn estimate_size_category(architecture: &str) -> String {
    match architecture {
        "phi" => "small".to_string(),
        "llama" | "mistral" | "gemma" => "medium".to_string(),
        _ => "unknown".to_string(),
    }
}

/// Extract parameter count and quantization info
fn extract_model_params(model_id: &str) -> (f32, bool, Option<String>) {
    let lower = model_id.to_lowercase();

    // Check for quantization
    let is_quantized = lower.contains("int4")
        || lower.contains("int8")
        || lower.contains("4bit")
        || lower.contains("8bit")
        || lower.contains("gguf");

    let quant_format = if lower.contains("int4") || lower.contains("4bit") {
        Some("int4".to_string())
    } else if lower.contains("int8") || lower.contains("8bit") {
        Some("int8".to_string())
    } else {
        None
    };

    // Extract parameter count from model ID if present
    // Split by '-' and look for number patterns like "7b", "13b", etc.
    let param_count = extract_param_count(&lower);

    (param_count, is_quantized, quant_format)
}

/// Check if this is a likely transformer model
fn is_likely_transformer_model(model_id: &str) -> bool {
    let lower = model_id.to_lowercase();

    // Check for common transformer naming patterns
    lower.contains("llama")
        || lower.contains("mistral")
        || lower.contains("phi")
        || lower.contains("qwen")
        || lower.contains("gemma")
        || lower.contains("gpt")
        || lower.contains("bert")
        || lower.contains("transformer")
        || lower.contains("falcon")
        || lower.contains("starcoder")
        || lower.contains("codellama")
}

/// Check if model is verified to work with MLX
fn is_verified_mlx_model(model_id: &str) -> bool {
    let lower = model_id.to_lowercase();

    // Check against registry
    MLX_MODEL_REGISTRY
        .iter()
        .any(|(id, _, _)| lower.contains(&id.to_lowercase()))
}

/// Estimate model download size in GB
fn estimate_model_size(architecture: &str) -> f32 {
    match architecture {
        "phi" => 5.0,
        "llama" => 25.0,
        "mistral" => 14.0,
        "gemma" => 5.0,
        _ => 10.0,
    }
}

/// Estimate recommended RAM in GB
fn estimate_ram_requirement(architecture: &str) -> f32 {
    // Rule of thumb: 2x model size for inference with some overhead
    estimate_model_size(architecture) * 2.5
}

/// Get MLX-specific features for architecture
fn get_mlx_features(model_type: &ModelType) -> Vec<String> {
    match model_type {
        ModelType::Llama => vec![
            "flash_attention".to_string(),
            "grouped_query_attention".to_string(),
            "rope_scaling".to_string(),
        ],
        ModelType::Mistral => vec![
            "sliding_window_attention".to_string(),
            "grouped_query_attention".to_string(),
        ],
        ModelType::Phi => vec!["memory_efficient_attention".to_string()],
        ModelType::Qwen => vec!["rotary_embeddings".to_string()],
        _ => vec![],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_llama_model() {
        let info = detect_mlx_model("meta-llama/Llama-2-7b").unwrap();
        assert_eq!(info.architecture, "llama");
        assert!(info.supports_mlx);
        assert!(info.has_mlx_weights);
    }

    #[test]
    fn test_detect_mistral_model() {
        let info = detect_mlx_model("mistralai/Mistral-7B").unwrap();
        assert_eq!(info.architecture, "mistral");
        assert!(info.supports_mlx);
    }

    #[test]
    fn test_detect_phi_model() {
        let info = detect_mlx_model("microsoft/phi-2").unwrap();
        assert_eq!(info.architecture, "phi");
        assert!(info.supports_mlx);
    }

    #[test]
    fn test_detect_qwen_model() {
        let info = detect_mlx_model("Qwen/Qwen-7B").unwrap();
        assert_eq!(info.architecture, "qwen");
        assert!(info.supports_mlx);
    }

    #[test]
    fn test_architecture_detection() {
        assert_eq!(detect_architecture("meta-llama/Llama-2-7b"), "llama");
        assert_eq!(detect_architecture("mistralai/Mistral-7B"), "mistral");
        assert_eq!(detect_architecture("microsoft/phi-2"), "phi");
        assert_eq!(detect_architecture("Qwen/Qwen-7B"), "qwen");
        assert_eq!(detect_architecture("google/gemma-7b"), "gemma");
    }

    #[test]
    fn test_param_count_extraction() {
        let (count, _, _) = extract_model_params("meta-llama/Llama-2-7b");
        assert_eq!(count, 7.0);

        let (count, _, _) = extract_model_params("meta-llama/Llama-2-70b");
        assert_eq!(count, 70.0);

        let (count, _, _) = extract_model_params("microsoft/phi-2");
        // Phi-2 doesn't have "b" suffix, so it defaults to 3.0
        assert_eq!(count, 3.0);
    }

    #[test]
    fn test_quantization_detection() {
        let (_, is_quantized, format) = extract_model_params("meta-llama/Llama-2-7b-int4");
        assert!(is_quantized);
        assert_eq!(format, Some("int4".to_string()));

        let (_, is_quantized, format) = extract_model_params("meta-llama/Llama-2-7b-int8");
        assert!(is_quantized);
        assert_eq!(format, Some("int8".to_string()));
    }

    #[test]
    fn test_mlx_config_creation() {
        let config = create_mlx_config("meta-llama/Llama-2-7b").unwrap();
        assert_eq!(config.model_id, "meta-llama/Llama-2-7b");
        assert_eq!(config.base_type, "llama");
        assert!(config.mlx_verified);
    }

    #[test]
    fn test_mlx_features() {
        let features = get_mlx_features(&ModelType::Llama);
        assert!(!features.is_empty());
        assert!(features.contains(&"flash_attention".to_string()));
    }

    #[test]
    fn test_unsupported_model() {
        let result = detect_mlx_model("random-non-existent-model");
        assert!(result.is_err());
    }
}
