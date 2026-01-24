/// Backend Selection and Routing Module - Phase 8-Step 3b Day 4
///
/// This module provides intelligent backend selection for inference based on:
/// - Model file format (GGUF, Safetensors, HuggingFace)
/// - User preferences (auto, force llama.cpp, force pure-rust, fallback)
/// - Model availability and supported features
///
/// # Architecture
///
/// ```text
/// User Request (model_path, backend_preference)
///     ↓
/// BackendSelector::select()
///     ├─ Format Detection
///     ├─ Preference Check
///     ├─ Availability Check
///     └─ Fallback Logic
///     ↓
/// InferenceBackend Implementation
/// (LlamaCppBackend or PureRustBackend)
/// ```
///
/// # Strategy
///
/// - **Auto (recommended)**: Intelligently choose based on format
///   - GGUF → LlamaCppBackend (optimized, GPU support)
///   - Safetensors → PureRustBackend (pure Rust, no deps)
///   - Fallback to other if primary unavailable
///
/// - **LlamaCpp (force)**: Use llama.cpp for all models
///   - Works with GGUF natively
///   - Error for unsupported formats with guidance
///
/// - **PureRust (force)**: Use pure Rust backend for all models
///   - Works with Safetensors and HuggingFace formats
///   - Error for unsupported formats
///
/// - **Fallback**: Try primary, automatically switch if fails
///   - Transparent to user
///   - Maximum compatibility
///
/// # Phase 9 Enhancements
///
/// - Model format conversion on-demand
/// - Backend capability detection
/// - Performance profiling and selection
/// - Caching backend instances
/// - Load balancing between backends
use crate::error::{MinervaError, MinervaResult};
use std::path::Path;

/// Backend selection preference
///
/// Controls which inference backend to use for model loading and generation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BackendPreference {
    /// Automatically select based on model format (recommended)
    ///
    /// - GGUF files → LlamaCppBackend
    /// - Safetensors files → PureRustBackend
    /// - Other formats → Error with guidance
    #[default]
    Auto,

    /// Force use of llama.cpp backend
    ///
    /// Works best with GGUF files. Other formats will produce errors
    /// with helpful messages suggesting conversion.
    LlamaCpp,

    /// Force use of pure Rust backend
    ///
    /// Works best with Safetensors and HuggingFace formats.
    /// Requires no external dependencies.
    PureRust,

    /// Try primary backend, automatically fallback to secondary if needed
    ///
    /// Useful for maximum compatibility when format is uncertain.
    /// Phase 9: Will implement actual fallback logic.
    Fallback,
}

/// Detected model file format
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelFormat {
    /// GGUF quantized format (llama.cpp native)
    Gguf,

    /// Safetensors format (HuggingFace standard)
    Safetensors,

    /// HuggingFace PyTorch format (.bin)
    HuggingFaceBin,

    /// PyTorch format (.pt, .pth)
    PyTorch,

    /// TensorFlow format (.pb)
    TensorFlow,

    /// Unknown or unsupported format
    Unknown,
}

impl ModelFormat {
    /// Detect format from file path/extension
    pub fn detect(path: &Path) -> Self {
        let extension = path
            .extension()
            .and_then(|ext| ext.to_str())
            .unwrap_or("")
            .to_lowercase();

        match extension.as_str() {
            "gguf" => ModelFormat::Gguf,
            "safetensors" => ModelFormat::Safetensors,
            "bin" => {
                // HuggingFace .bin files are PyTorch format
                // But we treat them specially for detection
                ModelFormat::HuggingFaceBin
            }
            "pt" | "pth" => ModelFormat::PyTorch,
            "pb" => ModelFormat::TensorFlow,
            _ => ModelFormat::Unknown,
        }
    }

    /// Check if this format is natively supported by a backend
    pub fn is_supported_by_llama_cpp(&self) -> bool {
        matches!(self, ModelFormat::Gguf)
    }

    pub fn is_supported_by_pure_rust(&self) -> bool {
        matches!(self, ModelFormat::Safetensors | ModelFormat::HuggingFaceBin)
    }

    /// Get human-readable format name
    pub fn name(&self) -> &'static str {
        match self {
            ModelFormat::Gguf => "GGUF",
            ModelFormat::Safetensors => "Safetensors",
            ModelFormat::HuggingFaceBin => "HuggingFace PyTorch",
            ModelFormat::PyTorch => "PyTorch",
            ModelFormat::TensorFlow => "TensorFlow",
            ModelFormat::Unknown => "Unknown",
        }
    }
}

/// Backend selection strategy result
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BackendChoice {
    /// Use llama.cpp backend
    UseLlamaCpp,

    /// Use pure Rust backend
    UsePureRust,

    /// Error with helpful guidance
    Error(String),
}

/// Backend selector for intelligent routing
pub struct BackendSelector;

impl BackendSelector {
    /// Select backend based on model path and preference
    ///
    /// # Arguments
    ///
    /// * `path` - Path to model file
    /// * `preference` - User's backend preference
    ///
    /// # Returns
    ///
    /// * `BackendChoice::UseLlamaCpp` - Use llama.cpp backend
    /// * `BackendChoice::UsePureRust` - Use pure Rust backend
    /// * `BackendChoice::Error(msg)` - Cannot select (with helpful guidance)
    pub fn select(path: &Path, preference: BackendPreference) -> BackendChoice {
        let format = ModelFormat::detect(path);

        match preference {
            BackendPreference::Auto => Self::select_auto(format),
            BackendPreference::LlamaCpp => Self::select_llama_cpp(format),
            BackendPreference::PureRust => Self::select_pure_rust(format),
            BackendPreference::Fallback => Self::select_fallback(format),
        }
    }

    /// Auto selection: choose best backend based on format
    fn select_auto(format: ModelFormat) -> BackendChoice {
        match format {
            ModelFormat::Gguf => {
                tracing::info!("Auto-selecting llama.cpp backend for GGUF format");
                BackendChoice::UseLlamaCpp
            }
            ModelFormat::Safetensors | ModelFormat::HuggingFaceBin => {
                tracing::info!(
                    "Auto-selecting pure Rust backend for {} format",
                    format.name()
                );
                BackendChoice::UsePureRust
            }
            ModelFormat::PyTorch => BackendChoice::Error(format!(
                "Unsupported format: {} (PyTorch .pt/.pth files)

SOLUTION:
The model is in PyTorch format but not supported directly.
Please convert to one of these formats:

1. GGUF (recommended for speed):
   - Use llama.cpp's conversion script
   - Command: python convert.py model.pth --outtype q4_k_m
   - Then use llama.cpp backend

2. Safetensors (for pure Rust):
   - Convert PyTorch to Safetensors
   - Use HuggingFace's safetensors library
   - Then use pure Rust backend

Learn more: https://huggingface.co/docs/safetensors/en/index",
                format.name()
            )),
            ModelFormat::TensorFlow => BackendChoice::Error(format!(
                "Unsupported format: {} (TensorFlow .pb files)

SOLUTION:
The model is in TensorFlow format but not supported directly.
Please convert to one of these formats:

1. Convert to GGUF:
   - Use tf2onnx to convert to ONNX first
   - Then use ONNX to GGUF conversion

2. Convert to Safetensors:
   - Use TensorFlow to PyTorch conversion
   - Then convert to Safetensors format",
                format.name()
            )),
            ModelFormat::Unknown => BackendChoice::Error(
                "Unknown or unsupported model format

SUPPORTED FORMATS:
- *.gguf (GGUF quantized format)
- *.safetensors (HuggingFace Safetensors)
- *.bin (HuggingFace PyTorch format)

UNSUPPORTED FORMATS (needs conversion):
- *.pt, *.pth (PyTorch)
- *.pb (TensorFlow)

Please ensure model has correct extension or convert to supported format."
                    .to_string(),
            ),
        }
    }

    /// Force llama.cpp selection
    fn select_llama_cpp(format: ModelFormat) -> BackendChoice {
        if format.is_supported_by_llama_cpp() {
            tracing::info!(
                "Using llama.cpp backend (forced) for {} format",
                format.name()
            );
            BackendChoice::UseLlamaCpp
        } else {
            BackendChoice::Error(format!(
                "Cannot use llama.cpp backend with {} format

llama.cpp ONLY supports:
- *.gguf (GGUF quantized format)

Your model format: {} ({})",
                format.name(),
                format.name(),
                match format {
                    ModelFormat::Safetensors => "supported by pure Rust backend",
                    ModelFormat::HuggingFaceBin => "supported by pure Rust backend",
                    _ => "conversion needed",
                }
            ))
        }
    }

    /// Force pure Rust selection
    fn select_pure_rust(format: ModelFormat) -> BackendChoice {
        if format.is_supported_by_pure_rust() {
            tracing::info!(
                "Using pure Rust backend (forced) for {} format",
                format.name()
            );
            BackendChoice::UsePureRust
        } else {
            BackendChoice::Error(format!(
                "Cannot use pure Rust backend with {} format

Pure Rust backend supports:
- *.safetensors (HuggingFace Safetensors)
- *.bin (HuggingFace PyTorch format)

Your model format: {} ({})",
                format.name(),
                format.name(),
                match format {
                    ModelFormat::Gguf => "supported by llama.cpp backend",
                    _ => "conversion needed",
                }
            ))
        }
    }

    /// Fallback selection: try primary, fallback to secondary
    fn select_fallback(format: ModelFormat) -> BackendChoice {
        // Primary: llama.cpp for all formats
        // Secondary: pure Rust for Safetensors/HuggingFace
        match format {
            ModelFormat::Gguf => {
                tracing::info!(
                    "Fallback: trying llama.cpp first for GGUF, will try pure Rust if needed"
                );
                BackendChoice::UseLlamaCpp
            }
            ModelFormat::Safetensors | ModelFormat::HuggingFaceBin => {
                tracing::info!(
                    "Fallback: using pure Rust for {} format (primary for this format)",
                    format.name()
                );
                BackendChoice::UsePureRust
            }
            _ => BackendChoice::Error(format!(
                "Format {} is not supported by any backend in fallback mode",
                format.name()
            )),
        }
    }

    /// Convert BackendChoice error to MinervaResult
    pub fn to_result(choice: BackendChoice) -> MinervaResult<BackendChoice> {
        match choice {
            BackendChoice::Error(msg) => Err(MinervaError::InferenceError(msg)),
            _ => Ok(choice),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_format_detection_gguf() {
        let path = Path::new("model.gguf");
        assert_eq!(ModelFormat::detect(path), ModelFormat::Gguf);
    }

    #[test]
    fn test_model_format_detection_safetensors() {
        let path = Path::new("model.safetensors");
        assert_eq!(ModelFormat::detect(path), ModelFormat::Safetensors);
    }

    #[test]
    fn test_model_format_detection_huggingface() {
        let path = Path::new("pytorch_model.bin");
        assert_eq!(ModelFormat::detect(path), ModelFormat::HuggingFaceBin);
    }

    #[test]
    fn test_model_format_detection_pytorch() {
        let path = Path::new("model.pt");
        assert_eq!(ModelFormat::detect(path), ModelFormat::PyTorch);
    }

    #[test]
    fn test_model_format_detection_tensorflow() {
        let path = Path::new("saved_model.pb");
        assert_eq!(ModelFormat::detect(path), ModelFormat::TensorFlow);
    }

    #[test]
    fn test_model_format_detection_unknown() {
        let path = Path::new("model.unknown");
        assert_eq!(ModelFormat::detect(path), ModelFormat::Unknown);
    }

    #[test]
    fn test_model_format_case_insensitive() {
        let path = Path::new("model.GGUF");
        assert_eq!(ModelFormat::detect(path), ModelFormat::Gguf);
    }

    #[test]
    fn test_format_supported_by_llama_cpp() {
        assert!(ModelFormat::Gguf.is_supported_by_llama_cpp());
        assert!(!ModelFormat::Safetensors.is_supported_by_llama_cpp());
    }

    #[test]
    fn test_format_supported_by_pure_rust() {
        assert!(ModelFormat::Safetensors.is_supported_by_pure_rust());
        assert!(ModelFormat::HuggingFaceBin.is_supported_by_pure_rust());
        assert!(!ModelFormat::Gguf.is_supported_by_pure_rust());
    }

    #[test]
    fn test_backend_selection_auto_gguf() {
        let path = Path::new("model.gguf");
        let choice = BackendSelector::select(path, BackendPreference::Auto);
        assert_eq!(choice, BackendChoice::UseLlamaCpp);
    }

    #[test]
    fn test_backend_selection_auto_safetensors() {
        let path = Path::new("model.safetensors");
        let choice = BackendSelector::select(path, BackendPreference::Auto);
        assert_eq!(choice, BackendChoice::UsePureRust);
    }

    #[test]
    fn test_backend_selection_auto_huggingface_bin() {
        let path = Path::new("pytorch_model.bin");
        let choice = BackendSelector::select(path, BackendPreference::Auto);
        assert_eq!(choice, BackendChoice::UsePureRust);
    }

    #[test]
    fn test_backend_selection_auto_pytorch_error() {
        let path = Path::new("model.pt");
        let choice = BackendSelector::select(path, BackendPreference::Auto);
        assert!(matches!(choice, BackendChoice::Error(_)));
    }

    #[test]
    fn test_backend_selection_force_llama_cpp() {
        let path = Path::new("model.gguf");
        let choice = BackendSelector::select(path, BackendPreference::LlamaCpp);
        assert_eq!(choice, BackendChoice::UseLlamaCpp);
    }

    #[test]
    fn test_backend_selection_force_llama_cpp_unsupported() {
        let path = Path::new("model.safetensors");
        let choice = BackendSelector::select(path, BackendPreference::LlamaCpp);
        assert!(matches!(choice, BackendChoice::Error(_)));
    }

    #[test]
    fn test_backend_selection_force_pure_rust() {
        let path = Path::new("model.safetensors");
        let choice = BackendSelector::select(path, BackendPreference::PureRust);
        assert_eq!(choice, BackendChoice::UsePureRust);
    }

    #[test]
    fn test_backend_selection_force_pure_rust_unsupported() {
        let path = Path::new("model.gguf");
        let choice = BackendSelector::select(path, BackendPreference::PureRust);
        assert!(matches!(choice, BackendChoice::Error(_)));
    }

    #[test]
    fn test_backend_selection_fallback_gguf() {
        let path = Path::new("model.gguf");
        let choice = BackendSelector::select(path, BackendPreference::Fallback);
        assert_eq!(choice, BackendChoice::UseLlamaCpp);
    }

    #[test]
    fn test_backend_selection_fallback_safetensors() {
        let path = Path::new("model.safetensors");
        let choice = BackendSelector::select(path, BackendPreference::Fallback);
        assert_eq!(choice, BackendChoice::UsePureRust);
    }

    #[test]
    fn test_backend_preference_default() {
        assert_eq!(BackendPreference::default(), BackendPreference::Auto);
    }

    #[test]
    fn test_model_format_names() {
        assert_eq!(ModelFormat::Gguf.name(), "GGUF");
        assert_eq!(ModelFormat::Safetensors.name(), "Safetensors");
        assert_eq!(ModelFormat::HuggingFaceBin.name(), "HuggingFace PyTorch");
    }

    #[test]
    fn test_error_messages_are_informative() {
        let path = Path::new("model.pt");
        let choice = BackendSelector::select(path, BackendPreference::Auto);
        if let BackendChoice::Error(msg) = choice {
            assert!(msg.contains("PyTorch"));
            assert!(msg.contains("SOLUTION"));
        } else {
            panic!("Expected error for PyTorch format");
        }
    }

    #[test]
    fn test_to_result_success() {
        let choice = BackendChoice::UseLlamaCpp;
        let result = BackendSelector::to_result(choice);
        assert!(result.is_ok());
    }

    #[test]
    fn test_to_result_error() {
        let choice = BackendChoice::Error("Test error".to_string());
        let result = BackendSelector::to_result(choice);
        assert!(result.is_err());
    }
}
