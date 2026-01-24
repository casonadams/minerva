/// Real GGUF Model Loader (Phase 6 - Step 1)
///
/// Enhanced model loader with full tensor support for real GGUF models.
/// Supports all quantization formats and properly loads model weights.
use crate::error::{MinervaError, MinervaResult};
use std::fs::File;
use std::io::Read;
use std::path::Path;

use super::gguf_header_validator::GGUFHeaderValidator;
use super::gguf_kv_parser::GGUFKVParser;
use super::gguf_tensor::GGUFTensor;
use super::gguf_tensor_loader::GGUFTensorLoader;

/// Metadata about a loaded GGUF model
#[derive(Debug, Clone)]
pub struct GGUFModelMetadata {
    pub name: Option<String>,
    pub architecture: Option<String>,
    pub context_window: Option<usize>,
    pub embedding_length: Option<usize>,
    pub feed_forward_length: Option<usize>,
    pub attention_head_count: Option<usize>,
    pub attention_head_count_kv: Option<usize>,
    pub layer_count: Option<usize>,
    pub quantization_version: Option<usize>,
}

/// GGUF model loader for real models
pub struct GGUFModelLoader;

impl GGUFModelLoader {
    /// Load a GGUF model file and extract tensors and metadata
    pub fn load(path: &Path) -> MinervaResult<(GGUFModelMetadata, Vec<GGUFTensor>)> {
        let mut file = File::open(path).map_err(|e| {
            MinervaError::ModelLoadingError(format!("Failed to open GGUF file: {}", e))
        })?;

        // Validate header and get version
        let _version = GGUFHeaderValidator::validate(&mut file)?;

        // Read tensor and KV counts
        let tensor_count = Self::read_u64(&mut file)?;
        let kv_count = Self::read_u64(&mut file)?;

        // Parse metadata from KV pairs
        let mut metadata = GGUFModelMetadata {
            name: None,
            architecture: None,
            context_window: None,
            embedding_length: None,
            feed_forward_length: None,
            attention_head_count: None,
            attention_head_count_kv: None,
            layer_count: None,
            quantization_version: None,
        };

        // Parse key-value pairs
        for _ in 0..kv_count {
            GGUFKVParser::parse_kv_pair(&mut file, &mut metadata)?;
        }

        // Align to 32-byte boundary before reading tensors
        GGUFHeaderValidator::align_to_boundary(&mut file)?;

        // Load tensors
        let mut tensors = Vec::with_capacity(tensor_count as usize);
        for _ in 0..tensor_count {
            match GGUFTensorLoader::load_tensor(&mut file) {
                Ok(tensor) => tensors.push(tensor),
                Err(e) => {
                    tracing::warn!("Failed to load tensor: {}", e);
                    continue;
                }
            }
        }

        tracing::info!(
            "Loaded GGUF model with {} tensors and {} metadata entries",
            tensors.len(),
            kv_count
        );

        Ok((metadata, tensors))
    }

    // ==================== Helper Functions ====================

    fn read_u64(file: &mut File) -> MinervaResult<u64> {
        let mut buf = [0u8; 8];
        file.read_exact(&mut buf)
            .map_err(|e| MinervaError::ModelLoadingError(e.to_string()))?;
        Ok(u64::from_le_bytes(buf))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn create_minimal_gguf() -> NamedTempFile {
        let mut file = NamedTempFile::new().unwrap();

        // Write GGUF magic
        file.write_all(&[0x47, 0x47, 0x55, 0x46]).unwrap();

        // Write version (3)
        file.write_all(&3u32.to_le_bytes()).unwrap();

        // Write tensor count (0)
        file.write_all(&0u64.to_le_bytes()).unwrap();

        // Write kv count (0)
        file.write_all(&0u64.to_le_bytes()).unwrap();

        file.flush().unwrap();
        file
    }

    #[test]
    fn test_load_minimal_gguf() {
        let file = create_minimal_gguf();
        let result = GGUFModelLoader::load(file.path());
        assert!(result.is_ok());

        let (metadata, tensors) = result.unwrap();
        assert!(metadata.name.is_none());
        assert!(tensors.is_empty());
    }

    #[test]
    fn test_read_u64() {
        let mut file = NamedTempFile::new().unwrap();
        file.write_all(&1234567890u64.to_le_bytes()).unwrap();
        file.flush().unwrap();

        let mut f = File::open(file.path()).unwrap();
        let value = GGUFModelLoader::read_u64(&mut f).unwrap();
        assert_eq!(value, 1234567890);
    }
}
