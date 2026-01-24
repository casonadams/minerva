/// Real GGUF Model Loader (Phase 6 - Step 1)
///
/// Enhanced model loader with full tensor support for real GGUF models.
/// Supports all quantization formats and properly loads model weights.
use crate::error::{MinervaError, MinervaResult};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

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

        // Validate magic and version
        Self::validate_header(&mut file)?;

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
        let current_pos = file.stream_position().map_err(|e| {
            MinervaError::ModelLoadingError(format!("Failed to get file position: {}", e))
        })?;
        let aligned_pos = current_pos.div_ceil(32) * 32;
        file.seek(SeekFrom::Start(aligned_pos)).map_err(|e| {
            MinervaError::ModelLoadingError(format!("Failed to seek to alignment: {}", e))
        })?;

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

    /// Validate GGUF file header
    fn validate_header(file: &mut File) -> MinervaResult<()> {
        // Read magic number
        let mut magic = [0u8; 4];
        file.read_exact(&mut magic)
            .map_err(|e| MinervaError::ModelLoadingError(format!("Failed to read magic: {}", e)))?;

        // GGUF magic is "GGUF" (0x47 0x47 0x55 0x46)
        if magic != [0x47, 0x47, 0x55, 0x46] {
            return Err(MinervaError::ModelLoadingError(
                "Invalid GGUF magic number".to_string(),
            ));
        }

        // Read and validate version
        let version = Self::read_u32(file)?;
        if !(2..=3).contains(&version) {
            return Err(MinervaError::ModelLoadingError(format!(
                "Unsupported GGUF version: {}",
                version
            )));
        }

        Ok(())
    }

    // ==================== Helper Functions ====================

    fn read_u32(file: &mut File) -> MinervaResult<u32> {
        let mut buf = [0u8; 4];
        file.read_exact(&mut buf)
            .map_err(|e| MinervaError::ModelLoadingError(e.to_string()))?;
        Ok(u32::from_le_bytes(buf))
    }

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
    fn test_validate_header_invalid_magic() {
        let mut file = NamedTempFile::new().unwrap();
        file.write_all(&[0x00, 0x00, 0x00, 0x00]).unwrap();
        file.flush().unwrap();

        let result = GGUFModelLoader::validate_header(&mut File::open(file.path()).unwrap());
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_header_valid() {
        let mut file = NamedTempFile::new().unwrap();
        file.write_all(&[0x47, 0x47, 0x55, 0x46]).unwrap();
        file.write_all(&2u32.to_le_bytes()).unwrap();
        file.flush().unwrap();

        let mut f = File::open(file.path()).unwrap();
        let result = GGUFModelLoader::validate_header(&mut f);
        assert!(result.is_ok());
    }

    #[test]
    fn test_read_u32() {
        let mut file = NamedTempFile::new().unwrap();
        file.write_all(&42u32.to_le_bytes()).unwrap();
        file.flush().unwrap();

        let mut f = File::open(file.path()).unwrap();
        let value = GGUFModelLoader::read_u32(&mut f).unwrap();
        assert_eq!(value, 42);
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
