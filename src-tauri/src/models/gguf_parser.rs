use super::gguf_header::{
    read_key, read_kv_count, read_value_type, skip_tensor_count, validate_magic, validate_version,
};
use super::gguf_reader::{read_string_value, read_u32_value, skip_value};
use crate::error::{MinervaError, MinervaResult};
use std::fs::File;
use std::io::Read;
use std::path::Path;

/// GGUF file format parser for extracting model metadata
pub struct GGUFParser;

/// Metadata extracted from GGUF file header
#[derive(Debug, Clone, Default)]
pub struct GGUFMetadata {
    pub context_window: Option<usize>,
    pub model_name: Option<String>,
    pub quantization: Option<String>,
}

impl GGUFParser {
    /// Parse GGUF file and extract metadata
    pub fn parse_metadata(path: &Path) -> MinervaResult<GGUFMetadata> {
        let mut file = File::open(path).map_err(|e| {
            MinervaError::ModelLoadingError(format!("Failed to open GGUF file: {}", e))
        })?;

        validate_magic(&mut file)?;
        validate_version(&mut file)?;
        skip_tensor_count(&mut file)?;

        let kv_count = read_kv_count(&mut file)?;
        let mut metadata = GGUFMetadata::default();

        for _ in 0..kv_count {
            match Self::parse_kv_pair(&mut file, &mut metadata) {
                Ok(_) => continue,
                Err(_) => break,
            }
        }

        Ok(metadata)
    }

    /// Parse a single key-value pair from GGUF file
    fn parse_kv_pair(file: &mut File, metadata: &mut GGUFMetadata) -> MinervaResult<()> {
        let key = read_key(file)?;
        let value_type = read_value_type(file)?;

        match key.as_ref() {
            "general.name" if value_type == 3 => {
                if let Ok(value) = read_string_value(file) {
                    metadata.model_name = Some(value);
                }
            }
            "llama.context_length" if value_type == 4 => {
                if let Ok(value) = read_u32_value(file) {
                    metadata.context_window = Some(value as usize);
                }
            }
            "llama.rope.context_length" if value_type == 4 => {
                if let Ok(value) = read_u32_value(file) {
                    metadata.context_window = Some(value as usize);
                }
            }
            "gptq.desc_act" if value_type == 1 => {
                metadata.quantization = Some("GPTQ".to_string());
                let _ = file.read_exact(&mut [0u8; 1]);
            }
            _ => {
                skip_value(file, value_type)?;
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::TempDir;

    fn create_minimal_gguf(path: &Path) -> MinervaResult<()> {
        let mut file = File::create(path).map_err(|e| {
            MinervaError::ModelLoadingError(format!("Failed to create file: {}", e))
        })?;

        file.write_all(&[0x47, 0x47, 0x55, 0x46])
            .map_err(|e| MinervaError::ModelLoadingError(format!("Write error: {}", e)))?;

        file.write_all(&2u32.to_le_bytes())
            .map_err(|e| MinervaError::ModelLoadingError(format!("Write error: {}", e)))?;

        file.write_all(&0u64.to_le_bytes())
            .map_err(|e| MinervaError::ModelLoadingError(format!("Write error: {}", e)))?;

        file.write_all(&0u64.to_le_bytes())
            .map_err(|e| MinervaError::ModelLoadingError(format!("Write error: {}", e)))?;

        Ok(())
    }

    #[test]
    fn test_parse_valid_gguf_minimal() {
        let temp_dir = TempDir::new().unwrap();
        let gguf_path = temp_dir.path().join("test.gguf");

        create_minimal_gguf(&gguf_path).unwrap();

        let metadata = GGUFParser::parse_metadata(&gguf_path).unwrap();
        assert_eq!(metadata.context_window, None);
        assert_eq!(metadata.model_name, None);
    }

    #[test]
    fn test_parse_invalid_magic() {
        let temp_dir = TempDir::new().unwrap();
        let gguf_path = temp_dir.path().join("bad.gguf");

        let mut file = File::create(&gguf_path).unwrap();
        file.write_all(&[0x00, 0x00, 0x00, 0x00]).unwrap();
        file.write_all(&2u32.to_le_bytes()).unwrap();
        drop(file);

        let result = GGUFParser::parse_metadata(&gguf_path);
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_nonexistent_file() {
        let result = GGUFParser::parse_metadata(Path::new("/nonexistent/test.gguf"));
        assert!(result.is_err());
    }
}
