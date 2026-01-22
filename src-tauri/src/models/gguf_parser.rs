use crate::error::{MinervaError, MinervaResult};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
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

        // Read and validate magic number
        let mut magic = [0u8; 4];
        file.read_exact(&mut magic).map_err(|e| {
            MinervaError::ModelLoadingError(format!("Failed to read GGUF magic: {}", e))
        })?;

        // GGUF magic is "GGUF" in little-endian
        if magic != [0x47, 0x47, 0x55, 0x46] {
            return Err(MinervaError::ModelLoadingError(
                "Invalid GGUF magic number".to_string(),
            ));
        }

        // Read version (4 bytes, little-endian)
        let mut version_bytes = [0u8; 4];
        file.read_exact(&mut version_bytes).map_err(|e| {
            MinervaError::ModelLoadingError(format!("Failed to read version: {}", e))
        })?;
        let version = u32::from_le_bytes(version_bytes);

        // Support versions 2 and 3
        if version < 2 {
            return Err(MinervaError::ModelLoadingError(
                "Unsupported GGUF version".to_string(),
            ));
        }

        // Read tensor count (8 bytes, little-endian)
        let mut tensor_count_bytes = [0u8; 8];
        file.read_exact(&mut tensor_count_bytes).map_err(|e| {
            MinervaError::ModelLoadingError(format!("Failed to read tensor count: {}", e))
        })?;
        let _tensor_count = u64::from_le_bytes(tensor_count_bytes);

        // Read kv pairs count (8 bytes, little-endian)
        let mut kv_count_bytes = [0u8; 8];
        file.read_exact(&mut kv_count_bytes).map_err(|e| {
            MinervaError::ModelLoadingError(format!("Failed to read kv count: {}", e))
        })?;
        let kv_count = u64::from_le_bytes(kv_count_bytes);

        // Parse key-value pairs
        let mut metadata = GGUFMetadata {
            context_window: None,
            model_name: None,
            quantization: None,
        };

        for _ in 0..kv_count {
            match Self::parse_kv_pair(&mut file, &mut metadata) {
                Ok(_) => continue,
                Err(_) => break, // Stop on any parsing error to avoid corrupting metadata
            }
        }

        Ok(metadata)
    }

    /// Parse a single key-value pair from GGUF file
    fn parse_kv_pair(file: &mut File, metadata: &mut GGUFMetadata) -> MinervaResult<()> {
        // Read key length (4 bytes, little-endian)
        let mut key_len_bytes = [0u8; 4];
        file.read_exact(&mut key_len_bytes).map_err(|e| {
            MinervaError::ModelLoadingError(format!("Failed to read key length: {}", e))
        })?;
        let key_len = u32::from_le_bytes(key_len_bytes) as usize;

        // Read key
        let mut key_bytes = vec![0u8; key_len];
        file.read_exact(&mut key_bytes)
            .map_err(|e| MinervaError::ModelLoadingError(format!("Failed to read key: {}", e)))?;
        let key = String::from_utf8_lossy(&key_bytes);

        // Read value type (4 bytes, little-endian)
        let mut value_type_bytes = [0u8; 4];
        file.read_exact(&mut value_type_bytes).map_err(|e| {
            MinervaError::ModelLoadingError(format!("Failed to read value type: {}", e))
        })?;
        let value_type = u32::from_le_bytes(value_type_bytes);

        // Parse value based on type
        match key.as_ref() {
            "general.name" if value_type == 3 => {
                if let Ok(value) = Self::read_string_value(file) {
                    metadata.model_name = Some(value);
                }
            }
            "llama.context_length" if value_type == 4 => {
                if let Ok(value) = Self::read_u32_value(file) {
                    metadata.context_window = Some(value as usize);
                }
            }
            "llama.rope.context_length" if value_type == 4 => {
                if let Ok(value) = Self::read_u32_value(file) {
                    metadata.context_window = Some(value as usize);
                }
            }
            "gptq.desc_act" if value_type == 1 => {
                metadata.quantization = Some("GPTQ".to_string());
                let _ = file.read_exact(&mut [0u8; 1]); // read boolean
            }
            _ => {
                // Skip unknown value types
                skip_value(file, value_type)?;
            }
        }

        Ok(())
    }

    /// Read string value from GGUF file
    fn read_string_value(file: &mut File) -> MinervaResult<String> {
        let mut len_bytes = [0u8; 4];
        file.read_exact(&mut len_bytes).map_err(|e| {
            MinervaError::ModelLoadingError(format!("Failed to read string length: {}", e))
        })?;
        let len = u32::from_le_bytes(len_bytes) as usize;

        let mut string_bytes = vec![0u8; len];
        file.read_exact(&mut string_bytes).map_err(|e| {
            MinervaError::ModelLoadingError(format!("Failed to read string value: {}", e))
        })?;

        Ok(String::from_utf8_lossy(&string_bytes).to_string())
    }

    /// Read u32 value from GGUF file
    fn read_u32_value(file: &mut File) -> MinervaResult<u32> {
        let mut bytes = [0u8; 4];
        file.read_exact(&mut bytes).map_err(|e| {
            MinervaError::ModelLoadingError(format!("Failed to read u32 value: {}", e))
        })?;
        Ok(u32::from_le_bytes(bytes))
    }
}

/// Skip value in GGUF file based on type
fn skip_value(file: &mut File, value_type: u32) -> MinervaResult<()> {
    match value_type {
        0 => skip_n_bytes(file, 4),   // u32
        1 => skip_n_bytes(file, 1),   // bool
        2 => skip_n_bytes(file, 4),   // i32
        3 => skip_string_value(file), // string
        4 => skip_n_bytes(file, 4),   // u32 (array)
        5 => skip_n_bytes(file, 4),   // i32 (array)
        6 => skip_n_bytes(file, 4),   // f32 (array)
        _ => Err(MinervaError::ModelLoadingError(format!(
            "Unknown GGUF value type: {}",
            value_type
        ))),
    }
}

/// Skip a specific number of bytes
fn skip_n_bytes(file: &mut File, n: u64) -> MinervaResult<()> {
    file.seek(SeekFrom::Current(n as i64))
        .map_err(|e| MinervaError::ModelLoadingError(format!("Failed to seek: {}", e)))?;
    Ok(())
}

/// Skip string value
fn skip_string_value(file: &mut File) -> MinervaResult<()> {
    let mut len_bytes = [0u8; 4];
    file.read_exact(&mut len_bytes).map_err(|e| {
        MinervaError::ModelLoadingError(format!("Failed to read string length: {}", e))
    })?;
    let len = u32::from_le_bytes(len_bytes) as u64;
    skip_n_bytes(file, len)
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

        // Write magic number
        file.write_all(&[0x47, 0x47, 0x55, 0x46])
            .map_err(|e| MinervaError::ModelLoadingError(format!("Write error: {}", e)))?;

        // Write version (2)
        file.write_all(&2u32.to_le_bytes())
            .map_err(|e| MinervaError::ModelLoadingError(format!("Write error: {}", e)))?;

        // Write tensor count (0)
        file.write_all(&0u64.to_le_bytes())
            .map_err(|e| MinervaError::ModelLoadingError(format!("Write error: {}", e)))?;

        // Write kv count (0)
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
        file.write_all(&[0x00, 0x00, 0x00, 0x00]).unwrap(); // Invalid magic
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
