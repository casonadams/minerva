use crate::error::{MinervaError, MinervaResult};
use std::fs::File;
use std::io::Read;

use super::gguf_loader::GGUFModelMetadata;
use super::gguf_metadata_store::GGUFMetadataStore;

/// Parses key-value pairs from GGUF metadata
pub struct GGUFKVParser;

impl GGUFKVParser {
    /// Parse a single key-value pair from file
    pub fn parse_kv_pair(file: &mut File, metadata: &mut GGUFModelMetadata) -> MinervaResult<()> {
        // Read key
        let key_len = Self::read_u32(file)? as usize;
        let mut key_bytes = vec![0u8; key_len];
        file.read_exact(&mut key_bytes)
            .map_err(|e| MinervaError::ModelLoadingError(format!("Failed to read key: {}", e)))?;
        let key = String::from_utf8_lossy(&key_bytes).to_string();

        // Read value type
        let value_type = Self::read_u32(file)?;

        // Parse value based on type
        match value_type {
            0 => {
                // u8
                let _val = Self::read_u8(file)?;
            }
            1 => {
                // i8
                let _val = Self::read_i8(file)?;
            }
            2 => {
                // u16
                let _val = Self::read_u16(file)?;
            }
            3 => {
                // i16
                let _val = Self::read_i16(file)?;
            }
            4 => {
                // u32
                let val = Self::read_u32(file)?;
                GGUFMetadataStore::store_u32(&key, val, metadata);
            }
            5 => {
                // i32
                let val = Self::read_i32(file)?;
                GGUFMetadataStore::store_i32(&key, val, metadata);
            }
            6 => {
                // f32
                let _val = Self::read_f32(file)?;
            }
            7 => {
                // u64
                let val = Self::read_u64(file)?;
                GGUFMetadataStore::store_u64(&key, val, metadata);
            }
            8 => {
                // i64
                let _val = Self::read_i64(file)?;
            }
            9 => {
                // f64
                let _val = Self::read_f64(file)?;
            }
            10 => {
                // bool
                let _val = Self::read_u8(file)? != 0;
            }
            11 => {
                // string
                let str_val = Self::read_string(file)?;
                GGUFMetadataStore::store_string(&key, &str_val, metadata);
            }
            _ => {
                return Err(MinervaError::ModelLoadingError(format!(
                    "Unknown metadata type: {}",
                    value_type
                )));
            }
        }

        Ok(())
    }

    fn read_u8(file: &mut File) -> MinervaResult<u8> {
        let mut buf = [0u8; 1];
        file.read_exact(&mut buf)
            .map_err(|e| MinervaError::ModelLoadingError(e.to_string()))?;
        Ok(buf[0])
    }

    fn read_i8(file: &mut File) -> MinervaResult<i8> {
        let mut buf = [0u8; 1];
        file.read_exact(&mut buf)
            .map_err(|e| MinervaError::ModelLoadingError(e.to_string()))?;
        Ok(buf[0] as i8)
    }

    fn read_u16(file: &mut File) -> MinervaResult<u16> {
        let mut buf = [0u8; 2];
        file.read_exact(&mut buf)
            .map_err(|e| MinervaError::ModelLoadingError(e.to_string()))?;
        Ok(u16::from_le_bytes(buf))
    }

    fn read_i16(file: &mut File) -> MinervaResult<i16> {
        let mut buf = [0u8; 2];
        file.read_exact(&mut buf)
            .map_err(|e| MinervaError::ModelLoadingError(e.to_string()))?;
        Ok(i16::from_le_bytes(buf))
    }

    fn read_u32(file: &mut File) -> MinervaResult<u32> {
        let mut buf = [0u8; 4];
        file.read_exact(&mut buf)
            .map_err(|e| MinervaError::ModelLoadingError(e.to_string()))?;
        Ok(u32::from_le_bytes(buf))
    }

    fn read_i32(file: &mut File) -> MinervaResult<i32> {
        let mut buf = [0u8; 4];
        file.read_exact(&mut buf)
            .map_err(|e| MinervaError::ModelLoadingError(e.to_string()))?;
        Ok(i32::from_le_bytes(buf))
    }

    fn read_f32(file: &mut File) -> MinervaResult<f32> {
        let mut buf = [0u8; 4];
        file.read_exact(&mut buf)
            .map_err(|e| MinervaError::ModelLoadingError(e.to_string()))?;
        Ok(f32::from_le_bytes(buf))
    }

    fn read_u64(file: &mut File) -> MinervaResult<u64> {
        let mut buf = [0u8; 8];
        file.read_exact(&mut buf)
            .map_err(|e| MinervaError::ModelLoadingError(e.to_string()))?;
        Ok(u64::from_le_bytes(buf))
    }

    fn read_i64(file: &mut File) -> MinervaResult<i64> {
        let mut buf = [0u8; 8];
        file.read_exact(&mut buf)
            .map_err(|e| MinervaError::ModelLoadingError(e.to_string()))?;
        Ok(i64::from_le_bytes(buf))
    }

    fn read_f64(file: &mut File) -> MinervaResult<f64> {
        let mut buf = [0u8; 8];
        file.read_exact(&mut buf)
            .map_err(|e| MinervaError::ModelLoadingError(e.to_string()))?;
        Ok(f64::from_le_bytes(buf))
    }

    fn read_string(file: &mut File) -> MinervaResult<String> {
        let len = Self::read_u32(file)? as usize;
        let mut buf = vec![0u8; len];
        file.read_exact(&mut buf)
            .map_err(|e| MinervaError::ModelLoadingError(e.to_string()))?;
        String::from_utf8(buf)
            .map_err(|e| MinervaError::ModelLoadingError(format!("Invalid UTF-8 in string: {}", e)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kv_parser_exists() {
        // Placeholder test - actual KV parsing is tested in integration tests
        let _ = GGUFKVParser;
    }
}
