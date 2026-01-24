use crate::error::{MinervaError, MinervaResult};
use std::fs::File;

use super::binary_reader::BinaryReader;
use super::gguf_loader::GGUFModelMetadata;
use super::gguf_metadata_store::GGUFMetadataStore;

/// Parses key-value pairs from GGUF metadata
pub struct GGUFKVParser;

impl GGUFKVParser {
    /// Parse a single key-value pair from file
    pub fn parse_kv_pair(file: &mut File, metadata: &mut GGUFModelMetadata) -> MinervaResult<()> {
        // Read key
        let key_len = BinaryReader::read_u32(file)? as usize;
        let mut key_bytes = vec![0u8; key_len];
        std::io::Read::read_exact(file, &mut key_bytes)
            .map_err(|e| MinervaError::ModelLoadingError(format!("Failed to read key: {}", e)))?;
        let key = String::from_utf8_lossy(&key_bytes).to_string();

        // Read value type
        let value_type = BinaryReader::read_u32(file)?;

        // Parse value based on type
        match value_type {
            0 => {
                // u8
                let _val = BinaryReader::read_u8(file)?;
            }
            1 => {
                // i8
                let _val = BinaryReader::read_i8(file)?;
            }
            2 => {
                // u16
                let _val = BinaryReader::read_u16(file)?;
            }
            3 => {
                // i16
                let _val = BinaryReader::read_i16(file)?;
            }
            4 => {
                // u32
                let val = BinaryReader::read_u32(file)?;
                GGUFMetadataStore::store_u32(&key, val, metadata);
            }
            5 => {
                // i32
                let val = BinaryReader::read_i32(file)?;
                GGUFMetadataStore::store_i32(&key, val, metadata);
            }
            6 => {
                // f32
                let _val = BinaryReader::read_f32(file)?;
            }
            7 => {
                // u64
                let val = BinaryReader::read_u64(file)?;
                GGUFMetadataStore::store_u64(&key, val, metadata);
            }
            8 => {
                // i64
                let _val = BinaryReader::read_i64(file)?;
            }
            9 => {
                // f64
                let _val = BinaryReader::read_f64(file)?;
            }
            10 => {
                // bool
                let _val = BinaryReader::read_u8(file)? != 0;
            }
            11 => {
                // string
                let str_val = BinaryReader::read_string(file)?;
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
