use crate::error::{MinervaError, MinervaResult};
use std::fs::File;
use std::io::Read;

/// Validate GGUF magic number
pub fn validate_magic(file: &mut File) -> MinervaResult<()> {
    let mut magic = [0u8; 4];
    file.read_exact(&mut magic).map_err(|e| {
        MinervaError::ModelLoadingError(format!("Failed to read GGUF magic: {}", e))
    })?;

    if magic != [0x47, 0x47, 0x55, 0x46] {
        return Err(MinervaError::ModelLoadingError(
            "Invalid GGUF magic number".to_string(),
        ));
    }
    Ok(())
}

/// Validate GGUF version (must be 2 or later)
pub fn validate_version(file: &mut File) -> MinervaResult<()> {
    let mut version_bytes = [0u8; 4];
    file.read_exact(&mut version_bytes)
        .map_err(|e| MinervaError::ModelLoadingError(format!("Failed to read version: {}", e)))?;
    let version = u32::from_le_bytes(version_bytes);

    if version < 2 {
        return Err(MinervaError::ModelLoadingError(
            "Unsupported GGUF version".to_string(),
        ));
    }
    Ok(())
}

/// Skip tensor count field
pub fn skip_tensor_count(file: &mut File) -> MinervaResult<()> {
    let mut tensor_count_bytes = [0u8; 8];
    file.read_exact(&mut tensor_count_bytes).map_err(|e| {
        MinervaError::ModelLoadingError(format!("Failed to read tensor count: {}", e))
    })?;
    Ok(())
}

/// Read KV pairs count
pub fn read_kv_count(file: &mut File) -> MinervaResult<u64> {
    let mut kv_count_bytes = [0u8; 8];
    file.read_exact(&mut kv_count_bytes)
        .map_err(|e| MinervaError::ModelLoadingError(format!("Failed to read kv count: {}", e)))?;
    Ok(u64::from_le_bytes(kv_count_bytes))
}

/// Read KV pair key
pub fn read_key(file: &mut File) -> MinervaResult<String> {
    let mut key_len_bytes = [0u8; 4];
    file.read_exact(&mut key_len_bytes).map_err(|e| {
        MinervaError::ModelLoadingError(format!("Failed to read key length: {}", e))
    })?;
    let key_len = u32::from_le_bytes(key_len_bytes) as usize;

    let mut key_bytes = vec![0u8; key_len];
    file.read_exact(&mut key_bytes)
        .map_err(|e| MinervaError::ModelLoadingError(format!("Failed to read key: {}", e)))?;
    Ok(String::from_utf8_lossy(&key_bytes).to_string())
}

/// Read value type field
pub fn read_value_type(file: &mut File) -> MinervaResult<u32> {
    let mut value_type_bytes = [0u8; 4];
    file.read_exact(&mut value_type_bytes).map_err(|e| {
        MinervaError::ModelLoadingError(format!("Failed to read value type: {}", e))
    })?;
    Ok(u32::from_le_bytes(value_type_bytes))
}
