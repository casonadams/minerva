use crate::error::{MinervaError, MinervaResult};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};

/// Read u32 value from GGUF file
pub fn read_u32_value(file: &mut File) -> MinervaResult<u32> {
    let mut bytes = [0u8; 4];
    file.read_exact(&mut bytes)
        .map_err(|e| MinervaError::ModelLoadingError(format!("Failed to read u32 value: {}", e)))?;
    Ok(u32::from_le_bytes(bytes))
}

/// Read string value from GGUF file
pub fn read_string_value(file: &mut File) -> MinervaResult<String> {
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

/// Skip n bytes in file
pub fn skip_n_bytes(file: &mut File, n: u64) -> MinervaResult<()> {
    file.seek(SeekFrom::Current(n as i64))
        .map_err(|e| MinervaError::ModelLoadingError(format!("Failed to seek: {}", e)))?;
    Ok(())
}

/// Skip string value (reads length, then skips that many bytes)
pub fn skip_string_value(file: &mut File) -> MinervaResult<()> {
    let mut len_bytes = [0u8; 4];
    file.read_exact(&mut len_bytes).map_err(|e| {
        MinervaError::ModelLoadingError(format!("Failed to read string length: {}", e))
    })?;
    let len = u32::from_le_bytes(len_bytes) as u64;
    skip_n_bytes(file, len)
}

/// Skip value in GGUF file based on type
pub fn skip_value(file: &mut File, value_type: u32) -> MinervaResult<()> {
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
