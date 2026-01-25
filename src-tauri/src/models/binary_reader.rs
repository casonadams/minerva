use crate::error::{MinervaError, MinervaResult};
use std::fs::File;
use std::io::Read;

/// Helper for reading binary data from GGUF files
pub struct BinaryReader;

impl BinaryReader {
    /// Read a u8 value
    pub fn read_u8(file: &mut File) -> MinervaResult<u8> {
        let mut buf = [0u8; 1];
        file.read_exact(&mut buf)
            .map_err(|e| MinervaError::ModelLoadingError(e.to_string()))?;
        Ok(buf[0])
    }

    /// Read an i8 value
    pub fn read_i8(file: &mut File) -> MinervaResult<i8> {
        let mut buf = [0u8; 1];
        file.read_exact(&mut buf)
            .map_err(|e| MinervaError::ModelLoadingError(e.to_string()))?;
        Ok(buf[0] as i8)
    }

    /// Read a u16 value
    pub fn read_u16(file: &mut File) -> MinervaResult<u16> {
        let mut buf = [0u8; 2];
        file.read_exact(&mut buf)
            .map_err(|e| MinervaError::ModelLoadingError(e.to_string()))?;
        Ok(u16::from_le_bytes(buf))
    }

    /// Read an i16 value
    pub fn read_i16(file: &mut File) -> MinervaResult<i16> {
        let mut buf = [0u8; 2];
        file.read_exact(&mut buf)
            .map_err(|e| MinervaError::ModelLoadingError(e.to_string()))?;
        Ok(i16::from_le_bytes(buf))
    }

    /// Read a u32 value
    pub fn read_u32(file: &mut File) -> MinervaResult<u32> {
        let mut buf = [0u8; 4];
        file.read_exact(&mut buf)
            .map_err(|e| MinervaError::ModelLoadingError(e.to_string()))?;
        Ok(u32::from_le_bytes(buf))
    }

    /// Read an i32 value
    pub fn read_i32(file: &mut File) -> MinervaResult<i32> {
        let mut buf = [0u8; 4];
        file.read_exact(&mut buf)
            .map_err(|e| MinervaError::ModelLoadingError(e.to_string()))?;
        Ok(i32::from_le_bytes(buf))
    }

    /// Read an f32 value
    pub fn read_f32(file: &mut File) -> MinervaResult<f32> {
        let mut buf = [0u8; 4];
        file.read_exact(&mut buf)
            .map_err(|e| MinervaError::ModelLoadingError(e.to_string()))?;
        Ok(f32::from_le_bytes(buf))
    }

    /// Read a u64 value
    pub fn read_u64(file: &mut File) -> MinervaResult<u64> {
        let mut buf = [0u8; 8];
        file.read_exact(&mut buf)
            .map_err(|e| MinervaError::ModelLoadingError(e.to_string()))?;
        Ok(u64::from_le_bytes(buf))
    }

    /// Read an i64 value
    pub fn read_i64(file: &mut File) -> MinervaResult<i64> {
        let mut buf = [0u8; 8];
        file.read_exact(&mut buf)
            .map_err(|e| MinervaError::ModelLoadingError(e.to_string()))?;
        Ok(i64::from_le_bytes(buf))
    }

    /// Read an f64 value
    pub fn read_f64(file: &mut File) -> MinervaResult<f64> {
        let mut buf = [0u8; 8];
        file.read_exact(&mut buf)
            .map_err(|e| MinervaError::ModelLoadingError(e.to_string()))?;
        Ok(f64::from_le_bytes(buf))
    }

    /// Read a string value
    pub fn read_string(file: &mut File) -> MinervaResult<String> {
        let len = Self::read_u32(file)? as usize;
        let mut buf = vec![0u8; len];
        file.read_exact(&mut buf)
            .map_err(|e| MinervaError::ModelLoadingError(e.to_string()))?;
        String::from_utf8(buf)
            .map_err(|e| MinervaError::ModelLoadingError(format!("Invalid UTF-8 in string: {}", e)))
    }
}
