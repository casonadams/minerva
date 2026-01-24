use crate::error::{MinervaError, MinervaResult};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};

use super::gguf_tensor::{GGUFDataType, GGUFTensor, GGUFTensorData};

/// Loads GGUF tensor data from file
pub struct GGUFTensorLoader;

impl GGUFTensorLoader {
    /// Load a single tensor from file
    pub fn load_tensor(file: &mut File) -> MinervaResult<GGUFTensor> {
        // Read tensor name
        let name = Self::read_string(file)?;

        // Read tensor dimensions
        let n_dims = Self::read_u32(file)? as usize;
        let mut shape = Vec::with_capacity(n_dims);
        for _ in 0..n_dims {
            shape.push(Self::read_u64(file)?);
        }

        // Read data type
        let dtype_u32 = Self::read_u32(file)?;
        let data_type = GGUFDataType::from_u32(dtype_u32).ok_or_else(|| {
            MinervaError::ModelLoadingError(format!("Unknown data type: {}", dtype_u32))
        })?;

        // Read data offset
        let data_offset = Self::read_u64(file)?;

        // Calculate expected data size
        let element_count: u64 = shape.iter().product();
        let expected_size = data_type.total_size(element_count as usize);

        // Save current position and read data
        let current_pos = file.stream_position().map_err(|e| {
            MinervaError::ModelLoadingError(format!("Failed to get position: {}", e))
        })?;

        // Seek to data offset
        file.seek(SeekFrom::Start(data_offset)).map_err(|e| {
            MinervaError::ModelLoadingError(format!("Failed to seek to tensor data: {}", e))
        })?;

        // Read tensor data
        let mut data = vec![0u8; expected_size];
        file.read_exact(&mut data).map_err(|e| {
            MinervaError::ModelLoadingError(format!("Failed to read tensor data: {}", e))
        })?;

        // Return to next tensor metadata position
        file.seek(SeekFrom::Start(current_pos))
            .map_err(|e| MinervaError::ModelLoadingError(format!("Failed to seek back: {}", e)))?;

        let tensor_data = GGUFTensorData {
            name,
            data_type,
            shape,
            data,
        };
        Ok(GGUFTensor::new(tensor_data))
    }

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
    fn test_tensor_loader_exists() {
        // Placeholder test - actual tensor loading is tested in integration tests
        let _ = GGUFTensorLoader;
    }
}
