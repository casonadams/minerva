use crate::error::{MinervaError, MinervaResult};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};

/// Validates and reads GGUF file headers
pub struct GGUFHeaderValidator;

impl GGUFHeaderValidator {
    /// Validate GGUF file header and return version info
    pub fn validate(file: &mut File) -> MinervaResult<u32> {
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

        Ok(version)
    }

    /// Align file position to 32-byte boundary
    pub fn align_to_boundary(file: &mut File) -> MinervaResult<()> {
        let current_pos = file.stream_position().map_err(|e| {
            MinervaError::ModelLoadingError(format!("Failed to get file position: {}", e))
        })?;
        let aligned_pos = current_pos.div_ceil(32) * 32;
        file.seek(SeekFrom::Start(aligned_pos)).map_err(|e| {
            MinervaError::ModelLoadingError(format!("Failed to seek to alignment: {}", e))
        })?;
        Ok(())
    }

    fn read_u32(file: &mut File) -> MinervaResult<u32> {
        let mut buf = [0u8; 4];
        file.read_exact(&mut buf)
            .map_err(|e| MinervaError::ModelLoadingError(e.to_string()))?;
        Ok(u32::from_le_bytes(buf))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_validate_header_invalid_magic() {
        let mut file = NamedTempFile::new().unwrap();
        file.write_all(&[0x00, 0x00, 0x00, 0x00]).unwrap();
        file.flush().unwrap();

        let result = GGUFHeaderValidator::validate(&mut File::open(file.path()).unwrap());
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_header_valid() {
        let mut file = NamedTempFile::new().unwrap();
        file.write_all(&[0x47, 0x47, 0x55, 0x46]).unwrap();
        file.write_all(&2u32.to_le_bytes()).unwrap();
        file.flush().unwrap();

        let result = GGUFHeaderValidator::validate(&mut File::open(file.path()).unwrap());
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 2);
    }

    #[test]
    fn test_validate_header_unsupported_version() {
        let mut file = NamedTempFile::new().unwrap();
        file.write_all(&[0x47, 0x47, 0x55, 0x46]).unwrap();
        file.write_all(&99u32.to_le_bytes()).unwrap();
        file.flush().unwrap();

        let result = GGUFHeaderValidator::validate(&mut File::open(file.path()).unwrap());
        assert!(result.is_err());
    }
}
