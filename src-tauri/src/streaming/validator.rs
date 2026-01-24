//! Streaming validation

/// Validates streaming parameters
pub struct StreamingValidator;

impl StreamingValidator {
    /// Validate stream parameter
    pub fn validate_stream(_stream: bool) -> Result<(), String> {
        // Stream is valid if it's a boolean (always valid)
        Ok(())
    }

    /// Validate streaming chunk size
    pub fn validate_chunk_size(size: usize) -> Result<(), String> {
        if size == 0 {
            return Err("Chunk size must be greater than 0".to_string());
        }
        if size > 1000 {
            return Err("Chunk size must not exceed 1000 tokens".to_string());
        }
        Ok(())
    }

    /// Check if streaming is compatible with model
    pub fn is_streaming_supported(model: &str) -> bool {
        // All models support streaming by default
        !model.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validator_chunk_size_valid() {
        assert!(StreamingValidator::validate_chunk_size(1).is_ok());
        assert!(StreamingValidator::validate_chunk_size(50).is_ok());
        assert!(StreamingValidator::validate_chunk_size(1000).is_ok());
    }

    #[test]
    fn test_validator_chunk_size_invalid() {
        assert!(StreamingValidator::validate_chunk_size(0).is_err());
        assert!(StreamingValidator::validate_chunk_size(1001).is_err());
    }

    #[test]
    fn test_validator_stream_always_valid() {
        assert!(StreamingValidator::validate_stream(true).is_ok());
        assert!(StreamingValidator::validate_stream(false).is_ok());
    }

    #[test]
    fn test_validator_streaming_supported() {
        assert!(StreamingValidator::is_streaming_supported("gpt-4"));
        assert!(StreamingValidator::is_streaming_supported("llama-2"));
        assert!(!StreamingValidator::is_streaming_supported(""));
    }
}
