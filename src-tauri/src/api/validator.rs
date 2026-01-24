//! API protocol validation

use super::types::ApiError;

/// API Protocol validation rules
pub struct ProtocolValidator;

impl ProtocolValidator {
    /// Validate model ID format
    pub fn validate_model_id(model_id: &str) -> Result<(), ApiError> {
        if model_id.is_empty() {
            return Err(ApiError {
                message: "Model ID cannot be empty".to_string(),
                code: "invalid_model_id".to_string(),
                type_: Some("invalid_request_error".to_string()),
                param: Some("model".to_string()),
            });
        }

        if model_id.len() > 256 {
            return Err(ApiError {
                message: "Model ID too long (max 256 characters)".to_string(),
                code: "invalid_model_id".to_string(),
                type_: Some("invalid_request_error".to_string()),
                param: Some("model".to_string()),
            });
        }

        Ok(())
    }

    /// Validate temperature parameter
    pub fn validate_temperature(temp: f32) -> Result<(), ApiError> {
        if !(0.0..=2.0).contains(&temp) {
            return Err(ApiError {
                message: "Temperature must be between 0 and 2".to_string(),
                code: "invalid_parameter".to_string(),
                type_: Some("invalid_request_error".to_string()),
                param: Some("temperature".to_string()),
            });
        }
        Ok(())
    }

    /// Validate max tokens
    pub fn validate_max_tokens(max_tokens: u32) -> Result<(), ApiError> {
        if max_tokens == 0 || max_tokens > 4096 {
            return Err(ApiError {
                message: "max_tokens must be between 1 and 4096".to_string(),
                code: "invalid_parameter".to_string(),
                type_: Some("invalid_request_error".to_string()),
                param: Some("max_tokens".to_string()),
            });
        }
        Ok(())
    }

    /// Validate top_p parameter
    pub fn validate_top_p(top_p: f32) -> Result<(), ApiError> {
        if !(0.0..=1.0).contains(&top_p) {
            return Err(ApiError {
                message: "top_p must be between 0 and 1".to_string(),
                code: "invalid_parameter".to_string(),
                type_: Some("invalid_request_error".to_string()),
                param: Some("top_p".to_string()),
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_model_id_valid() {
        let result = ProtocolValidator::validate_model_id("gpt-4");
        assert!(result.is_ok(), "Valid model ID should pass");
    }

    #[test]
    fn test_validate_model_id_empty() {
        let result = ProtocolValidator::validate_model_id("");
        assert!(result.is_err(), "Empty model ID should fail");
    }

    #[test]
    fn test_validate_temperature_valid() {
        assert!(ProtocolValidator::validate_temperature(0.5).is_ok());
        assert!(ProtocolValidator::validate_temperature(0.0).is_ok());
        assert!(ProtocolValidator::validate_temperature(2.0).is_ok());
    }

    #[test]
    fn test_validate_temperature_invalid() {
        assert!(ProtocolValidator::validate_temperature(-0.1).is_err());
        assert!(ProtocolValidator::validate_temperature(2.1).is_err());
    }

    #[test]
    fn test_validate_max_tokens_valid() {
        assert!(ProtocolValidator::validate_max_tokens(1).is_ok());
        assert!(ProtocolValidator::validate_max_tokens(4096).is_ok());
    }

    #[test]
    fn test_validate_max_tokens_invalid() {
        assert!(ProtocolValidator::validate_max_tokens(0).is_err());
        assert!(ProtocolValidator::validate_max_tokens(4097).is_err());
    }

    #[test]
    fn test_validate_top_p_valid() {
        assert!(ProtocolValidator::validate_top_p(0.0).is_ok());
        assert!(ProtocolValidator::validate_top_p(0.9).is_ok());
        assert!(ProtocolValidator::validate_top_p(1.0).is_ok());
    }

    #[test]
    fn test_validate_top_p_invalid() {
        assert!(ProtocolValidator::validate_top_p(-0.1).is_err());
        assert!(ProtocolValidator::validate_top_p(1.1).is_err());
    }
}
