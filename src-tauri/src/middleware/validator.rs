use super::param_validator::ParamValidator;
use crate::error::{MinervaError, MinervaResult};

/// Input validation for requests
pub struct Validator;

impl Validator {
    /// Validate prompt length and content
    pub fn prompt(prompt: &str, max_length: usize) -> MinervaResult<()> {
        if prompt.is_empty() {
            return Err(MinervaError::ValidationError(
                "Prompt cannot be empty".to_string(),
            ));
        }
        if prompt.len() > max_length {
            return Err(MinervaError::ValidationError(format!(
                "Prompt exceeds maximum length of {} characters (got {})",
                max_length,
                prompt.len()
            )));
        }
        Ok(())
    }

    /// Validate model ID format
    pub fn model_id(model_id: &str) -> MinervaResult<()> {
        if model_id.is_empty() {
            return Err(MinervaError::ValidationError(
                "Model ID cannot be empty".to_string(),
            ));
        }
        if model_id.len() > 255 {
            return Err(MinervaError::ValidationError(
                "Model ID exceeds maximum length of 255 characters".to_string(),
            ));
        }
        if !model_id
            .chars()
            .all(|c| c.is_alphanumeric() || c == '-' || c == '_' || c == '/' || c == '.')
        {
            return Err(MinervaError::ValidationError(
                "Model ID contains invalid characters".to_string(),
            ));
        }
        Ok(())
    }

    /// Validate message content
    pub fn content(content: &str) -> MinervaResult<()> {
        if content.is_empty() {
            return Err(MinervaError::ValidationError(
                "Message content cannot be empty".to_string(),
            ));
        }
        Ok(())
    }

    /// Validate temperature with delegation
    pub fn temperature(temp: f32) -> MinervaResult<()> {
        ParamValidator::temperature(temp)
    }

    /// Validate top_p with delegation
    pub fn top_p(top_p: f32) -> MinervaResult<()> {
        ParamValidator::top_p(top_p)
    }

    /// Validate token count with delegation
    pub fn token_count(tokens: usize, max: usize) -> MinervaResult<()> {
        ParamValidator::token_count(tokens, max)
    }

    /// Validate role with delegation
    pub fn role(role: &str) -> MinervaResult<()> {
        ParamValidator::role(role)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prompt_valid() {
        assert!(Validator::prompt("Hello world", 100).is_ok());
    }

    #[test]
    fn test_prompt_empty() {
        assert!(Validator::prompt("", 100).is_err());
    }

    #[test]
    fn test_prompt_too_long() {
        let long = "a".repeat(101);
        assert!(Validator::prompt(&long, 100).is_err());
    }

    #[test]
    fn test_model_id_valid() {
        assert!(Validator::model_id("llama-2-7b").is_ok());
        assert!(Validator::model_id("gpt-3.5-turbo").is_ok());
        assert!(Validator::model_id("meta-llama/Llama-2-7b").is_ok());
    }

    #[test]
    fn test_model_id_empty() {
        assert!(Validator::model_id("").is_err());
    }

    #[test]
    fn test_model_id_invalid_chars() {
        assert!(Validator::model_id("model@invalid").is_err());
    }

    #[test]
    fn test_model_id_too_long() {
        let long = "a".repeat(256);
        assert!(Validator::model_id(&long).is_err());
    }

    #[test]
    fn test_content_valid() {
        assert!(Validator::content("Hello").is_ok());
    }

    #[test]
    fn test_content_empty() {
        assert!(Validator::content("").is_err());
    }
}
