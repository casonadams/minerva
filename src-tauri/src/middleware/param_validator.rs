use crate::error::{MinervaError, MinervaResult};

/// Validates individual parameters
pub struct ParamValidator;

impl ParamValidator {
    /// Validate temperature parameter [0, 2]
    pub fn temperature(temp: f32) -> MinervaResult<()> {
        if !(0.0..=2.0).contains(&temp) {
            return Err(MinervaError::ValidationError(format!(
                "Temperature must be between 0 and 2, got {}",
                temp
            )));
        }
        Ok(())
    }

    /// Validate top_p parameter (0, 1]
    pub fn top_p(top_p: f32) -> MinervaResult<()> {
        if top_p <= 0.0 || top_p > 1.0 {
            return Err(MinervaError::ValidationError(format!(
                "top_p must be between 0 (exclusive) and 1, got {}",
                top_p
            )));
        }
        Ok(())
    }

    /// Validate token count
    pub fn token_count(tokens: usize, max_tokens: usize) -> MinervaResult<()> {
        if tokens > max_tokens {
            return Err(MinervaError::ValidationError(format!(
                "Token count {} exceeds maximum of {}",
                tokens, max_tokens
            )));
        }
        Ok(())
    }

    /// Validate message role
    pub fn role(role: &str) -> MinervaResult<()> {
        match role {
            "user" | "assistant" | "system" => Ok(()),
            _ => Err(MinervaError::ValidationError(format!(
                "Invalid role '{}'. Must be 'user', 'assistant', or 'system'",
                role
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_temperature_valid() {
        assert!(ParamValidator::temperature(0.0).is_ok());
        assert!(ParamValidator::temperature(0.7).is_ok());
        assert!(ParamValidator::temperature(2.0).is_ok());
    }

    #[test]
    fn test_temperature_invalid() {
        assert!(ParamValidator::temperature(-0.1).is_err());
        assert!(ParamValidator::temperature(2.1).is_err());
    }

    #[test]
    fn test_top_p_valid() {
        assert!(ParamValidator::top_p(0.5).is_ok());
        assert!(ParamValidator::top_p(0.95).is_ok());
        assert!(ParamValidator::top_p(1.0).is_ok());
    }

    #[test]
    fn test_top_p_invalid() {
        assert!(ParamValidator::top_p(0.0).is_err());
        assert!(ParamValidator::top_p(1.1).is_err());
    }

    #[test]
    fn test_token_count_valid() {
        assert!(ParamValidator::token_count(500, 1000).is_ok());
    }

    #[test]
    fn test_token_count_exceeds() {
        assert!(ParamValidator::token_count(1001, 1000).is_err());
    }

    #[test]
    fn test_role_valid() {
        assert!(ParamValidator::role("user").is_ok());
        assert!(ParamValidator::role("assistant").is_ok());
        assert!(ParamValidator::role("system").is_ok());
    }

    #[test]
    fn test_role_invalid() {
        assert!(ParamValidator::role("admin").is_err());
    }
}
