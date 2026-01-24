use super::GenerationConfig;
use crate::error::{MinervaError, MinervaResult};

/// Validates individual parameter ranges
pub struct ParameterValidator;

impl ParameterValidator {
    /// Validate temperature range
    pub fn validate_temperature(temp: f32) -> MinervaResult<()> {
        if !(0.0..=2.0).contains(&temp) {
            return Err(MinervaError::InvalidRequest(format!(
                "temperature must be between 0.0 and 2.0, got {}",
                temp
            )));
        }
        Ok(())
    }

    /// Validate top_p range
    pub fn validate_top_p(top_p: f32) -> MinervaResult<()> {
        if !(0.0..=1.0).contains(&top_p) {
            return Err(MinervaError::InvalidRequest(format!(
                "top_p must be between 0.0 and 1.0, got {}",
                top_p
            )));
        }
        Ok(())
    }

    /// Validate frequency penalty range
    pub fn validate_frequency_penalty(penalty: f32) -> MinervaResult<()> {
        if !(-2.0..=2.0).contains(&penalty) {
            return Err(MinervaError::InvalidRequest(format!(
                "frequency_penalty must be between -2.0 and 2.0, got {}",
                penalty
            )));
        }
        Ok(())
    }

    /// Validate max tokens range
    pub fn validate_max_tokens(tokens: usize) -> MinervaResult<()> {
        if !(1..=32768).contains(&tokens) {
            return Err(MinervaError::InvalidRequest(format!(
                "max_tokens must be between 1 and 32768, got {}",
                tokens
            )));
        }
        Ok(())
    }
}

/// Apply validated parameters to config
pub struct ParameterApplier;

impl ParameterApplier {
    /// Apply temperature to config
    pub fn apply_temperature(config: &mut GenerationConfig, temp: f32) -> MinervaResult<()> {
        ParameterValidator::validate_temperature(temp)?;
        config.temperature = temp;
        Ok(())
    }

    /// Apply top_p to config
    pub fn apply_top_p(config: &mut GenerationConfig, top_p: f32) -> MinervaResult<()> {
        ParameterValidator::validate_top_p(top_p)?;
        config.top_p = top_p;
        Ok(())
    }

    /// Apply frequency penalty to config
    pub fn apply_frequency_penalty(
        config: &mut GenerationConfig,
        penalty: f32,
    ) -> MinervaResult<()> {
        ParameterValidator::validate_frequency_penalty(penalty)?;
        config.repeat_penalty = 1.0 + (penalty / 10.0);
        Ok(())
    }

    /// Apply max tokens to config
    pub fn apply_max_tokens(config: &mut GenerationConfig, tokens: usize) -> MinervaResult<()> {
        ParameterValidator::validate_max_tokens(tokens)?;
        config.max_tokens = tokens;
        Ok(())
    }
}
