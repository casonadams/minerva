//! Configuration validation

use super::types::{ApiConfig, ApplicationConfig, ServerConfig, StreamingConfigEntry};

/// Configuration validator
pub struct ConfigValidator;

impl ConfigValidator {
    /// Validate server configuration
    pub fn validate_server(config: &ServerConfig) -> Result<(), String> {
        if config.port == 0 {
            return Err("Port cannot be 0".to_string());
        }
        if config.host.is_empty() {
            return Err("Host cannot be empty".to_string());
        }
        Ok(())
    }

    /// Validate API configuration
    pub fn validate_api(config: &ApiConfig) -> Result<(), String> {
        if config.version.is_empty() {
            return Err("API version cannot be empty".to_string());
        }
        if config.request_timeout_ms == 0 {
            return Err("Request timeout must be greater than 0".to_string());
        }
        if config.max_tokens == 0 || config.max_tokens > 32768 {
            return Err("Max tokens must be between 1 and 32768".to_string());
        }
        Ok(())
    }

    /// Validate streaming configuration
    pub fn validate_streaming(config: &StreamingConfigEntry) -> Result<(), String> {
        if config.chunk_size == 0 {
            return Err("Chunk size must be greater than 0".to_string());
        }
        if config.chunk_size > 1000 {
            return Err("Chunk size must not exceed 1000".to_string());
        }
        if config.keep_alive_ms == 0 {
            return Err("Keep-alive interval must be greater than 0".to_string());
        }
        Ok(())
    }

    /// Validate complete configuration
    pub fn validate_all(config: &ApplicationConfig) -> Result<(), String> {
        Self::validate_server(&config.server)?;
        Self::validate_api(&config.api)?;
        Self::validate_streaming(&config.streaming)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::types::{ApiConfig, ApplicationConfig, ServerConfig, StreamingConfigEntry};

    #[test]
    fn test_validate_server_valid() {
        let config = ServerConfig::default();
        assert!(ConfigValidator::validate_server(&config).is_ok());
    }

    #[test]
    fn test_validate_server_invalid_port() {
        let config = ServerConfig {
            host: "localhost".to_string(),
            port: 0,
            workers: None,
        };
        assert!(ConfigValidator::validate_server(&config).is_err());
    }

    #[test]
    fn test_validate_api_valid() {
        let config = ApiConfig::default();
        assert!(ConfigValidator::validate_api(&config).is_ok());
    }

    #[test]
    fn test_validate_api_zero_timeout() {
        let config = ApiConfig {
            version: "1.0".to_string(),
            request_timeout_ms: 0,
            max_tokens: 256,
        };
        assert!(ConfigValidator::validate_api(&config).is_err());
    }

    #[test]
    fn test_validate_streaming_valid() {
        let config = StreamingConfigEntry::default();
        assert!(ConfigValidator::validate_streaming(&config).is_ok());
    }

    #[test]
    fn test_validate_all_config() {
        let config = ApplicationConfig::default();
        assert!(ConfigValidator::validate_all(&config).is_ok());
    }
}
