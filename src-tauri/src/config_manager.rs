//! Configuration Management
//! Handles loading, validation, and merging of configuration from multiple sources

use serde::{Deserialize, Serialize};
use std::path::Path;

/// Configuration source priority (higher = more important)
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default)]
pub enum ConfigSource {
    /// Default configuration
    #[default]
    Default = 0,
    /// Configuration file (.toml, .json, .yaml)
    File = 1,
    /// Environment variables
    Environment = 2,
    /// Command-line arguments
    CommandLine = 3,
}

/// Server configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
    pub workers: Option<usize>,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".to_string(),
            port: 3000,
            workers: None,
        }
    }
}

/// API configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiConfig {
    pub version: String,
    pub request_timeout_ms: u64,
    pub max_tokens: u32,
}

impl Default for ApiConfig {
    fn default() -> Self {
        Self {
            version: "0.1.0".to_string(),
            request_timeout_ms: 30000,
            max_tokens: 4096,
        }
    }
}

/// Streaming configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamingConfigEntry {
    pub enabled: bool,
    pub chunk_size: usize,
    pub keep_alive_ms: u64,
}

impl Default for StreamingConfigEntry {
    fn default() -> Self {
        Self {
            enabled: true,
            chunk_size: 50,
            keep_alive_ms: 15000,
        }
    }
}

/// Complete application configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApplicationConfig {
    pub server: ServerConfig,
    pub api: ApiConfig,
    pub streaming: StreamingConfigEntry,
    #[serde(skip)]
    pub source: ConfigSource,
}

impl Default for ApplicationConfig {
    fn default() -> Self {
        Self {
            server: ServerConfig::default(),
            api: ApiConfig::default(),
            streaming: StreamingConfigEntry::default(),
            source: ConfigSource::Default,
        }
    }
}

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

/// Configuration loader
pub struct ConfigLoader;

impl ConfigLoader {
    /// Load configuration from JSON file
    pub fn load_json(path: &Path) -> Result<ApplicationConfig, String> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| format!("Failed to read config file: {}", e))?;

        let config: ApplicationConfig =
            serde_json::from_str(&content).map_err(|e| format!("Failed to parse JSON: {}", e))?;

        ConfigValidator::validate_all(&config)?;
        Ok(config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_server_config_defaults() {
        let config = ServerConfig::default();
        assert_eq!(config.host, "127.0.0.1");
        assert_eq!(config.port, 3000);
        assert!(config.workers.is_none());
    }

    #[test]
    fn test_api_config_defaults() {
        let config = ApiConfig::default();
        assert_eq!(config.version, "0.1.0");
        assert_eq!(config.request_timeout_ms, 30000);
        assert_eq!(config.max_tokens, 4096);
    }

    #[test]
    fn test_streaming_config_defaults() {
        let config = StreamingConfigEntry::default();
        assert!(config.enabled);
        assert_eq!(config.chunk_size, 50);
        assert_eq!(config.keep_alive_ms, 15000);
    }

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
    fn test_validate_server_empty_host() {
        let config = ServerConfig {
            host: "".to_string(),
            port: 3000,
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
    fn test_validate_api_invalid_max_tokens() {
        let config = ApiConfig {
            version: "1.0".to_string(),
            request_timeout_ms: 30000,
            max_tokens: 0,
        };
        assert!(ConfigValidator::validate_api(&config).is_err());
    }

    #[test]
    fn test_validate_streaming_valid() {
        let config = StreamingConfigEntry::default();
        assert!(ConfigValidator::validate_streaming(&config).is_ok());
    }

    #[test]
    fn test_validate_streaming_chunk_size() {
        let config = StreamingConfigEntry {
            enabled: true,
            chunk_size: 0,
            keep_alive_ms: 15000,
        };
        assert!(ConfigValidator::validate_streaming(&config).is_err());
    }

    #[test]
    fn test_config_source_ordering() {
        assert!(ConfigSource::Default < ConfigSource::File);
        assert!(ConfigSource::File < ConfigSource::Environment);
        assert!(ConfigSource::Environment < ConfigSource::CommandLine);
    }

    #[test]
    fn test_application_config_defaults() {
        let config = ApplicationConfig::default();
        assert_eq!(config.server.host, "127.0.0.1");
        assert_eq!(config.api.version, "0.1.0");
        assert!(config.streaming.enabled);
    }

    #[test]
    fn test_validate_all_config() {
        let config = ApplicationConfig::default();
        assert!(ConfigValidator::validate_all(&config).is_ok());
    }
}
