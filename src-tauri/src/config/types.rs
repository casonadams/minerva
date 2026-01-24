//! Configuration types and structures

use serde::{Deserialize, Serialize};

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
