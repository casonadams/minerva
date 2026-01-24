//! Configuration loading from files

use super::types::ApplicationConfig;
use super::validator::ConfigValidator;
use std::path::Path;

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
    use crate::config::types::ApplicationConfig;

    #[test]
    fn test_application_config_defaults() {
        let config = ApplicationConfig::default();
        assert_eq!(config.server.host, "127.0.0.1");
        assert_eq!(config.api.version, "0.1.0");
        assert!(config.streaming.enabled);
    }
}
