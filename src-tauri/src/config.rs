use crate::error::{MinervaError, MinervaResult};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(dead_code)]
pub struct AppConfig {
    pub models_dir: PathBuf,
    pub server: ServerConfig,
    pub gpu: GpuConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    pub port: u16,
    pub host: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuConfig {
    pub enabled: bool,
    pub backend: String,
}

impl AppConfig {
    /// Load configuration from ~/.minerva/config.json
    pub fn load() -> MinervaResult<Self> {
        let config_path = Self::config_path()?;

        if config_path.exists() {
            let content = fs::read_to_string(&config_path).map_err(MinervaError::IoError)?;
            serde_json::from_str(&content).map_err(MinervaError::JsonError)
        } else {
            Ok(Self::default())
        }
    }

    #[allow(dead_code)]
    /// Load config or return defaults if not found
    pub fn load_or_default() -> Self {
        Self::load().unwrap_or_default()
    }

    #[allow(dead_code)]
    /// Save configuration to ~/.minerva/config.json
    pub fn save(&self) -> MinervaResult<()> {
        let config_path = Self::config_path()?;

        // Ensure config directory exists
        if let Some(parent) = config_path.parent() {
            fs::create_dir_all(parent).map_err(MinervaError::IoError)?;
        }

        // Ensure models directory exists
        fs::create_dir_all(&self.models_dir).map_err(MinervaError::IoError)?;

        let content = serde_json::to_string_pretty(self).map_err(MinervaError::JsonError)?;

        fs::write(&config_path, content).map_err(MinervaError::IoError)?;

        Ok(())
    }

    /// Get path to config file
    fn config_path() -> MinervaResult<PathBuf> {
        let home_dir = home::home_dir().ok_or_else(|| {
            MinervaError::ServerError("Could not determine home directory".to_string())
        })?;

        Ok(home_dir.join(".minerva").join("config.json"))
    }

    #[allow(dead_code)]
    /// Get models directory, creating it if it doesn't exist
    pub fn ensure_models_dir(&self) -> MinervaResult<()> {
        fs::create_dir_all(&self.models_dir).map_err(MinervaError::IoError)?;
        Ok(())
    }
}

impl Default for AppConfig {
    fn default() -> Self {
        let home = home::home_dir().unwrap_or_else(|| PathBuf::from("."));

        Self {
            models_dir: home.join(".minerva").join("models"),
            server: ServerConfig {
                port: 11434,
                host: "127.0.0.1".to_string(),
            },
            gpu: GpuConfig {
                enabled: true,
                backend: "metal".to_string(),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = AppConfig::default();
        assert_eq!(config.server.port, 11434);
        assert_eq!(config.server.host, "127.0.0.1");
        assert_eq!(config.gpu.backend, "metal");
        assert!(config.gpu.enabled);
    }

    #[test]
    fn test_config_paths_contain_minerva() {
        let config = AppConfig::default();
        let models_path = config.models_dir.to_string_lossy();
        assert!(models_path.contains(".minerva"));
        assert!(models_path.contains("models"));
    }

    #[test]
    fn test_config_serialization() {
        let config = AppConfig::default();
        let json = serde_json::to_string(&config);
        assert!(json.is_ok());

        let deserialized: Result<AppConfig, _> = serde_json::from_str(&json.unwrap());
        assert!(deserialized.is_ok());
    }
}
