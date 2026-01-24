//! Configuration management module
//! Handles loading, validation, and merging of configuration from multiple sources

pub mod legacy;
pub mod loader;
pub mod types;
pub mod validator;

pub use legacy::{AppConfig, GpuConfig, LegacyServerConfig};
pub use loader::ConfigLoader;
pub use types::{ApiConfig, ApplicationConfig, ConfigSource, ServerConfig, StreamingConfigEntry};
pub use validator::ConfigValidator;
