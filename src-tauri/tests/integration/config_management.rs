// Configuration Management Tests

use minerva_lib::config_manager::{
    ApiConfig, ApplicationConfig, ConfigSource, ConfigValidator, ServerConfig, StreamingConfigEntry,
};

#[test]
fn test_config_source_default() {
    let config = ApplicationConfig::default();
    assert_eq!(config.source, ConfigSource::Default);
}

#[test]
fn test_server_config_valid() {
    let config = ServerConfig {
        host: "0.0.0.0".to_string(),
        port: 8080,
        workers: Some(4),
    };

    assert!(ConfigValidator::validate_server(&config).is_ok());
}

#[test]
fn test_server_config_localhost() {
    let config = ServerConfig {
        host: "localhost".to_string(),
        port: 3000,
        workers: None,
    };

    assert!(ConfigValidator::validate_server(&config).is_ok());
}

#[test]
fn test_server_config_invalid_port_zero() {
    let config = ServerConfig {
        host: "localhost".to_string(),
        port: 0,
        workers: None,
    };

    assert!(ConfigValidator::validate_server(&config).is_err());
}

#[test]
fn test_server_config_empty_host() {
    let config = ServerConfig {
        host: "".to_string(),
        port: 3000,
        workers: None,
    };

    assert!(ConfigValidator::validate_server(&config).is_err());
}

#[test]
fn test_server_config_high_port() {
    let config = ServerConfig {
        host: "localhost".to_string(),
        port: 65535,
        workers: Some(1),
    };

    assert!(ConfigValidator::validate_server(&config).is_ok());
}

#[test]
fn test_api_config_valid() {
    let config = ApiConfig {
        version: "1.0.0".to_string(),
        request_timeout_ms: 30000,
        max_tokens: 2048,
    };

    assert!(ConfigValidator::validate_api(&config).is_ok());
}

#[test]
fn test_api_config_boundary_tokens() {
    let config = ApiConfig {
        version: "1.0.0".to_string(),
        request_timeout_ms: 30000,
        max_tokens: 32768,
    };

    assert!(ConfigValidator::validate_api(&config).is_ok());
}

#[test]
fn test_api_config_invalid_timeout() {
    let config = ApiConfig {
        version: "1.0.0".to_string(),
        request_timeout_ms: 0,
        max_tokens: 256,
    };

    assert!(ConfigValidator::validate_api(&config).is_err());
}

#[test]
fn test_api_config_invalid_max_tokens_zero() {
    let config = ApiConfig {
        version: "1.0.0".to_string(),
        request_timeout_ms: 30000,
        max_tokens: 0,
    };

    assert!(ConfigValidator::validate_api(&config).is_err());
}

#[test]
fn test_api_config_invalid_max_tokens_over_limit() {
    let config = ApiConfig {
        version: "1.0.0".to_string(),
        request_timeout_ms: 30000,
        max_tokens: 32769,
    };

    assert!(ConfigValidator::validate_api(&config).is_err());
}

#[test]
fn test_api_config_empty_version() {
    let config = ApiConfig {
        version: "".to_string(),
        request_timeout_ms: 30000,
        max_tokens: 256,
    };

    assert!(ConfigValidator::validate_api(&config).is_err());
}

#[test]
fn test_streaming_config_valid() {
    let config = StreamingConfigEntry {
        enabled: true,
        chunk_size: 100,
        keep_alive_ms: 15000,
    };

    assert!(ConfigValidator::validate_streaming(&config).is_ok());
}

#[test]
fn test_streaming_config_disabled() {
    let config = StreamingConfigEntry {
        enabled: false,
        chunk_size: 50,
        keep_alive_ms: 15000,
    };

    assert!(ConfigValidator::validate_streaming(&config).is_ok());
}

#[test]
fn test_streaming_config_chunk_size_boundary() {
    let config = StreamingConfigEntry {
        enabled: true,
        chunk_size: 1000,
        keep_alive_ms: 15000,
    };

    assert!(ConfigValidator::validate_streaming(&config).is_ok());
}

#[test]
fn test_streaming_config_invalid_chunk_size_zero() {
    let config = StreamingConfigEntry {
        enabled: true,
        chunk_size: 0,
        keep_alive_ms: 15000,
    };

    assert!(ConfigValidator::validate_streaming(&config).is_err());
}

#[test]
fn test_streaming_config_invalid_chunk_size_over_limit() {
    let config = StreamingConfigEntry {
        enabled: true,
        chunk_size: 1001,
        keep_alive_ms: 15000,
    };

    assert!(ConfigValidator::validate_streaming(&config).is_err());
}

#[test]
fn test_streaming_config_invalid_keep_alive_zero() {
    let config = StreamingConfigEntry {
        enabled: true,
        chunk_size: 50,
        keep_alive_ms: 0,
    };

    assert!(ConfigValidator::validate_streaming(&config).is_err());
}

#[test]
fn test_application_config_all_defaults() {
    let config = ApplicationConfig::default();

    assert!(ConfigValidator::validate_all(&config).is_ok());
}

#[test]
fn test_application_config_invalid_server() {
    let config = ApplicationConfig {
        server: ServerConfig {
            host: "".to_string(),
            port: 3000,
            workers: None,
        },
        api: ApiConfig::default(),
        streaming: StreamingConfigEntry::default(),
        source: ConfigSource::Default,
    };

    assert!(ConfigValidator::validate_all(&config).is_err());
}

#[test]
fn test_application_config_invalid_api() {
    let config = ApplicationConfig {
        server: ServerConfig::default(),
        api: ApiConfig {
            version: "1.0".to_string(),
            request_timeout_ms: 0,
            max_tokens: 256,
        },
        streaming: StreamingConfigEntry::default(),
        source: ConfigSource::Default,
    };

    assert!(ConfigValidator::validate_all(&config).is_err());
}

#[test]
fn test_application_config_invalid_streaming() {
    let config = ApplicationConfig {
        server: ServerConfig::default(),
        api: ApiConfig::default(),
        streaming: StreamingConfigEntry {
            enabled: true,
            chunk_size: 0,
            keep_alive_ms: 15000,
        },
        source: ConfigSource::Default,
    };

    assert!(ConfigValidator::validate_all(&config).is_err());
}

#[test]
fn test_config_source_ordering() {
    assert!(ConfigSource::Default < ConfigSource::File);
    assert!(ConfigSource::File < ConfigSource::Environment);
    assert!(ConfigSource::Environment < ConfigSource::CommandLine);
}

#[test]
fn test_server_config_with_workers() {
    let config = ServerConfig {
        host: "127.0.0.1".to_string(),
        port: 3000,
        workers: Some(8),
    };

    assert_eq!(config.workers, Some(8));
    assert!(ConfigValidator::validate_server(&config).is_ok());
}

#[test]
fn test_multiple_config_combinations() {
    let configs = vec![
        ApplicationConfig {
            server: ServerConfig {
                host: "0.0.0.0".to_string(),
                port: 8000,
                workers: Some(4),
            },
            api: ApiConfig {
                version: "2.0".to_string(),
                request_timeout_ms: 60000,
                max_tokens: 8192,
            },
            streaming: StreamingConfigEntry {
                enabled: true,
                chunk_size: 100,
                keep_alive_ms: 30000,
            },
            source: ConfigSource::File,
        },
        ApplicationConfig {
            server: ServerConfig {
                host: "localhost".to_string(),
                port: 3000,
                workers: None,
            },
            api: ApiConfig::default(),
            streaming: StreamingConfigEntry::default(),
            source: ConfigSource::Environment,
        },
    ];

    for config in configs {
        assert!(ConfigValidator::validate_all(&config).is_ok());
    }
}

#[test]
fn test_config_serialization() {
    let config = ApplicationConfig::default();

    // Should be serializable
    let json = serde_json::to_string(&config).expect("Should serialize");
    assert!(json.contains("server"), "Should contain server");
    assert!(json.contains("api"), "Should contain api");
    assert!(json.contains("streaming"), "Should contain streaming");
}
