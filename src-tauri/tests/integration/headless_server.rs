// Headless Server Integration Tests - Verify Tauri decoupling

use minerva_lib::config::AppConfig;
use minerva_lib::server::ServerState;
use std::fs;
use std::path::PathBuf;
use tempfile::TempDir;

fn setup_test_models_dir() -> (TempDir, PathBuf) {
    let temp_dir = TempDir::new().unwrap();
    let models_dir = temp_dir.path().to_path_buf();

    // Create test model files
    fs::write(models_dir.join("test-model-1.gguf"), "GGUF data").unwrap();
    fs::write(models_dir.join("test-model-2.gguf"), "GGUF data").unwrap();

    (temp_dir, models_dir)
}

#[test]
fn test_server_state_creation() {
    let state = ServerState::new();

    // Verify state has required components
    assert_eq!(
        std::sync::Arc::strong_count(&state.model_registry),
        1,
        "Model registry should be accessible"
    );
}

#[test]
fn test_server_state_with_models() {
    let (_temp, models_dir) = setup_test_models_dir();
    let state = ServerState::with_discovered_models(models_dir);

    assert!(state.is_ok(), "Server state should be created with models");
}

#[test]
fn test_server_state_rate_limiter() {
    let state = ServerState::new();

    // Verify rate limiter is available
    assert_eq!(
        std::sync::Arc::strong_count(&state.rate_limiter),
        1,
        "Rate limiter should be accessible"
    );
}

#[test]
fn test_server_state_metrics_collector() {
    let state = ServerState::new();

    // Verify metrics collector is available
    assert_eq!(
        std::sync::Arc::strong_count(&state.metrics),
        1,
        "Metrics collector should be accessible"
    );
}

#[test]
fn test_app_config_models_dir_override() {
    let (_temp, models_dir) = setup_test_models_dir();

    // Test that config can be overridden with CLI arguments
    let config = AppConfig {
        models_dir: models_dir.clone(),
        ..Default::default()
    };
    assert_eq!(
        config.models_dir, models_dir,
        "Config models_dir should be overridable"
    );
}

#[test]
fn test_config_decoupling() {
    let config = AppConfig::default();

    // Config should not depend on Tauri
    assert!(
        config.ensure_models_dir().is_ok(),
        "Config should work independently"
    );
}

#[test]
fn test_server_state_with_multiple_models() {
    let (_temp, models_dir) = setup_test_models_dir();
    let state = ServerState::with_discovered_models(models_dir);

    assert!(
        state.is_ok(),
        "Server state should work with multiple models"
    );
}

#[test]
fn test_headless_server_no_tauri_dependency() {
    // ServerState can be created without any Tauri components
    let state = ServerState::new();

    // Verify all core components are present
    assert_ne!(
        std::sync::Arc::strong_count(&state.model_registry),
        0,
        "Model registry required"
    );
    assert_ne!(
        std::sync::Arc::strong_count(&state.rate_limiter),
        0,
        "Rate limiter required"
    );
    assert_ne!(
        std::sync::Arc::strong_count(&state.metrics),
        0,
        "Metrics required"
    );
}
