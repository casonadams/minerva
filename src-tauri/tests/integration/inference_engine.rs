// Inference Engine Integration Tests

use minerva_lib::inference::GenerationConfig;
use std::fs;
use std::path::PathBuf;
use tempfile::TempDir;

fn setup_temp_model() -> (TempDir, PathBuf) {
    let temp_dir = TempDir::new().unwrap();
    let model_path = temp_dir.path().join("test_model.gguf");
    fs::write(&model_path, "dummy").unwrap();
    (temp_dir, model_path)
}

#[test]
fn test_inference_engine_lifecycle() {
    use minerva_lib::inference::InferenceEngine;

    let (_temp, model_path) = setup_temp_model();
    let mut engine = InferenceEngine::new(model_path);

    assert!(!engine.is_loaded());

    assert!(engine.load_model().is_ok());
    assert!(engine.is_loaded());

    engine.unload_model();
    assert!(!engine.is_loaded());
}

#[test]
fn test_inference_generate_response() {
    use minerva_lib::inference::InferenceEngine;

    let (_temp, model_path) = setup_temp_model();
    let mut engine = InferenceEngine::new(model_path);
    assert!(engine.load_model().is_ok());

    let response = engine.generate("Hello").unwrap();
    assert!(!response.is_empty());
    assert!(response.contains("Hello") || response.contains("local"));
}

#[test]
fn test_generation_config_validation_temperature() {
    let invalid = GenerationConfig {
        temperature: 3.0,
        ..Default::default()
    };
    assert!(invalid.validate().is_err());
}

#[test]
fn test_generation_config_validation_top_p() {
    let invalid = GenerationConfig {
        top_p: 1.5,
        ..Default::default()
    };
    assert!(invalid.validate().is_err());
}

#[test]
fn test_generation_config_validation_max_tokens() {
    let invalid = GenerationConfig {
        max_tokens: 0,
        ..Default::default()
    };
    assert!(invalid.validate().is_err());
}

#[test]
fn test_generation_config_validation_valid() {
    let config = GenerationConfig::default();
    assert!(config.validate().is_ok());
}

#[test]
fn test_inference_context_info() {
    use minerva_lib::inference::InferenceEngine;

    let (_temp, model_path) = setup_temp_model();
    let mut engine = InferenceEngine::new(model_path);
    assert!(engine.load_model().is_ok());

    let info = engine.get_model_info().unwrap();
    assert_eq!(info.context_window, 2048);
    assert_eq!(info.vocab_size, 32000);
}
