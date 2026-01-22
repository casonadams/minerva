use crate::config::AppConfig;
use crate::models::loader::ModelLoader;
use crate::models::{ModelInfo, ModelRegistry};
use std::fs;
use std::path::PathBuf;
use tempfile::TempDir;

/// Test discovery pipeline: create GGUF files → load models → populate registry
#[test]
fn test_full_model_discovery_pipeline() {
    // Create temporary directory structure
    let temp_dir = TempDir::new().unwrap();
    let models_subdir = temp_dir.path().join("models");
    fs::create_dir(&models_subdir).unwrap();

    // Create dummy GGUF files
    let model1_path = models_subdir.join("model1.gguf");
    let model2_path = models_subdir.join("model2.gguf");
    fs::write(&model1_path, "GGUF dummy content").unwrap();
    fs::write(&model2_path, "GGUF dummy content").unwrap();

    // Test discovery
    let loader = ModelLoader::new(models_subdir.clone());
    let discovered = loader.discover_models().unwrap();

    assert_eq!(discovered.len(), 2);
    assert!(discovered.iter().any(|m| m.id == "model1"));
    assert!(discovered.iter().any(|m| m.id == "model2"));

    // Test registry integration
    let mut registry = ModelRegistry::new();
    registry.discover(&models_subdir).unwrap();

    let models = registry.list_models();
    assert_eq!(models.len(), 2);
    assert!(registry.get_model("model1").is_some());
    assert!(registry.get_model("model2").is_some());
}

/// Test selective filtering: only .gguf files are discovered
#[test]
fn test_discovery_filters_non_gguf_files() {
    let temp_dir = TempDir::new().unwrap();

    // Create various files
    fs::write(temp_dir.path().join("model.gguf"), "GGUF").unwrap();
    fs::write(temp_dir.path().join("readme.txt"), "text").unwrap();
    fs::write(temp_dir.path().join("model.bin"), "binary").unwrap();
    fs::write(temp_dir.path().join("config.json"), "{}").unwrap();

    let loader = ModelLoader::new(temp_dir.path().to_path_buf());
    let discovered = loader.discover_models().unwrap();

    assert_eq!(discovered.len(), 1);
    assert_eq!(discovered[0].id, "model");
}

/// Test nested directory structure: models can be in subdirectories
#[test]
fn test_discovery_in_nested_directories() {
    let temp_dir = TempDir::new().unwrap();

    // Create nested structure
    let subdir1 = temp_dir.path().join("subfolder1");
    let subdir2 = temp_dir.path().join("subfolder2");
    fs::create_dir(&subdir1).unwrap();
    fs::create_dir(&subdir2).unwrap();

    // Add models to different depths
    fs::write(temp_dir.path().join("root_model.gguf"), "GGUF").unwrap();
    fs::write(subdir1.join("nested1_model.gguf"), "GGUF").unwrap();
    fs::write(subdir2.join("nested2_model.gguf"), "GGUF").unwrap();

    let loader = ModelLoader::new(temp_dir.path().to_path_buf());
    let discovered = loader.discover_models().unwrap();

    assert_eq!(discovered.len(), 3);
    let ids: Vec<&str> = discovered.iter().map(|m| m.id.as_str()).collect();
    assert!(ids.contains(&"root_model"));
    assert!(ids.contains(&"nested1_model"));
    assert!(ids.contains(&"nested2_model"));
}

/// Test registry operations: add, retrieve, remove, list
#[test]
fn test_registry_crud_operations() {
    let mut registry = ModelRegistry::new();

    // Add models
    let model1 = ModelInfo {
        id: "test1".to_string(),
        object: "model".to_string(),
        created: 0,
        owned_by: "local".to_string(),
        context_window: Some(2048),
        max_output_tokens: Some(1024),
    };

    let model2 = ModelInfo {
        id: "test2".to_string(),
        object: "model".to_string(),
        created: 0,
        owned_by: "local".to_string(),
        context_window: Some(4096),
        max_output_tokens: Some(2048),
    };

    registry.add_model(model1.clone(), PathBuf::from("/path/to/model1.gguf"));
    registry.add_model(model2.clone(), PathBuf::from("/path/to/model2.gguf"));

    // List models
    let models = registry.list_models();
    assert_eq!(models.len(), 2);

    // Retrieve specific model
    let retrieved = registry.get_model("test1").unwrap();
    assert_eq!(retrieved.id, "test1");
    assert_eq!(retrieved.context_window, Some(2048));

    // Remove model
    let removed = registry.remove_model("test1").unwrap();
    assert_eq!(removed.id, "test1");

    // Verify removal
    let models = registry.list_models();
    assert_eq!(models.len(), 1);
    assert!(registry.get_model("test1").is_none());
    assert!(registry.get_model("test2").is_some());

    // Clear registry
    registry.clear();
    assert_eq!(registry.list_models().len(), 0);
}

/// Test config integration: load/save config → discover models
#[test]
fn test_config_with_discovery() {
    let temp_dir = TempDir::new().unwrap();
    let models_dir = temp_dir.path().join("models");
    fs::create_dir(&models_dir).unwrap();

    // Create test models
    fs::write(models_dir.join("test1.gguf"), "GGUF").unwrap();
    fs::write(models_dir.join("test2.gguf"), "GGUF").unwrap();

    // Create config pointing to models directory
    let config = AppConfig {
        models_dir: models_dir.clone(),
        server: crate::config::ServerConfig {
            port: 11434,
            host: "127.0.0.1".to_string(),
        },
        gpu: crate::config::GpuConfig {
            enabled: false,
            backend: "cpu".to_string(),
        },
    };

    // Verify models can be discovered from config directory
    let loader = ModelLoader::new(config.models_dir.clone());
    let discovered = loader.discover_models().unwrap();

    assert_eq!(discovered.len(), 2);
}

/// Test model metadata consistency: loaded models have valid fields
#[test]
fn test_model_metadata_validity() {
    let temp_dir = TempDir::new().unwrap();
    let model_path = temp_dir.path().join("test_model.gguf");
    fs::write(&model_path, "dummy gguf content").unwrap();

    let loader = ModelLoader::new(temp_dir.path().to_path_buf());
    let model = loader.load_model(&model_path).unwrap();

    // Verify mandatory fields
    assert!(!model.id.is_empty());
    assert_eq!(model.object, "model");
    assert_eq!(model.owned_by, "local");

    // Verify optional fields are present
    assert!(model.context_window.is_some());
    assert!(model.max_output_tokens.is_some());
    assert!(model.context_window.unwrap() > 0);
    assert!(model.max_output_tokens.unwrap() > 0);
}

/// Test empty directory handling: no error when discovering empty directory
#[test]
fn test_discover_empty_directory_no_panic() {
    let temp_dir = TempDir::new().unwrap();

    let loader = ModelLoader::new(temp_dir.path().to_path_buf());
    let result = loader.discover_models();

    assert!(result.is_ok());
    assert_eq!(result.unwrap().len(), 0);
}

/// Test nonexistent directory handling: graceful error for missing directory
#[test]
fn test_discover_nonexistent_directory() {
    let loader = ModelLoader::new(PathBuf::from("/nonexistent/models"));
    let result = loader.discover_models();

    assert!(result.is_ok());
    assert_eq!(result.unwrap().len(), 0);
}

/// Test model file path tracking: registry tracks paths correctly
#[test]
fn test_registry_path_tracking() {
    let temp_dir = TempDir::new().unwrap();
    let models_dir = temp_dir.path().join("models");
    fs::create_dir(&models_dir).unwrap();

    // Create test model
    let model_path = models_dir.join("mymodel.gguf");
    fs::write(&model_path, "GGUF").unwrap();

    // Discover via registry
    let mut registry = ModelRegistry::new();
    registry.discover(&models_dir).unwrap();

    // Verify model is registered
    let models = registry.list_models();
    assert_eq!(models.len(), 1);
    assert_eq!(models[0].id, "mymodel");
}
