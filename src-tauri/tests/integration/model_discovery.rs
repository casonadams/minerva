// Model Discovery and Registry Integration Tests

use minerva_lib::models::loader::ModelLoader;
use std::fs;
use tempfile::TempDir;

fn setup_test_models_dir() -> (TempDir, std::path::PathBuf) {
    let temp_dir = TempDir::new().unwrap();
    let models_dir = temp_dir.path().join("models");
    fs::create_dir(&models_dir).unwrap();
    (temp_dir, models_dir)
}

fn create_dummy_gguf(dir: &std::path::Path, name: &str) -> std::path::PathBuf {
    let path = dir.join(format!("{}.gguf", name));
    fs::write(&path, "GGUF dummy content").unwrap();
    path
}

#[test]
fn test_model_discovery_basic() {
    let (_temp, models_dir) = setup_test_models_dir();
    create_dummy_gguf(&models_dir, "model1");

    let loader = ModelLoader::new(models_dir);
    let discovered = loader.discover_models().unwrap();

    assert_eq!(discovered.len(), 1);
    assert_eq!(discovered[0].id, "model1");
}

#[test]
fn test_model_discovery_multiple() {
    let (_temp, models_dir) = setup_test_models_dir();
    create_dummy_gguf(&models_dir, "model1");
    create_dummy_gguf(&models_dir, "model2");
    create_dummy_gguf(&models_dir, "model3");

    let loader = ModelLoader::new(models_dir);
    let discovered = loader.discover_models().unwrap();

    assert_eq!(discovered.len(), 3);
}

#[test]
fn test_model_discovery_filters_non_gguf() {
    let (_temp, models_dir) = setup_test_models_dir();
    create_dummy_gguf(&models_dir, "valid");
    fs::write(models_dir.join("invalid.txt"), "not a model").unwrap();

    let loader = ModelLoader::new(models_dir);
    let discovered = loader.discover_models().unwrap();

    assert_eq!(discovered.len(), 1);
    assert_eq!(discovered[0].id, "valid");
}

#[test]
fn test_model_registry_discovery() {
    use minerva_lib::models::ModelRegistry;

    let (_temp, models_dir) = setup_test_models_dir();
    create_dummy_gguf(&models_dir, "model1");
    create_dummy_gguf(&models_dir, "model2");

    let mut registry = ModelRegistry::new();
    registry.discover(&models_dir).unwrap();
    let models = registry.list_models();

    assert_eq!(models.len(), 2);
}
