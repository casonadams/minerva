use minerva::inference::inference_backend_trait::{GenerationParams, InferenceBackend};
use minerva::inference::mock_backend::MockBackend;
use std::fs;
use tempfile::TempDir;

#[test]
fn test_mock_backend_creation() {
    let backend = MockBackend::new();
    assert!(!backend.is_loaded());
    assert_eq!(backend.context_size(), 0);
    assert!(backend.thread_count() > 0);
}

#[test]
fn test_mock_backend_load() {
    let temp_dir = TempDir::new().unwrap();
    let model_path = temp_dir.path().join("test.gguf");
    fs::write(&model_path, "dummy").unwrap();

    let mut backend = MockBackend::new();
    assert!(backend.load_model(&model_path, 2048).is_ok());
    assert!(backend.is_loaded());
    assert_eq!(backend.context_size(), 2048);
}

#[test]
fn test_mock_backend_unload() {
    let temp_dir = TempDir::new().unwrap();
    let model_path = temp_dir.path().join("test.gguf");
    fs::write(&model_path, "dummy").unwrap();

    let mut backend = MockBackend::new();
    assert!(backend.load_model(&model_path, 2048).is_ok());
    backend.unload_model();
    assert!(!backend.is_loaded());
}

#[test]
fn test_mock_backend_generate() {
    let temp_dir = TempDir::new().unwrap();
    let model_path = temp_dir.path().join("test.gguf");
    fs::write(&model_path, "dummy").unwrap();

    let mut backend = MockBackend::new();
    assert!(backend.load_model(&model_path, 2048).is_ok());

    let params = GenerationParams {
        max_tokens: 100,
        temperature: 0.7,
        top_p: 0.9,
    };
    let result = backend.generate("hello", params);
    assert!(result.is_ok());
    assert!(!result.unwrap().is_empty());
}

#[test]
fn test_mock_backend_generate_without_load() {
    let backend = MockBackend::new();
    let params = GenerationParams {
        max_tokens: 100,
        temperature: 0.7,
        top_p: 0.9,
    };
    let result = backend.generate("hello", params);
    assert!(result.is_err());
}

#[test]
fn test_mock_backend_tokenize() {
    let backend = MockBackend::new();
    let tokens = backend.tokenize("hello world test").unwrap();
    assert_eq!(tokens.len(), 3);
}

#[test]
fn test_mock_backend_detokenize() {
    let backend = MockBackend::new();
    let result = backend.detokenize(&[1, 2, 3]).unwrap();
    assert_eq!(result, "[3 tokens]");
}

#[test]
fn test_mock_backend_intelligent_response_hello() {
    let temp_dir = TempDir::new().unwrap();
    let model_path = temp_dir.path().join("test.gguf");
    fs::write(&model_path, "dummy").unwrap();

    let mut backend = MockBackend::new();
    backend.load_model(&model_path, 2048).unwrap();

    let params = GenerationParams {
        max_tokens: 100,
        temperature: 0.7,
        top_p: 0.9,
    };
    let result = backend.generate("hello", params).unwrap();
    assert!(result.contains("Hello") || result.contains("hello"));
}

#[test]
fn test_mock_backend_intelligent_response_what() {
    let temp_dir = TempDir::new().unwrap();
    let model_path = temp_dir.path().join("test.gguf");
    fs::write(&model_path, "dummy").unwrap();

    let mut backend = MockBackend::new();
    backend.load_model(&model_path, 2048).unwrap();

    let params = GenerationParams {
        max_tokens: 100,
        temperature: 0.7,
        top_p: 0.9,
    };
    let result = backend.generate("what is this?", params).unwrap();
    assert!(!result.is_empty());
}
