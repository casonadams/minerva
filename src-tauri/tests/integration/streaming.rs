// Token Streaming and Backend Abstraction Integration Tests

use minerva_lib::inference::llama_adapter::InferenceBackend;
use std::fs;
use std::path::PathBuf;
use tempfile::TempDir;

fn setup_temp_model() -> (TempDir, PathBuf) {
    let temp_dir = TempDir::new().unwrap();
    let model_path = temp_dir.path().join("test_model.gguf");
    fs::write(&model_path, "dummy").unwrap();
    (temp_dir, model_path)
}

// Token Streaming Tests

#[test]
fn test_token_stream_collection() {
    use minerva_lib::inference::token_stream::TokenStream;

    let stream = TokenStream::new();
    let total = stream.total_tokens();
    assert_eq!(total, 0);

    let string_repr = stream.to_string();
    assert_eq!(string_repr, "");
}

#[test]
fn test_token_stream_iteration() {
    use minerva_lib::inference::token_stream::TokenStream;

    let mut stream = TokenStream::new();
    stream.push_token("a".to_string());
    stream.push_token("b".to_string());
    stream.push_token("c".to_string());

    assert_eq!(stream.total_tokens(), 3);
    let first = stream.next_token();
    assert_eq!(first, Some("a".to_string()));
}

#[test]
fn test_token_stream_reset() {
    use minerva_lib::inference::token_stream::TokenStream;

    let stream = TokenStream::new();
    stream.push_token("test".to_string());

    let mut stream = stream;
    stream.next_token();
    assert_eq!(stream.position(), 1);
    stream.reset();
    assert_eq!(stream.position(), 0);
}

#[test]
fn test_token_stream_callback_streaming() {
    use minerva_lib::inference::token_stream::TokenStream;
    use std::sync::{Arc, Mutex};

    let callback_count = Arc::new(Mutex::new(0));
    let callback_count_clone = callback_count.clone();

    let callback: Arc<dyn Fn(String) + Send + Sync> = Arc::new(move |_token: String| {
        *callback_count_clone.lock().unwrap() += 1;
    });

    let stream = TokenStream::with_callback(callback);
    stream.push_token("test".to_string());
    stream.push_token("token".to_string());

    assert_eq!(*callback_count.lock().unwrap(), 2);
}

#[test]
fn test_token_stream_callback_with_content() {
    use minerva_lib::inference::token_stream::TokenStream;
    use std::sync::{Arc, Mutex};

    let collected = Arc::new(Mutex::new(Vec::new()));
    let collected_clone = collected.clone();

    let callback: Arc<dyn Fn(String) + Send + Sync> = Arc::new(move |token: String| {
        collected_clone.lock().unwrap().push(token);
    });

    let stream = TokenStream::with_callback(callback);
    stream.push_token("Hello".to_string());
    stream.push_token(" World".to_string());

    let result = collected.lock().unwrap();
    assert_eq!(result.len(), 2);
    assert_eq!(result[0], "Hello");
    assert_eq!(result[1], " World");
}

// Backend Abstraction Tests

#[test]
fn test_mock_backend_generation() {
    use minerva_lib::inference::llama_adapter::MockBackend;

    let (_temp, model_path) = setup_temp_model();
    let mut backend = MockBackend::new();

    assert!(backend.load_model(&model_path, 2048).is_ok());
    assert!(backend.is_loaded());

    let result = backend.generate("test", 100, 0.7, 0.9);
    assert!(result.is_ok());
}

#[test]
fn test_mock_backend_tokenization() {
    use minerva_lib::inference::llama_adapter::MockBackend;

    let backend = MockBackend::new();
    let tokens = backend.tokenize("hello world").unwrap();
    assert!(!tokens.is_empty());
}

#[test]
fn test_mock_backend_detokenize() {
    use minerva_lib::inference::llama_adapter::MockBackend;

    let backend = MockBackend::new();
    let tokens = vec![1, 2, 3];
    let text = backend.detokenize(&tokens).unwrap();
    assert!(!text.is_empty());
}

#[test]
fn test_mock_backend_lifecycle() {
    use minerva_lib::inference::llama_adapter::MockBackend;

    let (_temp, model_path) = setup_temp_model();
    let mut backend = MockBackend::new();

    assert!(!backend.is_loaded());
    assert!(backend.load_model(&model_path, 2048).is_ok());
    assert!(backend.is_loaded());

    backend.unload_model();
    assert!(!backend.is_loaded());
}

#[test]
fn test_backend_with_streaming() {
    use minerva_lib::inference::llama_adapter::MockBackend;
    use minerva_lib::inference::token_stream::TokenStream;
    use std::sync::{Arc, Mutex};

    let (_temp, model_path) = setup_temp_model();
    let mut backend = MockBackend::new();
    assert!(backend.load_model(&model_path, 2048).is_ok());

    let token_count = Arc::new(Mutex::new(0));
    let token_count_clone = token_count.clone();
    let callback: Arc<dyn Fn(String) + Send + Sync> = Arc::new(move |_: String| {
        *token_count_clone.lock().unwrap() += 1;
    });

    let _stream = TokenStream::with_callback(callback);

    let result = backend.generate("prompt", 10, 0.7, 0.9);
    assert!(result.is_ok());
    drop(_stream);
}

#[test]
fn test_streaming_with_inference_backend() {
    use minerva_lib::inference::llama_adapter::MockBackend;
    use std::sync::{Arc, Mutex};

    let (_temp, model_path) = setup_temp_model();
    let mut backend = MockBackend::new();
    assert!(backend.load_model(&model_path, 2048).is_ok());

    let collected = Arc::new(Mutex::new(String::new()));
    let collected_clone = collected.clone();

    let result = backend.generate("test", 5, 0.7, 0.9).unwrap();
    assert!(!result.is_empty());

    collected_clone.lock().unwrap().push_str(&result);
    assert!(!collected.lock().unwrap().is_empty());
}
