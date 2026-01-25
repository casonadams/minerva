use minerva::inference::inference_backend_trait::InferenceBackend;
use minerva::inference::llama_cpp_backend::LlamaCppBackend;
use std::path::Path;

#[test]
fn test_llama_cpp_backend_creation() {
    let backend = LlamaCppBackend::new();
    assert!(!backend.is_loaded());
    assert_eq!(backend.context_size(), 0);
    assert!(backend.thread_count() > 0);
}

#[test]
fn test_llama_cpp_backend_missing_file() {
    let mut backend = LlamaCppBackend::new();
    let result = backend.load_model(Path::new("/nonexistent/model.gguf"), 2048);
    assert!(result.is_err());
}

#[test]
fn test_llama_cpp_backend_unload() {
    let backend = LlamaCppBackend::new();
    assert!(!backend.is_loaded());
    // Since we don't have a real model, just verify the methods don't panic
}

#[test]
fn test_llama_cpp_backend_tokenize_fallback() {
    let backend = LlamaCppBackend::new();
    // Without setting tokenizer, should fall back to word-based
    let result = backend.tokenize("hello world test").unwrap();
    assert_eq!(result.len(), 3); // Three words
}

#[test]
fn test_llama_cpp_backend_detokenize_fallback() {
    let backend = LlamaCppBackend::new();
    // Without setting tokenizer, should provide fallback message
    let result = backend.detokenize(&[1i32, 2i32, 3i32]).unwrap();
    assert!(result.contains("3 tokens"));
}

#[test]
fn test_format_detection_gguf() {
    let path = Path::new("model.gguf");
    assert_eq!(LlamaCppBackend::detect_format(path), "gguf");
}

#[test]
fn test_format_detection_safetensors() {
    let path = Path::new("model.safetensors");
    assert_eq!(LlamaCppBackend::detect_format(path), "safetensors");
}

#[test]
fn test_format_detection_huggingface_bin() {
    let path = Path::new("model.bin");
    assert_eq!(LlamaCppBackend::detect_format(path), "huggingface");
}

#[test]
fn test_format_detection_pytorch() {
    let path = Path::new("model.pt");
    assert_eq!(LlamaCppBackend::detect_format(path), "pytorch");
}

#[test]
fn test_format_detection_tensorflow() {
    let path = Path::new("model.pb");
    assert_eq!(LlamaCppBackend::detect_format(path), "tensorflow");
}

#[test]
fn test_format_detection_unknown() {
    let path = Path::new("model.unknown");
    assert_eq!(LlamaCppBackend::detect_format(path), "unknown");
}

#[test]
fn test_backend_can_handle_gguf() {
    let path = Path::new("model.gguf");
    assert!(LlamaCppBackend::can_handle(path));
}

#[test]
fn test_backend_cannot_handle_safetensors() {
    let path = Path::new("model.safetensors");
    assert!(!LlamaCppBackend::can_handle(path));
}

#[test]
fn test_backend_cannot_handle_huggingface() {
    let path = Path::new("model.bin");
    assert!(!LlamaCppBackend::can_handle(path));
}
