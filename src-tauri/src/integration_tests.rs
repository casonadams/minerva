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

/// Test chat completion workflow: request → parameter parsing → response
#[test]
fn test_chat_completion_workflow() {
    use crate::inference::parameters::ParameterParser;
    use crate::models::{ChatCompletionRequest, ChatMessage};

    // Create a chat request
    let request = ChatCompletionRequest {
        model: "test-model".to_string(),
        messages: vec![ChatMessage {
            role: "user".to_string(),
            content: "What is Rust?".to_string(),
        }],
        temperature: Some(0.7),
        max_tokens: Some(256),
        stream: Some(false),
        top_p: Some(0.9),
        frequency_penalty: Some(0.0),
        presence_penalty: None,
    };

    // Parse parameters
    let config = ParameterParser::from_request(&request).unwrap();

    // Verify parameter extraction
    assert_eq!(config.temperature, 0.7);
    assert_eq!(config.max_tokens, 256);
    assert_eq!(config.top_p, 0.9);
}

/// Test parameter validation across valid range
#[test]
fn test_parameter_validation_boundary_conditions() {
    use crate::inference::parameters::ParameterParser;
    use crate::models::{ChatCompletionRequest, ChatMessage};

    let make_request = |temp: f32, max_tokens: usize| ChatCompletionRequest {
        model: "test".to_string(),
        messages: vec![ChatMessage {
            role: "user".to_string(),
            content: "test".to_string(),
        }],
        temperature: Some(temp),
        max_tokens: Some(max_tokens),
        stream: None,
        top_p: None,
        frequency_penalty: None,
        presence_penalty: None,
    };

    // Test minimum values
    let req = make_request(0.0, 1);
    assert!(ParameterParser::from_request(&req).is_ok());

    // Test maximum values
    let req = make_request(2.0, 32768);
    assert!(ParameterParser::from_request(&req).is_ok());

    // Test mid-range values
    let req = make_request(1.0, 512);
    assert!(ParameterParser::from_request(&req).is_ok());

    // Test just outside boundaries
    let req = make_request(2.1, 32768);
    assert!(ParameterParser::from_request(&req).is_err());

    let req = make_request(2.0, 32769);
    assert!(ParameterParser::from_request(&req).is_err());
}

/// Test streaming response format validation
#[test]
fn test_streaming_response_format() {
    use crate::inference::streaming::StreamingResponse;

    let response = StreamingResponse::new("gpt-3.5-turbo".to_string());

    // Create a token chunk
    let chunk = response.chunk("hello", 0);
    assert_eq!(chunk.object, "chat.completion.chunk");
    assert_eq!(chunk.model, "gpt-3.5-turbo");
    assert_eq!(chunk.choices[0].delta.content, Some("hello".to_string()));

    // Convert to SSE format
    let sse = StreamingResponse::to_sse_string(&chunk);
    assert!(sse.starts_with("data: {"));
    assert!(sse.ends_with("\n\n"));
    assert!(sse.contains("chat.completion.chunk"));
}

/// Test context manager with multiple model loading
#[test]
fn test_context_manager_multi_model_workflow() {
    use crate::inference::context_manager::ContextManager;
    use tempfile::TempDir;

    let temp_dir = TempDir::new().unwrap();
    let model1 = temp_dir.path().join("model1.gguf");
    let model2 = temp_dir.path().join("model2.gguf");
    fs::write(&model1, "test1").unwrap();
    fs::write(&model2, "test2").unwrap();

    let mut manager = ContextManager::new(2);

    // Load first model
    assert_eq!(manager.loaded_count(), 0);
    let result = manager.load_model("model1", model1);
    assert!(result.is_ok());
    assert_eq!(manager.loaded_count(), 1);

    // Load second model
    let result = manager.load_model("model2", model2);
    assert!(result.is_ok());
    assert_eq!(manager.loaded_count(), 2);

    // Check models are loaded
    assert!(manager.is_loaded("model1"));
    assert!(manager.is_loaded("model2"));

    let loaded = manager.get_loaded_models();
    assert_eq!(loaded.len(), 2);
    assert!(loaded.contains(&"model1".to_string()));
    assert!(loaded.contains(&"model2".to_string()));
}

/// Test error handling for invalid requests
#[test]
fn test_invalid_request_error_handling() {
    use crate::inference::parameters::ParameterParser;
    use crate::models::{ChatCompletionRequest, ChatMessage};

    // Create request with invalid temperature
    let request = ChatCompletionRequest {
        model: "test".to_string(),
        messages: vec![ChatMessage {
            role: "user".to_string(),
            content: "test".to_string(),
        }],
        temperature: Some(3.0),
        max_tokens: None,
        stream: None,
        top_p: None,
        frequency_penalty: None,
        presence_penalty: None,
    };

    let result = ParameterParser::from_request(&request);
    assert!(result.is_err());
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("temperature must be between")
    );
}

/// Test model discovery and registry workflow end-to-end
#[test]
fn test_end_to_end_model_discovery_and_registry() {
    let temp_dir = TempDir::new().unwrap();
    let models_dir = temp_dir.path().join("models");
    fs::create_dir(&models_dir).unwrap();

    // Create multiple GGUF files
    for i in 1..=3 {
        let model_file = models_dir.join(format!("model{}.gguf", i));
        fs::write(&model_file, format!("GGUF model {}", i)).unwrap();
    }

    // Discover models
    let loader = ModelLoader::new(models_dir.clone());
    let discovered = loader.discover_models().unwrap();
    assert_eq!(discovered.len(), 3);

    // Create registry and add discovered models
    let mut registry = ModelRegistry::new();
    registry.discover(&models_dir).unwrap();

    let models = registry.list_models();
    assert_eq!(models.len(), 3);

    // Verify all models can be retrieved
    for i in 1..=3 {
        let model_id = format!("model{}", i);
        let model = registry.get_model(&model_id);
        assert!(model.is_some());
        assert_eq!(model.unwrap().id, model_id);
        assert_eq!(model.unwrap().object, "model");
        assert_eq!(model.unwrap().owned_by, "local");
    }
}

/// Test request summarization for logging
#[test]
fn test_request_summary_logging() {
    use crate::inference::parameters::ParameterParser;
    use crate::models::{ChatCompletionRequest, ChatMessage};

    let request = ChatCompletionRequest {
        model: "gpt-4".to_string(),
        messages: vec![
            ChatMessage {
                role: "user".to_string(),
                content: "Hello".to_string(),
            },
            ChatMessage {
                role: "assistant".to_string(),
                content: "Hi there".to_string(),
            },
        ],
        temperature: Some(0.5),
        max_tokens: Some(1024),
        stream: Some(true),
        top_p: None,
        frequency_penalty: None,
        presence_penalty: None,
    };

    let summary = ParameterParser::summarize_request(&request);

    assert!(summary.contains("model=gpt-4"));
    assert!(summary.contains("messages=2"));
    assert!(summary.contains("temp=0.5"));
    assert!(summary.contains("max_tokens=1024"));
    assert!(summary.contains("stream=true"));
}

/// Test Phase 3.5: Real inference engine lifecycle
#[test]
fn test_phase35_llama_engine_lifecycle() {
    use crate::inference::llama_engine::LlamaEngine;

    let temp_dir = TempDir::new().unwrap();
    let model_path = temp_dir.path().join("test.gguf");
    fs::write(&model_path, "dummy").unwrap();

    let mut engine = LlamaEngine::new(model_path);
    assert!(!engine.is_loaded());

    // Load model
    assert!(engine.load(2048).is_ok());
    assert!(engine.is_loaded());

    // Generate response
    let response = engine.generate("Hello", 100);
    assert!(response.is_ok());
    assert!(!response.unwrap().is_empty());

    // Unload model
    engine.unload();
    assert!(!engine.is_loaded());
}

/// Test Phase 3.5: GPU context initialization
#[test]
fn test_phase35_gpu_context_detection() {
    use crate::inference::gpu_context::GpuContext;

    let ctx = GpuContext::new().unwrap_or_default();
    assert!(ctx.available_memory() > 0);
    assert_eq!(ctx.allocated_memory(), 0);
}

/// Test Phase 3.5: GPU memory allocation and deallocation
#[test]
fn test_phase35_gpu_memory_management() {
    use crate::inference::gpu_context::GpuContext;

    let mut ctx = GpuContext::new().unwrap_or_default();
    let initial_available = ctx.available_memory();

    let alloc_size = 100 * 1024 * 1024; // 100MB
    assert!(ctx.allocate(alloc_size).is_ok());
    assert_eq!(ctx.allocated_memory(), alloc_size);
    assert_eq!(ctx.available_memory(), initial_available - alloc_size);

    assert!(ctx.deallocate(alloc_size).is_ok());
    assert_eq!(ctx.allocated_memory(), 0);
    assert_eq!(ctx.available_memory(), initial_available);
}

/// Test Phase 3.5: Token stream from llama.cpp
#[test]
fn test_phase35_token_stream_collection() {
    use crate::inference::token_stream::TokenStream;

    let stream = TokenStream::new();
    stream.push_token("Hello".to_string());
    stream.push_token(" ".to_string());
    stream.push_token("World".to_string());

    assert_eq!(stream.total_tokens(), 3);
    assert_eq!(stream.to_string(), "Hello World");
}

/// Test Phase 3.5: Token stream iteration
#[test]
fn test_phase35_token_stream_iteration() {
    use crate::inference::token_stream::TokenStream;

    let stream = TokenStream::new();
    stream.push_token("token1".to_string());
    stream.push_token("token2".to_string());

    let mut stream = stream;
    assert!(stream.has_next());
    assert_eq!(stream.position(), 0);

    let t1 = stream.next_token();
    assert_eq!(t1, Some("token1".to_string()));
    assert_eq!(stream.position(), 1);

    let t2 = stream.next_token();
    assert_eq!(t2, Some("token2".to_string()));
    assert_eq!(stream.position(), 2);

    assert!(!stream.has_next());
    assert_eq!(stream.next_token(), None);
}

/// Test Phase 3.5: Token stream reset
#[test]
fn test_phase35_token_stream_reset() {
    use crate::inference::token_stream::TokenStream;

    let stream = TokenStream::new();
    stream.push_token("test".to_string());

    let mut stream = stream;
    stream.next_token();
    assert_eq!(stream.position(), 1);

    stream.reset();
    assert_eq!(stream.position(), 0);
    assert!(stream.has_next());
}

/// Test Phase 3.5: Full inference pipeline with streaming
#[test]
fn test_phase35_full_inference_pipeline() {
    use crate::inference::gpu_context::GpuContext;
    use crate::inference::llama_engine::LlamaEngine;
    use crate::inference::token_stream::TokenStream;

    let temp_dir = TempDir::new().unwrap();
    let model_path = temp_dir.path().join("test.gguf");
    fs::write(&model_path, "dummy").unwrap();

    // Initialize GPU context
    let ctx = GpuContext::new().unwrap_or_default();
    assert!(ctx.available_memory() > 0);

    // Load model with engine
    let mut engine = LlamaEngine::new(model_path);
    assert!(engine.load(2048).is_ok());

    // Create token stream for output
    let stream = TokenStream::new();
    stream.push_token("This".to_string());
    stream.push_token(" is".to_string());
    stream.push_token(" a".to_string());
    stream.push_token(" response".to_string());

    // Verify stream collection
    assert_eq!(stream.total_tokens(), 4);
    let mut stream = stream;
    let mut collected = String::new();
    while stream.has_next() {
        if let Some(token) = stream.next_token() {
            collected.push_str(&token);
        }
    }
    assert_eq!(collected, "This is a response");

    // Unload
    engine.unload();
    assert!(!engine.is_loaded());
}

/// Test Phase 3.5: Multiple concurrent models with GPU context
#[test]
fn test_phase35_multi_model_gpu_context() {
    use crate::inference::gpu_context::GpuContext;
    use crate::inference::llama_engine::LlamaEngine;

    let temp_dir = TempDir::new().unwrap();

    // Create multiple models
    let model1_path = temp_dir.path().join("model1.gguf");
    let model2_path = temp_dir.path().join("model2.gguf");
    fs::write(&model1_path, "dummy").unwrap();
    fs::write(&model2_path, "dummy").unwrap();

    // Initialize shared GPU context
    let mut gpu_ctx = GpuContext::new().unwrap_or_default();
    let initial_memory = gpu_ctx.available_memory();

    // Load multiple models
    let mut engine1 = LlamaEngine::new(model1_path);
    let mut engine2 = LlamaEngine::new(model2_path);

    assert!(engine1.load(2048).is_ok());
    assert!(engine2.load(2048).is_ok());

    // Simulate memory allocation
    let model_size = 500 * 1024 * 1024; // 500MB per model
    assert!(gpu_ctx.allocate(model_size).is_ok());
    assert!(gpu_ctx.allocate(model_size).is_ok());

    assert_eq!(gpu_ctx.allocated_memory(), model_size * 2);
    assert!(gpu_ctx.available_memory() < initial_memory);

    // Generate from both
    let resp1 = engine1.generate("prompt1", 100);
    let resp2 = engine2.generate("prompt2", 100);

    assert!(resp1.is_ok());
    assert!(resp2.is_ok());

    // Cleanup
    engine1.unload();
    engine2.unload();
    gpu_ctx.deallocate(model_size).ok();
    gpu_ctx.deallocate(model_size).ok();
    assert_eq!(gpu_ctx.allocated_memory(), 0);
}

/// Test Phase 3.5: Error handling for out-of-memory
#[test]
fn test_phase35_out_of_memory_error() {
    use crate::inference::gpu_context::GpuContext;

    // Create a context and try to allocate more than available
    let mut ctx = GpuContext::new().unwrap_or_default();
    let available = ctx.available_memory();

    // Should fail when trying to allocate more than available
    let result = ctx.allocate(available + 1);
    assert!(result.is_err());
}

/// Test Phase 3.5: Context info retrieval
#[test]
fn test_phase35_context_info() {
    use crate::inference::llama_engine::LlamaEngine;

    let temp_dir = TempDir::new().unwrap();
    let model_path = temp_dir.path().join("test.gguf");
    fs::write(&model_path, "dummy").unwrap();

    let mut engine = LlamaEngine::new(model_path.clone());
    assert!(engine.load(2048).is_ok());

    let info = engine.get_context_info().unwrap();
    assert_eq!(info.context_size, 2048);
    assert!(info.thread_count > 0);
    assert_eq!(info.model_path, model_path);
}
