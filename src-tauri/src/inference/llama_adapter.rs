/// Llama.cpp Adapter Module - Re-exports
///
/// This module provides an abstraction layer between the inference engine
/// and the actual llama_cpp crate. This design allows:
///
/// 1. Easy mocking for testing
/// 2. Flexible switching between mock and real inference
/// 3. Graceful fallback if llama.cpp is unavailable
/// 4. Future integration with different inference backends
///
/// # Integration Path
///
/// When ready to integrate real llama.cpp:
///
/// ```ignore
/// 1. Create LlamaModel and LlamaContext from llama_cpp
/// 2. Implement LlamaBackend trait for real inference
/// 3. Update LlamaEngine to use LlamaBackend
/// 4. Set is_mock = false in production builds
/// ```
pub use super::inference_backend_trait::{GenerationParams, InferenceBackend};
pub use super::llama_cpp_backend::LlamaCppBackend;
pub use super::mock_backend::MockBackend;
