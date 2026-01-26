pub mod attention_kernel;
/// GPU-accelerated inference backend
///
/// Provides high-performance LLM inference using GPU computation.
///
/// # Features
/// - KV cache for efficient generation
/// - Flash attention (planned)
/// - INT8 quantization support (planned)
/// - Request batching (planned)
/// - Speculative decoding (planned)
pub mod backend;
pub mod config;
pub mod format_loader;
pub mod gguf_loader;
pub mod inference;
pub mod kv_cache;
pub mod layers;
pub mod loader;
pub mod openai_api;
pub mod tool_api;
pub mod tool_optimized_loader;

pub use attention_kernel::{causal_mask, flash_attention_approx, gqa_attention, softmax_1d};
pub use backend::GPUSafeTensorsBackend;
pub use config::ModelConfig;
pub use format_loader::{
    FormatLoader, LoadedModel, ModelConfig as UnifiedModelConfig, ModelFormat,
};
pub use gguf_loader::GGUFLoader;
pub use inference::{FastInferenceEngine, InferenceMetrics, KVCacheOptimized};
pub use kv_cache::KVCache;
pub use loader::SafeTensorsLoader;
pub use openai_api::{
    OpenAIAPI, OpenAICompletionRequest, OpenAICompletionResponse, OpenAIListModelsResponse,
    OpenAIModelInfo, OpenAIModelRegistry,
};
pub use tool_optimized_loader::{ModelFormatQuick, QuickConfig, ToolOptimizedLoader};
