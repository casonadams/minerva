pub mod backend;
pub mod config;
pub mod format_loader;
pub mod gguf_loader;
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
pub mod kv_cache;
pub mod layers;
pub mod loader;

pub use backend::GPUSafeTensorsBackend;
pub use config::ModelConfig;
pub use format_loader::{
    FormatLoader, LoadedModel, ModelConfig as UnifiedModelConfig, ModelFormat,
};
pub use kv_cache::KVCache;
pub use loader::SafeTensorsLoader;
