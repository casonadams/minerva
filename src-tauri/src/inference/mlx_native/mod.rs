/// Rust-native MLX implementation
///
/// This module implements MLX's core concepts in pure Rust for Apple Silicon
/// optimization without Python dependencies.
///
/// Phases:
/// 1. Model Loader (DONE) - Load SafeTensors from HuggingFace
/// 2. Unified Memory (DONE) - CPU/GPU memory abstraction
/// 3. KV Quantization (DONE) - 8x memory savings for KV cache
/// 4. Compute Graphs (IN PROGRESS) - Operation fusion and optimization
/// 5. Metal GPU - Apple Metal acceleration
pub mod compute_graph;
pub mod compute_ops;
pub mod config;
pub mod graph_executor;
pub mod kv_quantization;
mod kv_quantization_helpers;
#[cfg(test)]
mod kv_quantization_test;
pub mod loader;
pub mod unified_memory;

pub use config::GPTOSSConfig;
pub use kv_quantization::QuantizedKVCache;
pub use loader::{MLXLayerWeights, MLXModel, load_mlx_model};
pub use unified_memory::{ArrayShape, Device, MLXArray, MemoryPool};
