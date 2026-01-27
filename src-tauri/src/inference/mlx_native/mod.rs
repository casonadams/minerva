/// Rust-native MLX implementation
///
/// This module implements MLX's core concepts in pure Rust for Apple Silicon
/// optimization without Python dependencies.
///
/// Phases:
/// 1. Model Loader (DONE) - Load SafeTensors from HuggingFace
/// 2. Unified Memory (DONE) - CPU/GPU memory abstraction
/// 3. KV Quantization (DONE) - 8x memory savings for KV cache
/// 4. Compute Graphs (DONE) - Graph structure and execution
/// 4B. Operation Fusion (DONE) - Operation fusion and optimization
/// 5. Metal GPU (IN PROGRESS) - Apple Metal acceleration
pub mod compute_graph;
pub mod compute_ops;
pub mod config;
pub mod gpu_buffer;
pub mod gpu_buffer_pool;
#[cfg(test)]
mod gpu_buffer_pool_tests;
pub mod gpu_execution_helpers;
pub mod gpu_graph_executor;
pub mod graph_executor;
#[cfg(test)]
mod graph_executor_tests;
pub mod graph_fusion;
pub mod graph_fusion_ops;
#[cfg(test)]
mod graph_fusion_ops_tests;
#[cfg(test)]
mod graph_fusion_tests;
pub mod graph_optimizer;
#[cfg(test)]
mod graph_optimizer_tests;
pub mod kv_quantization;
mod kv_quantization_helpers;
#[cfg(test)]
mod kv_quantization_test;
pub mod loader;
pub mod metal_gpu;
#[cfg(test)]
mod metal_gpu_tests;
pub mod metal_kernels_wrapper;
pub mod metal_stubs;
#[cfg(test)]
mod phase4b_e2e_tests;
#[cfg(test)]
mod phase4b_integration_tests;
pub mod unified_memory;

pub use config::GPTOSSConfig;
pub use kv_quantization::QuantizedKVCache;
pub use loader::{MLXLayerWeights, MLXModel, load_mlx_model};
pub use unified_memory::{ArrayShape, Device, MLXArray, MemoryPool};
