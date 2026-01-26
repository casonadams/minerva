/// Rust-native MLX implementation
///
/// This module implements MLX's core concepts in pure Rust for Apple Silicon
/// optimization without Python dependencies.
///
/// Phases:
/// 1. Model Loader (CURRENT) - Load SafeTensors from HuggingFace
/// 2. Unified Memory - CPU/GPU memory abstraction
/// 3. KV Quantization - 8x memory savings for KV cache
/// 4. Compute Graphs - Operation fusion and optimization
/// 5. Metal GPU - Apple Metal acceleration
pub mod config;
pub mod loader;

pub use config::GPTOSSConfig;
pub use loader::{MLXLayerWeights, MLXModel, load_mlx_model};
