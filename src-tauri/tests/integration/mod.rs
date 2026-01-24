// Phases 1-3.5: Core Functionality
pub mod error_recovery_e2e; // Error recovery and resilience patterns
pub mod gpu_and_parameters; // GPU context and parameter validation
pub mod inference_engine; // Inference engine lifecycle
pub mod model_discovery; // Model discovery and registry
pub mod streaming; // Token streaming and backend abstraction

// Phase 4 Steps 1-2: Multi-Model Support & Caching
pub mod context_management; // Context lifecycle and memory tracking
pub mod model_caching; // LRU/LFU/FIFO cache strategies
pub mod model_registry; // Model registry and cache management
pub mod multimodel_support; // Multi-model context management
pub mod preloading; // Preload manager and strategies

// Phase 4 Step 3: Advanced Parameter Tuning & Optimization
pub mod phase4_step3;

// Phase 4 Step 4: Real Tokenization and Vocabulary
pub mod tokenization; // Real tokenization and vocabulary support

// Phase 4 Step 6: Batch Processing
pub mod batch_processing; // Batch processing for tokenization and inference

// Phase 6 & 7: Performance and Observability
pub mod performance_metrics; // Performance tracking and benchmarking

// Phase 11: REST API Decoupling & Headless Server
pub mod headless_server;
pub mod http_api; // HTTP API endpoints and contracts // Headless server and Tauri decoupling

use std::fs;
use std::path::{Path, PathBuf};
use tempfile::TempDir;

/// Create a temporary directory with test models
#[allow(dead_code)]
pub fn setup_test_models_dir() -> (TempDir, PathBuf) {
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let models_dir = temp_dir.path().to_path_buf();
    (temp_dir, models_dir)
}

/// Create a dummy GGUF file for testing
#[allow(dead_code)]
pub fn create_dummy_gguf(dir: &Path, name: &str) -> PathBuf {
    let path = dir.join(format!("{}.gguf", name));
    fs::write(&path, b"GGUF").expect("Failed to write dummy GGUF");
    path
}
