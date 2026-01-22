// Phase 4 Step 3: Advanced Parameter Tuning & Optimization
pub mod phase4_step3;

// Phase 4 Step 4: Real Tokenization and Vocabulary
pub mod tokenization; // Real tokenization and vocabulary support

// Phase 4 Step 6: Batch Processing
pub mod batch_processing; // Batch processing for tokenization and inference

// Phases 1-3.5: Core Functionality
pub mod error_recovery_e2e;
pub mod gpu_and_parameters; // GPU context and parameter validation
pub mod inference_engine; // Inference engine lifecycle
pub mod model_discovery; // Model discovery and registry
pub mod streaming; // Token streaming and backend abstraction

// Phase 4 Steps 1-2: Multi-Model Support
pub mod multimodel_support; // Multi-model caching and preloading

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
