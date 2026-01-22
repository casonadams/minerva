pub mod phase4_step3;

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
