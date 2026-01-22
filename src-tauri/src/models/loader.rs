use crate::error::{MinervaError, MinervaResult};
use crate::models::ModelInfo;
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

/// Discovers and loads GGUF models from a directory
pub struct ModelLoader {
    models_dir: PathBuf,
}

impl ModelLoader {
    /// Create a new model loader for the specified directory
    pub fn new(models_dir: PathBuf) -> Self {
        Self { models_dir }
    }

    /// Discover all GGUF model files in the models directory
    pub fn discover_models(&self) -> MinervaResult<Vec<ModelInfo>> {
        let mut models = Vec::new();

        if !self.models_dir.exists() {
            return Ok(models);
        }

        for entry in WalkDir::new(&self.models_dir)
            .into_iter()
            .filter_map(|e| e.ok())
        {
            let path = entry.path();

            // Only process .gguf files
            if path.extension().and_then(|s| s.to_str()) == Some("gguf") {
                match self.load_model(path) {
                    Ok(model_info) => models.push(model_info),
                    Err(e) => {
                        tracing::warn!("Failed to load model {}: {}", path.display(), e);
                    }
                }
            }
        }

        Ok(models)
    }

    /// Load a single GGUF model file
    pub fn load_model(&self, path: &Path) -> MinervaResult<ModelInfo> {
        if !path.exists() {
            return Err(MinervaError::ModelNotFound(format!(
                "Model file not found: {}",
                path.display()
            )));
        }

        let file_name = path
            .file_stem()
            .and_then(|n| n.to_str())
            .ok_or_else(|| MinervaError::ModelLoadingError("Invalid model filename".to_string()))?
            .to_string();

        // Get file metadata
        let _metadata = std::fs::metadata(path).map_err(MinervaError::IoError)?;

        // Create model info
        let model_info = ModelInfo {
            id: file_name.clone(),
            object: "model".to_string(),
            created: chrono::Utc::now().timestamp(),
            owned_by: "local".to_string(),
            context_window: Some(4096), // Default, can be enhanced with GGUF parsing
            max_output_tokens: Some(2048),
        };

        Ok(model_info)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    #[test]
    fn test_loader_new() {
        let loader = ModelLoader::new(PathBuf::from("/tmp/models"));
        assert_eq!(loader.models_dir, PathBuf::from("/tmp/models"));
    }

    #[test]
    fn test_discover_models_empty_dir() {
        let temp_dir = TempDir::new().unwrap();
        let loader = ModelLoader::new(temp_dir.path().to_path_buf());

        let models = loader.discover_models().unwrap();
        assert_eq!(models.len(), 0);
    }

    #[test]
    fn test_discover_models_ignores_non_gguf() {
        let temp_dir = TempDir::new().unwrap();

        // Create non-GGUF files
        fs::write(temp_dir.path().join("model.txt"), "test").unwrap();
        fs::write(temp_dir.path().join("model.bin"), "test").unwrap();

        let loader = ModelLoader::new(temp_dir.path().to_path_buf());
        let models = loader.discover_models().unwrap();
        assert_eq!(models.len(), 0);
    }

    #[test]
    fn test_load_model_file_not_found() {
        let loader = ModelLoader::new(PathBuf::from("/tmp/models"));
        let result = loader.load_model(Path::new("/nonexistent/model.gguf"));

        assert!(result.is_err());
    }

    #[test]
    fn test_load_model_valid_file() {
        let temp_dir = TempDir::new().unwrap();
        let model_path = temp_dir.path().join("test-model.gguf");

        // Create a dummy GGUF file
        fs::write(&model_path, "dummy content").unwrap();

        let loader = ModelLoader::new(temp_dir.path().to_path_buf());
        let model = loader.load_model(&model_path).unwrap();

        assert_eq!(model.id, "test-model");
        assert_eq!(model.object, "model");
        assert_eq!(model.owned_by, "local");
        assert_eq!(model.context_window, Some(4096));
    }
}
