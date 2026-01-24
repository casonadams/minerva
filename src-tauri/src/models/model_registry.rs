use super::model_info::ModelInfo;
use std::collections::HashMap;

pub struct ModelRegistry {
    models: HashMap<String, ModelInfo>,
    #[allow(dead_code)]
    model_paths: HashMap<String, std::path::PathBuf>,
}

impl ModelRegistry {
    #[allow(dead_code)]
    pub fn new() -> Self {
        Self {
            models: HashMap::new(),
            model_paths: HashMap::new(),
        }
    }

    pub fn get_model(&self, id: &str) -> Option<&ModelInfo> {
        self.models.get(id)
    }

    pub fn list_models(&self) -> Vec<ModelInfo> {
        self.models.values().cloned().collect()
    }

    #[allow(dead_code)]
    pub fn add_model(&mut self, model: ModelInfo, path: std::path::PathBuf) {
        let id = model.id.clone();
        self.models.insert(id.clone(), model);
        self.model_paths.insert(id, path);
    }

    #[allow(dead_code)]
    pub fn remove_model(&mut self, id: &str) -> Option<ModelInfo> {
        self.model_paths.remove(id);
        self.models.remove(id)
    }

    #[allow(dead_code)]
    pub fn clear(&mut self) {
        self.models.clear();
        self.model_paths.clear();
    }

    #[allow(dead_code)]
    pub fn discover(&mut self, models_dir: &std::path::Path) -> crate::error::MinervaResult<()> {
        let loader = super::loader::ModelLoader::new(models_dir.to_path_buf());
        let models = loader.discover_models()?;

        for model in models {
            let model_path = models_dir.join(&model.id).with_extension("gguf");
            self.add_model(model, model_path);
        }

        Ok(())
    }
}

impl Default for ModelRegistry {
    fn default() -> Self {
        Self::new()
    }
}
