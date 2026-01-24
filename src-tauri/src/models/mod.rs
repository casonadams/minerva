pub mod gguf_header;
pub mod gguf_loader;
pub mod gguf_parser;
pub mod gguf_reader;
pub mod gguf_tensor;
pub mod loader;

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(dead_code)]
pub struct ModelInfo {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub owned_by: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context_window: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<usize>,
}

#[derive(Debug, Serialize)]
pub struct ModelsListResponse {
    pub object: String,
    pub data: Vec<ModelInfo>,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub max_tokens: Option<usize>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub frequency_penalty: Option<f32>,
    #[serde(default)]
    pub presence_penalty: Option<f32>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[allow(dead_code)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Serialize)]
#[allow(dead_code)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub model: String,
    pub choices: Vec<Choice>,
    pub usage: Usage,
}

#[derive(Debug, Serialize)]
#[allow(dead_code)]
pub struct Choice {
    pub index: usize,
    pub message: ChatMessage,
    pub finish_reason: String,
}

#[derive(Debug, Serialize)]
#[allow(dead_code)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

#[derive(Debug, Serialize)]
#[allow(dead_code)]
pub struct ChatCompletionChunk {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub model: String,
    pub choices: Vec<ChoiceDelta>,
}

#[derive(Debug, Serialize)]
#[allow(dead_code)]
pub struct ChoiceDelta {
    pub index: usize,
    pub delta: DeltaMessage,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Serialize)]
#[allow(dead_code)]
pub struct DeltaMessage {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
}

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
        let loader = loader::ModelLoader::new(models_dir.to_path_buf());
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
