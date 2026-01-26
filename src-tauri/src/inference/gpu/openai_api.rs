/// OpenAI API Compatible Model Loader
///
/// Implements OpenAI-style endpoints for model loading and inference
/// Works seamlessly with any OpenAI-compatible tool/client
use super::tool_optimized_loader::ToolOptimizedLoader;
use crate::error::{MinervaError, MinervaResult};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// OpenAI API compatible model info
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAIModelInfo {
    pub id: String,
    pub object: String, // "model"
    pub created: i64,
    pub owned_by: String,
    pub permission: Vec<serde_json::Value>,
    pub root: Option<String>,
    pub parent: Option<String>,
    /// Custom fields for GGUF/SafeTensors metadata
    #[serde(skip_serializing_if = "Option::is_none")]
    pub quantization: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub file_size_mb: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tensor_count: Option<usize>,
}

/// OpenAI API compatible list response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAIListModelsResponse {
    pub object: String, // "list"
    pub data: Vec<OpenAIModelInfo>,
}

/// OpenAI API compatible completion request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAICompletionRequest {
    pub model: String,
    pub prompt: String,
    #[serde(default)]
    pub max_tokens: Option<u32>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
}

/// OpenAI API compatible completion response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAICompletionResponse {
    pub id: String,
    pub object: String, // "text_completion"
    pub created: i64,
    pub model: String,
    pub choices: Vec<CompletionChoice>,
    pub usage: CompletionUsage,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionChoice {
    pub text: String,
    pub index: usize,
    pub logprobs: Option<serde_json::Value>,
    pub finish_reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionUsage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

/// OpenAI API compatible implementation
pub struct OpenAIAPI {
    model_path: std::path::PathBuf,
}

impl OpenAIAPI {
    pub fn new(model_path: &Path) -> Self {
        Self {
            model_path: model_path.to_path_buf(),
        }
    }

    /// List available models (OpenAI compatible)
    pub fn list_models(&self) -> MinervaResult<OpenAIListModelsResponse> {
        let loader = ToolOptimizedLoader::quick_load(&self.model_path)?;

        let model_info = OpenAIModelInfo {
            id: loader.model_name.clone(),
            object: "model".to_string(),
            created: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs() as i64,
            owned_by: "local".to_string(),
            permission: vec![],
            root: None,
            parent: None,
            quantization: Some(loader.quantization),
            file_size_mb: Some(loader.file_size_mb),
            tensor_count: Some(loader.tensor_count),
        };

        Ok(OpenAIListModelsResponse {
            object: "list".to_string(),
            data: vec![model_info],
        })
    }

    /// Get model info (OpenAI compatible)
    pub fn get_model(&self, _model_id: &str) -> MinervaResult<OpenAIModelInfo> {
        let loader = ToolOptimizedLoader::quick_load(&self.model_path)?;

        Ok(OpenAIModelInfo {
            id: loader.model_name,
            object: "model".to_string(),
            created: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs() as i64,
            owned_by: "local".to_string(),
            permission: vec![],
            root: None,
            parent: None,
            quantization: Some(loader.quantization),
            file_size_mb: Some(loader.file_size_mb),
            tensor_count: Some(loader.tensor_count),
        })
    }

    /// Create completion (stub for now)
    pub fn create_completion(
        &self,
        _request: OpenAICompletionRequest,
    ) -> MinervaResult<OpenAICompletionResponse> {
        // TODO: Implement actual inference
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64;

        Ok(OpenAICompletionResponse {
            id: format!("cmpl-{}", now),
            object: "text_completion".to_string(),
            created: now,
            model: "gpt-oss-20b".to_string(),
            choices: vec![CompletionChoice {
                text: "[model inference not yet implemented]".to_string(),
                index: 0,
                logprobs: None,
                finish_reason: "stop".to_string(),
            }],
            usage: CompletionUsage {
                prompt_tokens: 0,
                completion_tokens: 0,
                total_tokens: 0,
            },
        })
    }

    /// Check if model is available
    pub fn is_available(&self) -> bool {
        self.model_path.exists()
    }
}

/// Quick helper for OpenAI API compatibility
pub struct OpenAIModelRegistry {
    models: std::collections::HashMap<String, std::path::PathBuf>,
}

impl OpenAIModelRegistry {
    pub fn new() -> Self {
        Self {
            models: std::collections::HashMap::new(),
        }
    }

    pub fn register(&mut self, name: &str, path: &Path) {
        self.models.insert(name.to_string(), path.to_path_buf());
    }

    pub fn get(&self, name: &str) -> Option<OpenAIAPI> {
        self.models.get(name).map(|path| OpenAIAPI::new(path))
    }

    pub fn list(&self) -> Vec<&str> {
        self.models.keys().map(|s| s.as_str()).collect()
    }
}

impl Default for OpenAIModelRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_openai_model_info_serialization() {
        let info = OpenAIModelInfo {
            id: "gpt-oss-20b".to_string(),
            object: "model".to_string(),
            created: 1674000000,
            owned_by: "local".to_string(),
            permission: vec![],
            root: None,
            parent: None,
            quantization: Some("MXFP4".to_string()),
            file_size_mb: Some(12109.6),
            tensor_count: Some(459),
        };

        let json = serde_json::to_string(&info).unwrap();
        assert!(json.contains("gpt-oss-20b"));
        assert!(json.contains("MXFP4"));
        assert!(json.contains("12109"));
    }

    #[test]
    fn test_openai_list_models_response() {
        let response = OpenAIListModelsResponse {
            object: "list".to_string(),
            data: vec![OpenAIModelInfo {
                id: "gpt-oss-20b".to_string(),
                object: "model".to_string(),
                created: 1674000000,
                owned_by: "local".to_string(),
                permission: vec![],
                root: None,
                parent: None,
                quantization: Some("MXFP4".to_string()),
                file_size_mb: Some(12109.6),
                tensor_count: Some(459),
            }],
        };

        let json = serde_json::to_string(&response).unwrap();
        assert!(json.contains("list"));
        assert!(json.contains("data"));
    }

    #[test]
    fn test_model_registry() {
        let mut registry = OpenAIModelRegistry::new();
        let path = std::path::Path::new("/tmp/dummy.gguf");

        registry.register("gpt-oss-20b", path);
        assert!(registry.get("gpt-oss-20b").is_some());
        assert!(registry.get("nonexistent").is_none());

        let models = registry.list();
        assert!(models.contains(&"gpt-oss-20b"));
    }

    #[test]
    fn test_completion_request_deserialization() {
        let json = r#"{
            "model": "gpt-oss-20b",
            "prompt": "Hello, world!",
            "max_tokens": 100,
            "temperature": 0.7
        }"#;

        let req: OpenAICompletionRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.model, "gpt-oss-20b");
        assert_eq!(req.prompt, "Hello, world!");
        assert_eq!(req.max_tokens, Some(100));
    }
}
