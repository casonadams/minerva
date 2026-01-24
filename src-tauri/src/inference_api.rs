/// Phase 10 Day 1: Inference API Layer
///
/// Provides REST/Tauri command interfaces for inference operations:
/// - Single inference requests (/infer, /complete)
/// - Streaming inference (/stream)
/// - Model management (/models/load, /models/list)
/// - Configuration and validation
///
/// This module bridges the Phase 9 inference pipeline with HTTP/IPC endpoints.

use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::Mutex;
use crate::error::{MinervaError, MinervaResult};
use crate::inference::inference_engine::InferenceEngine;
use crate::inference::model_loader::{ModelLoader, ModelLoaderConfig};
use std::collections::HashMap;

// ============================================================================
// Response Types
// ============================================================================

/// Standard error response format
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorResponse {
    pub error: ErrorDetail,
}

/// Error details matching OpenAI format
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorDetail {
    pub message: String,
    pub code: String,
    pub param: Option<String>,
}

/// Single token prediction response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenResponse {
    pub token_id: u32,
    pub token: String,
    pub logit: f32,
    pub probability: f32,
}

/// Inference completion response (single request)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceResponse {
    pub id: String,
    pub model: String,
    pub tokens: Vec<TokenResponse>,
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
    pub created: u64,
    pub finish_reason: String,
}

/// Streaming response chunk (SSE format)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamChunk {
    pub id: String,
    pub model: String,
    pub token: TokenResponse,
    pub index: usize,
    pub created: u64,
    pub finish_reason: Option<String>,
}

/// Model information response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfoResponse {
    pub id: String,
    pub name: String,
    pub model_type: String,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    pub max_seq_len: usize,
    pub loaded: bool,
    pub memory_mb: u64,
}

/// List of loaded models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelsResponse {
    pub object: String,
    pub data: Vec<ModelInfoResponse>,
}

// ============================================================================
// Request Types
// ============================================================================

/// Inference request (prompt completion)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceRequest {
    pub model: String,
    pub prompt: String,
    pub max_tokens: Option<usize>,
    pub temperature: Option<f32>,
    pub top_k: Option<usize>,
    pub top_p: Option<f32>,
    pub seed: Option<u64>,
}

/// Batch inference request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchInferenceRequest {
    pub model: String,
    pub prompts: Vec<String>,
    pub max_tokens: Option<usize>,
    pub temperature: Option<f32>,
}

/// Model load request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadModelRequest {
    pub model_id: String,
    pub model_dir: String,
}

// ============================================================================
// Inference API State
// ============================================================================

/// Thread-safe model registry for inference
pub struct InferenceModelRegistry {
    models: HashMap<String, Arc<Mutex<InferenceEngine>>>,
    configs: HashMap<String, ModelLoaderConfig>,
}

impl InferenceModelRegistry {
    /// Create new registry
    pub fn new() -> Self {
        Self {
            models: HashMap::new(),
            configs: HashMap::new(),
        }
    }

    /// Load a model from directory
    pub async fn load_model(
        &mut self,
        model_id: &str,
        model_dir: &str,
    ) -> MinervaResult<()> {
        // Create loader config
        let config = ModelLoaderConfig::from_directory(model_dir)?;

        // Load model using static method
        let engine = ModelLoader::load_model(&config)?;

        // Store in registry
        self.models.insert(
            model_id.to_string(),
            Arc::new(Mutex::new(engine)),
        );
        self.configs.insert(model_id.to_string(), config);

        Ok(())
    }

    /// Get reference to loaded model
    pub fn get_model(&self, model_id: &str) -> MinervaResult<Arc<Mutex<InferenceEngine>>> {
        self.models
            .get(model_id)
            .cloned()
            .ok_or_else(|| MinervaError::ModelNotFound(format!(
                "Model '{}' not loaded. Use /load to load it first.",
                model_id
            )))
    }

    /// List all loaded models
    pub fn list_models(&self) -> Vec<&str> {
        self.models.keys().map(|k| k.as_str()).collect()
    }

    /// Unload a model
    pub fn unload_model(&mut self, model_id: &str) -> MinervaResult<()> {
        self.models
            .remove(model_id)
            .ok_or_else(|| MinervaError::ModelNotFound(format!(
                "Model '{}' not found",
                model_id
            )))?;
        self.configs.remove(model_id);
        Ok(())
    }
}

impl Default for InferenceModelRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// API Handlers
// ============================================================================

/// Infer single prompt (synchronous completion)
pub async fn infer_prompt(
    registry: Arc<Mutex<InferenceModelRegistry>>,
    req: InferenceRequest,
) -> MinervaResult<InferenceResponse> {
    let registry = registry.lock().await;
    let engine_ref = registry.get_model(&req.model)?;

    let engine = engine_ref.lock().await;

    // Tokenize prompt
    let tokens = tokenize_prompt(&req.prompt)?;

    if tokens.is_empty() {
        return Err(MinervaError::InvalidRequest(
            "Prompt cannot be empty".to_string(),
        ));
    }

    // Forward pass
    let max_tokens = req.max_tokens.unwrap_or(100).min(512);
    let mut output_tokens = Vec::new();

    for _ in 0..max_tokens {
        let logits = engine.forward(tokens.as_slice())?;
        let sampled = sample_from_logits(
            &logits,
            req.temperature,
            req.top_k,
            req.top_p,
            req.seed,
        )?;

        let should_stop = sampled.token_id == 2; // EOS token
        output_tokens.push(sampled.clone());

        if should_stop {
            break;
        }
    }

    let completion_tokens = output_tokens.len();
    let response = InferenceResponse {
        id: uuid::Uuid::new_v4().to_string(),
        model: req.model.clone(),
        tokens: output_tokens,
        prompt_tokens: tokens.len(),
        completion_tokens,
        total_tokens: tokens.len() + completion_tokens,
        created: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs(),
        finish_reason: "stop".to_string(),
    };

    Ok(response)
}

/// Load a model into the registry
pub async fn load_model(
    registry: Arc<Mutex<InferenceModelRegistry>>,
    req: LoadModelRequest,
) -> MinervaResult<ModelInfoResponse> {
    let mut registry = registry.lock().await;

    // Load model
    registry.load_model(&req.model_id, &req.model_dir).await?;

    // Get model info
    let _engine_ref = registry.get_model(&req.model_id)?;

    Ok(ModelInfoResponse {
        id: req.model_id,
        name: req.model_dir,
        model_type: "unknown".to_string(), // TODO: Extract from config
        vocab_size: 32000, // TODO: Get from engine
        hidden_size: 4096, // TODO: Get from engine
        num_layers: 32, // TODO: Get from engine
        num_attention_heads: 32, // TODO: Get from engine
        intermediate_size: 11008, // TODO: Get from engine
        max_seq_len: 2048, // TODO: Get from engine
        loaded: true,
        memory_mb: 0, // TODO: Calculate from weights
    })
}

/// List all loaded models
pub async fn list_models(
    registry: Arc<Mutex<InferenceModelRegistry>>,
) -> MinervaResult<ModelsResponse> {
    let registry = registry.lock().await;
    let models = registry.list_models();

    Ok(ModelsResponse {
        object: "list".to_string(),
        data: models
            .into_iter()
            .map(|id| ModelInfoResponse {
                id: id.to_string(),
                name: id.to_string(),
                model_type: "unknown".to_string(),
                vocab_size: 32000,
                hidden_size: 4096,
                num_layers: 32,
                num_attention_heads: 32,
                intermediate_size: 11008,
                max_seq_len: 2048,
                loaded: true,
                memory_mb: 0,
            })
            .collect(),
    })
}

/// Unload a model
pub async fn unload_model(
    registry: Arc<Mutex<InferenceModelRegistry>>,
    model_id: String,
) -> MinervaResult<()> {
    let mut registry = registry.lock().await;
    registry.unload_model(&model_id)
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Simple tokenization (mock - would use actual tokenizer)
fn tokenize_prompt(prompt: &str) -> MinervaResult<Vec<usize>> {
    // TODO: Integrate with tokenizer from Phase 3
    // For now: simple word-based tokenization
    let tokens: Vec<usize> = prompt
        .split_whitespace()
        .enumerate()
        .map(|(i, _)| i + 1)
        .collect();

    Ok(tokens)
}

/// Sample token from logits distribution
fn sample_from_logits(
    _logits: &[f32],
    _temperature: Option<f32>,
    _top_k: Option<usize>,
    _top_p: Option<f32>,
    _seed: Option<u64>,
) -> MinervaResult<TokenResponse> {
    // TODO: Integrate with sampling module from Phase 9
    Ok(TokenResponse {
        token_id: 1,
        token: "hello".to_string(),
        logit: 5.0,
        probability: 0.9,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_registry_creation() {
        let registry = InferenceModelRegistry::new();
        assert!(registry.list_models().is_empty());
    }

    #[test]
    fn test_tokenize_prompt() {
        let tokens = tokenize_prompt("hello world").unwrap();
        assert_eq!(tokens.len(), 2);
    }

    #[test]
    fn test_inference_request_deserialize() {
        let json = r#"{
            "model": "llama",
            "prompt": "what is rust?",
            "max_tokens": 100,
            "temperature": 0.7
        }"#;
        let req: InferenceRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.model, "llama");
        assert_eq!(req.prompt, "what is rust?");
        assert_eq!(req.max_tokens, Some(100));
        assert_eq!(req.temperature, Some(0.7));
    }

    #[test]
    fn test_error_response_format() {
        let err = ErrorResponse {
            error: ErrorDetail {
                message: "test error".to_string(),
                code: "test_code".to_string(),
                param: None,
            },
        };
        let json = serde_json::to_string(&err).unwrap();
        assert!(json.contains("test error"));
    }
}
