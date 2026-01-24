//! API Request Handlers
//!
//! Lean handlers that route to backend inference.
//! Each handler is focused and ≤25 lines.

use super::types::*;
use crate::error::MinervaResult;

/// Infer single prompt
pub async fn infer_prompt(
    _req: InferenceRequest,
) -> MinervaResult<InferenceResponse> {
    // TODO: Integrate with UnifiedModelRegistry
    Ok(InferenceResponse {
        id: uuid::Uuid::new_v4().to_string(),
        model: "llama".to_string(),
        tokens: vec![],
        prompt_tokens: 0,
        completion_tokens: 0,
        total_tokens: 0,
        created: current_timestamp(),
        finish_reason: "not_implemented".to_string(),
    })
}

/// Load a model
pub async fn load_model(
    _req: LoadModelRequest,
) -> MinervaResult<ModelInfoResponse> {
    // TODO: Integrate with UnifiedModelRegistry
    Ok(ModelInfoResponse {
        id: "test".to_string(),
        name: "Test".to_string(),
        model_type: "unknown".to_string(),
        vocab_size: 32000,
        hidden_size: 4096,
        num_layers: 32,
        num_attention_heads: 32,
        intermediate_size: 11008,
        max_seq_len: 2048,
        loaded: false,
        memory_mb: 0,
    })
}

/// List all models
pub async fn list_models() -> MinervaResult<ModelsResponse> {
    Ok(ModelsResponse {
        object: "list".to_string(),
        data: vec![],
    })
}

/// Unload a model
pub async fn unload_model(_model_id: String) -> MinervaResult<()> {
    // TODO: Integrate with UnifiedModelRegistry
    Ok(())
}

fn current_timestamp() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_infer_prompt() {
        let req = InferenceRequest {
            model: "llama".to_string(),
            prompt: "hello".to_string(),
            max_tokens: Some(100),
            temperature: None,
            top_k: None,
            top_p: None,
            seed: None,
        };
        let resp = infer_prompt(req).await.unwrap();
        assert!(!resp.id.is_empty());
    }

    #[tokio::test]
    async fn test_list_models() {
        let resp = list_models().await.unwrap();
        assert_eq!(resp.object, "list");
    }
}
