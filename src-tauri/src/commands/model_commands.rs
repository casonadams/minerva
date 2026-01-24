//! Model Management Commands
//!
//! Tauri commands for loading and downloading models.
//! Exposed to frontend for GUI operations.

use crate::inference::downloader::{ModelDownloader, ModelDownloadRequest};
use serde::{Deserialize, Serialize};

// ============================================================================
// Command Types
// ============================================================================

/// Download model command response
#[derive(Debug, Serialize, Deserialize)]
pub struct DownloadModelResponse {
    pub success: bool,
    pub message: String,
    pub model_id: Option<String>,
    pub path: Option<String>,
}

// ============================================================================
// Commands
// ============================================================================

/// Download model from HuggingFace
#[tauri::command]
pub async fn download_model(
    model_id: String,
    local_dir: String,
) -> Result<DownloadModelResponse, String> {
    let downloader = ModelDownloader::new();
    let req = ModelDownloadRequest {
        model_id: model_id.clone(),
        revision: None,
        local_dir: local_dir.clone(),
        files: None,
    };

    match downloader.download(&req).await {
        Ok(result) => Ok(DownloadModelResponse {
            success: result.success,
            message: if result.success {
                format!("Downloaded {}", model_id)
            } else {
                "Download incomplete".to_string()
            },
            model_id: Some(model_id),
            path: Some(result.local_path.to_string_lossy().to_string()),
        }),
        Err(e) => Err(e.to_string()),
    }
}

/// List available models on HuggingFace
#[tauri::command]
pub async fn list_hf_models(query: String) -> Result<Vec<String>, String> {
    // TODO: Query HuggingFace Hub
    // For now, return example models
    Ok(vec![
        "meta-llama/Llama-2-7b".to_string(),
        "mistralai/Mistral-7B".to_string(),
        "microsoft/phi-2".to_string(),
    ]
    .into_iter()
    .filter(|m| m.to_lowercase().contains(&query.to_lowercase()))
    .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_list_hf_models() {
        let models = list_hf_models("llama".to_string()).await.unwrap();
        assert!(models.iter().any(|m| m.contains("Llama")));
    }

    #[tokio::test]
    async fn test_list_empty_query() {
        let models = list_hf_models("nonexistent".to_string()).await.unwrap();
        assert!(models.is_empty());
    }
}
