/// Model Downloader - Phase 10 Feature
///
/// Download and manage models from HuggingFace Hub.
/// Supports CLI and GUI operations.
///
/// # Features
/// - Download models from HuggingFace
/// - Progress tracking
/// - Parallel downloads
/// - Resume incomplete downloads
/// - Verify model integrity
/// - Cache management

use crate::error::{MinervaError, MinervaResult};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

// ============================================================================
// Download Types
// ============================================================================

/// Model download request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelDownloadRequest {
    /// HuggingFace model ID (e.g., "meta-llama/Llama-2-7b")
    pub model_id: String,
    /// Optional revision/branch (default: main)
    pub revision: Option<String>,
    /// Local directory to download to
    pub local_dir: String,
    /// Files to download (if None, download all)
    pub files: Option<Vec<String>>,
}

/// Download progress
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DownloadProgress {
    /// Model identifier
    pub model_id: String,
    /// Total size in bytes
    pub total_bytes: u64,
    /// Downloaded bytes so far
    pub downloaded_bytes: u64,
    /// Files downloaded
    pub files_completed: usize,
    /// Total files
    pub total_files: usize,
    /// Download speed in MB/s
    pub speed_mbps: f32,
    /// Estimated time remaining in seconds
    pub eta_seconds: u64,
}

/// Download result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DownloadResult {
    /// Model identifier
    pub model_id: String,
    /// Download path
    pub local_path: PathBuf,
    /// Success
    pub success: bool,
    /// Total size downloaded
    pub total_bytes: u64,
    /// Duration in seconds
    pub duration_secs: u64,
}

// ============================================================================
// Model Downloader
// ============================================================================

/// Downloads models from HuggingFace Hub
pub struct ModelDownloader {
    /// HuggingFace API token (optional for private models)
    #[allow(dead_code)]
    hf_token: Option<String>,
    /// Max concurrent downloads
    #[allow(dead_code)]
    max_parallel: usize,
}

impl ModelDownloader {
    /// Create new downloader
    pub fn new() -> Self {
        Self {
            hf_token: None,
            max_parallel: 3,
        }
    }

    /// Set HuggingFace token for private models
    pub fn with_token(mut self, token: String) -> Self {
        self.hf_token = Some(token);
        self
    }

    /// Download a model
    pub async fn download(
        &self,
        req: &ModelDownloadRequest,
    ) -> MinervaResult<DownloadResult> {
        self.validate_request(req)?;
        self.download_from_hf(req).await
    }

    /// Download and show progress
    pub async fn download_with_progress<F>(
        &self,
        req: &ModelDownloadRequest,
        progress_cb: F,
    ) -> MinervaResult<DownloadResult>
    where
        F: Fn(DownloadProgress) + Send + 'static,
    {
        self.validate_request(req)?;
        self.download_with_callback(req, progress_cb).await
    }

    /// Validate download request
    fn validate_request(&self, req: &ModelDownloadRequest) -> MinervaResult<()> {
        if req.model_id.is_empty() {
            return Err(MinervaError::InvalidRequest(
                "model_id cannot be empty".to_string(),
            ));
        }

        let path = Path::new(&req.local_dir);
        if !path.parent().map_or(true, |p| p.exists()) {
            return Err(MinervaError::InvalidRequest(
                format!("Parent directory does not exist: {}", req.local_dir),
            ));
        }

        Ok(())
    }

    // ========================================================================
    // Implementation Details (stubs for now)
    // ========================================================================

    async fn download_from_hf(
        &self,
        req: &ModelDownloadRequest,
    ) -> MinervaResult<DownloadResult> {
        // TODO: Implement actual HF Hub download
        Ok(DownloadResult {
            model_id: req.model_id.clone(),
            local_path: PathBuf::from(&req.local_dir),
            success: false,
            total_bytes: 0,
            duration_secs: 0,
        })
    }

    async fn download_with_callback<F>(
        &self,
        req: &ModelDownloadRequest,
        _progress_cb: F,
    ) -> MinervaResult<DownloadResult>
    where
        F: Fn(DownloadProgress) + Send + 'static,
    {
        // TODO: Implement with progress callback
        self.download_from_hf(req).await
    }
}

impl Default for ModelDownloader {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_downloader_creation() {
        let downloader = ModelDownloader::new();
        assert_eq!(downloader.max_parallel, 3);
    }

    #[test]
    fn test_validate_request_empty_id() {
        let downloader = ModelDownloader::new();
        let req = ModelDownloadRequest {
            model_id: String::new(),
            revision: None,
            local_dir: "/tmp".to_string(),
            files: None,
        };
        assert!(downloader.validate_request(&req).is_err());
    }

    #[test]
    fn test_validate_request_valid() {
        let downloader = ModelDownloader::new();
        let req = ModelDownloadRequest {
            model_id: "meta-llama/Llama-2-7b".to_string(),
            revision: None,
            local_dir: "/tmp".to_string(),
            files: None,
        };
        assert!(downloader.validate_request(&req).is_ok());
    }

    #[tokio::test]
    async fn test_download_result() {
        let downloader = ModelDownloader::new();
        let req = ModelDownloadRequest {
            model_id: "test-model".to_string(),
            revision: None,
            local_dir: "/tmp".to_string(),
            files: None,
        };
        let result = downloader.download(&req).await.unwrap();
        assert_eq!(result.model_id, "test-model");
    }
}
