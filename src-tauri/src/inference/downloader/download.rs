//! Model Download - HuggingFace Hub integration

use crate::error::{MinervaError, MinervaResult};
use futures::stream::StreamExt;
use serde::{Deserialize, Serialize};
use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelDownloadRequest {
    pub model_id: String,
    pub revision: Option<String>,
    pub local_dir: String,
    pub files: Option<Vec<String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DownloadResult {
    pub model_id: String,
    pub local_path: PathBuf,
    pub success: bool,
    pub total_bytes: u64,
    pub duration_secs: u64,
}

pub struct ModelDownloader {
    hf_token: Option<String>,
    client: reqwest::Client,
}

impl ModelDownloader {
    pub fn new() -> Self {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(3600))
            .build()
            .unwrap_or_else(|_| reqwest::Client::new());

        Self {
            hf_token: None,
            client,
        }
    }

    pub fn with_token(mut self, token: String) -> Self {
        self.hf_token = Some(token);
        self
    }

    pub async fn download(&self, req: &ModelDownloadRequest) -> MinervaResult<DownloadResult> {
        if req.model_id.is_empty() {
            return Err(MinervaError::InvalidRequest(
                "model_id required".to_string(),
            ));
        }

        let start = std::time::Instant::now();
        let local_dir = PathBuf::from(&req.local_dir);
        fs::create_dir_all(&local_dir)?;

        let mut total_bytes: u64 = 0;
        let files = req
            .files
            .as_ref()
            .map(|f| f.iter().map(|s| s.as_str()).collect::<Vec<_>>())
            .unwrap_or_else(|| vec!["model.safetensors", "config.json", "tokenizer.json"]);

        for file in files {
            if let Ok(size) = self
                .download_file(&req.model_id, file, &local_dir.join(file))
                .await
            {
                total_bytes += size;
            }
        }

        let duration = start.elapsed().as_secs();
        Ok(DownloadResult {
            model_id: req.model_id.clone(),
            local_path: local_dir,
            success: total_bytes > 0,
            total_bytes,
            duration_secs: duration,
        })
    }

    pub async fn download_file(
        &self,
        model_id: &str,
        file_name: &str,
        local_path: &Path,
    ) -> MinervaResult<u64> {
        let url = format!(
            "https://huggingface.co/{}/resolve/main/{}",
            model_id, file_name
        );

        if let Some(parent) = local_path.parent() {
            fs::create_dir_all(parent)?;
        }

        let response = self
            .client
            .get(&url)
            .send()
            .await
            .map_err(|e| MinervaError::ServerError(format!("HTTP error: {}", e)))?;

        let total_size = response
            .content_length()
            .ok_or_else(|| MinervaError::ServerError("Unknown content length".to_string()))?;

        let mut file = File::create(local_path)?;
        let mut stream = response.bytes_stream();
        let mut downloaded: u64 = 0;

        while let Some(chunk) = stream.next().await {
            let chunk =
                chunk.map_err(|e| MinervaError::ServerError(format!("Download error: {}", e)))?;
            file.write_all(&chunk)?;
            downloaded += chunk.len() as u64;
        }

        if downloaded == total_size {
            Ok(total_size)
        } else {
            Err(MinervaError::ServerError(format!(
                "Incomplete: {} / {} bytes",
                downloaded, total_size
            )))
        }
    }
}

impl Default for ModelDownloader {
    fn default() -> Self {
        Self::new()
    }
}
