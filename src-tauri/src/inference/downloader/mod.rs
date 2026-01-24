//! Model Downloader Module - Phase 10 Day 4
//!
//! Download and manage models from HuggingFace Hub.
//! Supports CLI and GUI operations with progress tracking.

pub mod download;
pub mod cache;
pub mod progress;

pub use download::{ModelDownloader, ModelDownloadRequest, DownloadResult};
pub use cache::{DownloadCache, CacheEntry};
pub use progress::DownloadProgress;
