//! Model Downloader Module - Phase 10 Day 4
//!
//! Download and manage models from HuggingFace Hub.
//! Supports CLI and GUI operations with progress tracking.

pub mod cache;
pub mod download;
pub mod progress;

pub use cache::{CacheEntry, DownloadCache};
pub use download::{DownloadResult, ModelDownloadRequest, ModelDownloader};
pub use progress::DownloadProgress;
