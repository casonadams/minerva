/// Download Progress Tracking
///
/// Tracks download progress with speed calculation and ETA.
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};

// ============================================================================
// Progress Types
// ============================================================================

/// Progress update input
#[derive(Debug, Clone)]
pub struct ProgressUpdate {
    pub downloaded: u64,
    pub files_total: usize,
    pub files_done: usize,
}

/// Download progress state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DownloadProgress {
    /// Model ID
    pub model_id: String,
    /// Total bytes
    pub total_bytes: u64,
    /// Downloaded bytes
    pub downloaded_bytes: u64,
    /// Percent complete (0-100)
    pub percent: u8,
    /// Files completed
    pub files_completed: usize,
    /// Total files
    pub total_files: usize,
    /// Speed (MB/s)
    pub speed_mbps: f32,
    /// ETA (seconds)
    pub eta_seconds: u64,
}

/// Progress tracker
pub struct ProgressTracker {
    model_id: String,
    total_bytes: u64,
    start_time: Instant,
    last_update: Instant,
}

impl ProgressTracker {
    /// Create tracker
    pub fn new(model_id: String, total_bytes: u64) -> Self {
        let now = Instant::now();
        Self {
            model_id,
            total_bytes,
            start_time: now,
            last_update: now,
        }
    }

    /// Get progress
    pub fn progress(&self, update: ProgressUpdate) -> DownloadProgress {
        let elapsed = self.start_time.elapsed().as_secs_f32();
        let speed_mbps = if elapsed > 0.0 {
            (update.downloaded as f32 / (1024.0 * 1024.0)) / elapsed
        } else {
            0.0
        };

        let percent = if self.total_bytes > 0 {
            ((update.downloaded as f32 / self.total_bytes as f32) * 100.0) as u8
        } else {
            0
        };

        let remaining_bytes = self.total_bytes.saturating_sub(update.downloaded);
        let eta_seconds = if speed_mbps > 0.0 {
            (remaining_bytes as f32 / (1024.0 * 1024.0) / speed_mbps) as u64
        } else {
            0
        };

        DownloadProgress {
            model_id: self.model_id.clone(),
            total_bytes: self.total_bytes,
            downloaded_bytes: update.downloaded,
            percent,
            files_completed: update.files_done,
            total_files: update.files_total,
            speed_mbps,
            eta_seconds,
        }
    }

    /// Check if should update (throttle updates)
    pub fn should_update(&mut self) -> bool {
        let now = Instant::now();
        if now.duration_since(self.last_update) > Duration::from_millis(500) {
            self.last_update = now;
            true
        } else {
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tracker_creation() {
        let tracker = ProgressTracker::new("test".to_string(), 1000);
        assert_eq!(tracker.total_bytes, 1000);
    }

    #[test]
    fn test_progress_calculation() {
        let tracker = ProgressTracker::new("test".to_string(), 1000);
        let progress = tracker.progress(ProgressUpdate {
            downloaded: 500,
            files_total: 10,
            files_done: 5,
        });
        assert_eq!(progress.percent, 50);
        assert_eq!(progress.files_completed, 5);
    }

    #[test]
    fn test_progress_tracking() {
        let tracker = ProgressTracker::new("test".to_string(), 1000);
        let progress = tracker.progress(ProgressUpdate {
            downloaded: 250,
            files_total: 10,
            files_done: 2,
        });
        // 250 / 1000 = 25%
        assert_eq!(progress.percent, 25);
    }
}
