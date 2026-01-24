use super::token_bucket::TokenBucket;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;

/// Rate limiter with per-client token buckets
pub struct RateLimiter {
    buckets: Arc<RwLock<HashMap<String, TokenBucket>>>,
    max_tokens: f64,
    refill_rate: f64,
    cleanup_interval: Duration,
    last_cleanup: Arc<RwLock<Instant>>,
}

impl RateLimiter {
    /// Create new rate limiter
    pub fn new(max_tokens: f64, requests_per_sec: f64) -> Self {
        Self {
            buckets: Arc::new(RwLock::new(HashMap::new())),
            max_tokens,
            refill_rate: requests_per_sec,
            cleanup_interval: Duration::from_secs(300),
            last_cleanup: Arc::new(RwLock::new(Instant::now())),
        }
    }

    /// Check if client can make request
    pub async fn allow_request(&self, client_id: &str, tokens: f64) -> bool {
        let mut buckets = self.buckets.write().await;
        let bucket = buckets
            .entry(client_id.to_string())
            .or_insert_with(|| TokenBucket::new(self.max_tokens, self.refill_rate));

        bucket.try_take(tokens)
    }

    /// Get remaining tokens for client
    pub async fn remaining(&self, client_id: &str) -> f64 {
        let mut buckets = self.buckets.write().await;
        let bucket = buckets
            .entry(client_id.to_string())
            .or_insert_with(|| TokenBucket::new(self.max_tokens, self.refill_rate));

        bucket.available()
    }

    /// Get retry-after duration in seconds
    pub async fn retry_after(&self, client_id: &str, tokens: f64) -> u64 {
        let mut buckets = self.buckets.write().await;
        let bucket = buckets
            .entry(client_id.to_string())
            .or_insert_with(|| TokenBucket::new(self.max_tokens, self.refill_rate));

        bucket.retry_seconds(tokens)
    }

    /// Clean up old buckets (call periodically)
    pub async fn cleanup_old_buckets(&self, max_age: Duration) {
        let mut last = self.last_cleanup.write().await;
        if last.elapsed() < self.cleanup_interval {
            return;
        }

        let mut buckets = self.buckets.write().await;
        let cutoff = Instant::now() - max_age;
        buckets.retain(|_, bucket| bucket.last_refill > cutoff);
        *last = Instant::now();
    }

    /// Reset client bucket
    pub async fn reset_client(&self, client_id: &str) {
        let mut buckets = self.buckets.write().await;
        buckets.remove(client_id);
    }

    /// Get bucket count (for testing)
    #[allow(dead_code)]
    pub async fn bucket_count(&self) -> usize {
        self.buckets.read().await.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_allows_under_limit() {
        let limiter = RateLimiter::new(10.0, 2.0);
        assert!(limiter.allow_request("client1", 5.0).await);
        assert!(limiter.allow_request("client1", 4.0).await);
    }

    #[tokio::test]
    async fn test_blocks_over_limit() {
        let limiter = RateLimiter::new(10.0, 2.0);
        assert!(limiter.allow_request("client1", 10.0).await);
        assert!(!limiter.allow_request("client1", 1.0).await);
    }

    #[tokio::test]
    async fn test_separate_clients() {
        let limiter = RateLimiter::new(10.0, 2.0);
        assert!(limiter.allow_request("client1", 10.0).await);
        assert!(limiter.allow_request("client2", 10.0).await);
    }

    #[tokio::test]
    async fn test_remaining() {
        let limiter = RateLimiter::new(10.0, 2.0);
        limiter.allow_request("client1", 3.0).await;
        let remaining = limiter.remaining("client1").await;
        assert!(remaining >= 7.0 && remaining <= 10.0);
    }

    #[tokio::test]
    async fn test_retry_after() {
        let limiter = RateLimiter::new(10.0, 2.0);
        limiter.allow_request("client1", 10.0).await;
        let retry = limiter.retry_after("client1", 1.0).await;
        assert!(retry > 0);
    }

    #[tokio::test]
    async fn test_reset_client() {
        let limiter = RateLimiter::new(10.0, 2.0);
        limiter.allow_request("client1", 10.0).await;
        limiter.reset_client("client1").await;
        assert!(limiter.allow_request("client1", 10.0).await);
    }

    #[tokio::test]
    async fn test_bucket_count() {
        let limiter = RateLimiter::new(10.0, 2.0);
        assert_eq!(limiter.bucket_count().await, 0);
        limiter.allow_request("client1", 1.0).await;
        assert_eq!(limiter.bucket_count().await, 1);
        limiter.allow_request("client2", 1.0).await;
        assert_eq!(limiter.bucket_count().await, 2);
    }
}
