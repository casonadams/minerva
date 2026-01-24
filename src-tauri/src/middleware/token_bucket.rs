use std::time::Instant;

/// Token bucket for rate limiting
#[derive(Clone)]
pub struct TokenBucket {
    max_tokens: f64,
    refill_rate: f64,
    tokens: f64,
    pub last_refill: Instant,
}

impl TokenBucket {
    pub fn new(max_tokens: f64, refill_rate: f64) -> Self {
        Self {
            max_tokens,
            refill_rate,
            tokens: max_tokens,
            last_refill: Instant::now(),
        }
    }

    fn refill(&mut self) {
        let elapsed = self.last_refill.elapsed().as_secs_f64();
        let new_tokens = self.refill_rate * elapsed;
        self.tokens = (self.tokens + new_tokens).min(self.max_tokens);
        self.last_refill = Instant::now();
    }

    pub fn try_take(&mut self, amount: f64) -> bool {
        self.refill();
        if self.tokens >= amount {
            self.tokens -= amount;
            true
        } else {
            false
        }
    }

    pub fn available(&mut self) -> f64 {
        self.refill();
        self.tokens
    }

    pub fn retry_seconds(&mut self, tokens: f64) -> u64 {
        self.refill();
        let needed = (tokens - self.tokens).max(0.0);
        (needed / self.refill_rate).ceil() as u64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_creation() {
        let bucket = TokenBucket::new(10.0, 2.0);
        assert_eq!(bucket.max_tokens, 10.0);
        assert_eq!(bucket.refill_rate, 2.0);
        assert_eq!(bucket.tokens, 10.0);
    }

    #[test]
    fn test_take_success() {
        let mut bucket = TokenBucket::new(10.0, 2.0);
        assert!(bucket.try_take(5.0));
        assert_eq!(bucket.tokens, 5.0);
    }

    #[test]
    fn test_take_failure() {
        let mut bucket = TokenBucket::new(10.0, 2.0);
        assert!(bucket.try_take(10.0));
        assert!(!bucket.try_take(1.0));
        assert!(bucket.tokens < 0.01);
    }

    #[test]
    fn test_refill() {
        let mut bucket = TokenBucket::new(10.0, 2.0);
        bucket.tokens = 0.0;
        std::thread::sleep(Duration::from_millis(600));
        bucket.refill();
        assert!(bucket.tokens >= 1.0);
    }

    #[test]
    fn test_max_cap() {
        let mut bucket = TokenBucket::new(10.0, 2.0);
        bucket.tokens = 0.0;
        std::thread::sleep(Duration::from_secs(10));
        bucket.refill();
        assert!(bucket.tokens <= 10.0);
    }

    #[test]
    fn test_available() {
        let mut bucket = TokenBucket::new(10.0, 2.0);
        bucket.try_take(3.0);
        let avail = bucket.available();
        assert!((7.0..=10.0).contains(&avail));
    }
}
