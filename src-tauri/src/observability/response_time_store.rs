use parking_lot::RwLock;
use std::sync::Arc;
use std::time::Duration;

/// Manages response time history with bounded storage
pub struct ResponseTimeStore {
    times: Arc<RwLock<Vec<Duration>>>,
}

impl ResponseTimeStore {
    /// Create new response time store
    pub fn new() -> Self {
        Self {
            times: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Store response time with bounded growth
    pub fn store(&self, time: Duration) {
        let mut times = self.times.write();
        times.push(time);
        // Keep only last 10000 measurements to avoid unbounded memory
        if times.len() > 10000 {
            times.drain(0..5000);
        }
    }

    /// Get current response times
    pub fn get_times(&self) -> Vec<Duration> {
        self.times.read().clone()
    }

    /// Clear all response times
    pub fn clear(&self) {
        self.times.write().clear();
    }
}

impl Clone for ResponseTimeStore {
    fn clone(&self) -> Self {
        Self {
            times: Arc::clone(&self.times),
        }
    }
}

impl Default for ResponseTimeStore {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_store_single_time() {
        let store = ResponseTimeStore::new();
        store.store(Duration::from_millis(100));
        assert_eq!(store.get_times().len(), 1);
    }

    #[test]
    fn test_store_multiple_times() {
        let store = ResponseTimeStore::new();
        for i in 0..10 {
            store.store(Duration::from_millis(i));
        }
        assert_eq!(store.get_times().len(), 10);
    }

    #[test]
    fn test_bounded_growth() {
        let store = ResponseTimeStore::new();
        for i in 0..15000 {
            store.store(Duration::from_millis(i as u64));
        }
        // Should be around 10000 after draining
        assert!(store.get_times().len() <= 10000);
        assert!(store.get_times().len() >= 5000);
    }

    #[test]
    fn test_clear() {
        let store = ResponseTimeStore::new();
        store.store(Duration::from_millis(100));
        store.clear();
        assert_eq!(store.get_times().len(), 0);
    }

    #[test]
    fn test_clone_shares_state() {
        let store1 = ResponseTimeStore::new();
        let store2 = store1.clone();
        store1.store(Duration::from_millis(100));
        assert_eq!(store2.get_times().len(), 1);
    }
}
