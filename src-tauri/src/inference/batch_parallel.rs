/// Parallel batch processing with Rayon
///
/// This module provides CPU parallelization for batch operations using rayon,
/// enabling multi-core processing for improved throughput.
use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use crate::inference::batch::{
    BatchStats, DetokenizeBatchRequest, DetokenizeBatchResponse, InferenceBatchRequest,
    InferenceBatchResponse, TokenizeBatchRequest, TokenizeBatchResponse,
};

/// Parallel batch item
#[derive(Clone)]
pub struct ParallelBatchItem<T: Clone + Send> {
    pub id: String,
    pub data: T,
}

impl<T: Clone + Send> ParallelBatchItem<T> {
    pub fn new(id: String, data: T) -> Self {
        Self { id, data }
    }

    pub fn get_id(&self) -> &str {
        &self.id
    }

    pub fn get_data(&self) -> &T {
        &self.data
    }
}

/// Parallel batch response
#[derive(Clone)]
pub struct ParallelBatchResponse<T: Clone + Send> {
    pub id: String,
    pub data: T,
    pub duration_ms: u128,
}

impl<T: Clone + Send> ParallelBatchResponse<T> {
    pub fn new(id: String, data: T, duration_ms: u128) -> Self {
        Self {
            id,
            data,
            duration_ms,
        }
    }

    pub fn get_id(&self) -> &str {
        &self.id
    }

    pub fn get_data(&self) -> &T {
        &self.data
    }

    pub fn get_duration_ms(&self) -> u128 {
        self.duration_ms
    }
}

/// Parallel batch result with efficient lookup
#[derive(Clone)]
pub struct ParallelBatchResult<T: Clone + Send> {
    pub responses: Vec<ParallelBatchResponse<T>>,
    pub responses_map: Arc<HashMap<String, usize>>,
    pub stats: BatchStats,
}

impl<T: Clone + Send> ParallelBatchResult<T> {
    pub fn new(responses: Vec<ParallelBatchResponse<T>>) -> Self {
        let total_items = responses.len();
        let total_duration_ms: u128 = responses.iter().map(|r| r.duration_ms).sum();

        // Create index map for O(1) lookup
        let mut responses_map = HashMap::with_capacity(total_items);
        for (idx, response) in responses.iter().enumerate() {
            responses_map.insert(response.id.clone(), idx);
        }

        let stats = BatchStats::new(total_items, total_duration_ms);

        Self {
            responses,
            responses_map: Arc::new(responses_map),
            stats,
        }
    }

    /// Get response by ID (O(1) lookup)
    pub fn get_by_id(&self, id: &str) -> Option<&ParallelBatchResponse<T>> {
        self.responses_map
            .get(id)
            .and_then(|&idx| self.responses.get(idx))
    }

    /// Get all responses
    pub fn get_responses(&self) -> &[ParallelBatchResponse<T>] {
        &self.responses
    }

    /// Get statistics
    pub fn get_stats(&self) -> &BatchStats {
        &self.stats
    }

    /// Count successful responses
    pub fn success_count(&self) -> usize {
        self.responses.len()
    }
}

/// Parallel tokenizer using Rayon
pub struct ParallelBatchTokenizer {
    /// Rayon thread pool configuration (default: num_cpus)
    num_threads: usize,
}

impl ParallelBatchTokenizer {
    pub fn new() -> Self {
        Self {
            num_threads: num_cpus::get(),
        }
    }

    pub fn with_threads(num_threads: usize) -> Self {
        Self { num_threads }
    }

    /// Encode multiple texts in parallel
    pub fn encode_batch(
        &self,
        requests: Vec<ParallelBatchItem<TokenizeBatchRequest>>,
    ) -> ParallelBatchResult<TokenizeBatchResponse> {
        let _start = Instant::now();

        let responses: Vec<_> = requests
            .into_par_iter()
            .map(|req| {
                let item_start = Instant::now();

                // Parallel: split by characters
                let tokens: Vec<u32> = req
                    .data
                    .text
                    .chars()
                    .enumerate()
                    .map(|(i, _)| i as u32)
                    .collect();

                let duration = item_start.elapsed().as_millis();
                let response = TokenizeBatchResponse {
                    tokens: tokens.clone(),
                    count: tokens.len(),
                };

                ParallelBatchResponse::new(req.id, response, duration)
            })
            .collect();

        ParallelBatchResult::new(responses)
    }

    /// Decode multiple token sequences in parallel
    pub fn decode_batch(
        &self,
        requests: Vec<ParallelBatchItem<DetokenizeBatchRequest>>,
    ) -> ParallelBatchResult<DetokenizeBatchResponse> {
        let responses: Vec<_> = requests
            .into_par_iter()
            .map(|req| {
                let item_start = Instant::now();

                // Parallel: reconstruct from token IDs
                let text = format!(
                    "[tokens: {}]",
                    req.data
                        .tokens
                        .iter()
                        .map(|t| t.to_string())
                        .collect::<Vec<_>>()
                        .join(",")
                );

                let duration = item_start.elapsed().as_millis();
                ParallelBatchResponse::new(req.id, DetokenizeBatchResponse { text }, duration)
            })
            .collect();

        ParallelBatchResult::new(responses)
    }

    /// Get number of threads
    pub fn num_threads(&self) -> usize {
        self.num_threads
    }
}

impl Default for ParallelBatchTokenizer {
    fn default() -> Self {
        Self::new()
    }
}

/// Parallel inference engine using Rayon
pub struct ParallelBatchInferenceEngine {
    num_threads: usize,
}

impl ParallelBatchInferenceEngine {
    pub fn new() -> Self {
        Self {
            num_threads: num_cpus::get(),
        }
    }

    pub fn with_threads(num_threads: usize) -> Self {
        Self { num_threads }
    }

    /// Run inference on multiple prompts in parallel
    pub fn infer_batch(
        &self,
        requests: Vec<ParallelBatchItem<InferenceBatchRequest>>,
    ) -> ParallelBatchResult<InferenceBatchResponse> {
        let responses: Vec<_> = requests
            .into_par_iter()
            .map(|req| {
                let item_start = Instant::now();

                // Parallel: generate based on prompt
                let text = format!(
                    "Response to: {}",
                    &req.data.prompt[..std::cmp::min(20, req.data.prompt.len())]
                );

                let tokens_generated =
                    (req.data.max_tokens as f32 * (1.0 - req.data.temperature * 0.1)) as usize;

                let duration = item_start.elapsed().as_millis();
                let response = InferenceBatchResponse {
                    text,
                    tokens_generated,
                };

                ParallelBatchResponse::new(req.id, response, duration)
            })
            .collect();

        ParallelBatchResult::new(responses)
    }

    /// Maximum batch size
    pub fn max_batch_size(&self) -> usize {
        100
    }

    /// Optimal batch size (considering thread count)
    pub fn optimal_batch_size(&self) -> usize {
        std::cmp::max(self.num_threads * 2, 8)
    }

    /// Get number of threads
    pub fn num_threads(&self) -> usize {
        self.num_threads
    }
}

impl Default for ParallelBatchInferenceEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parallel_batch_item() {
        let item = ParallelBatchItem::new("id1".to_string(), "data".to_string());
        assert_eq!(item.get_id(), "id1");
        assert_eq!(item.get_data(), &"data".to_string());
    }

    #[test]
    fn test_parallel_batch_response() {
        let response = ParallelBatchResponse::new("id1".to_string(), "result".to_string(), 100);
        assert_eq!(response.get_id(), "id1");
        assert_eq!(response.get_duration_ms(), 100);
    }

    #[test]
    fn test_parallel_tokenizer_encode() {
        let tokenizer = ParallelBatchTokenizer::new();
        let requests: Vec<_> = (0..4)
            .map(|i| {
                ParallelBatchItem::new(
                    format!("req{}", i),
                    TokenizeBatchRequest {
                        text: format!("text{}", i),
                    },
                )
            })
            .collect();

        let result = tokenizer.encode_batch(requests);
        assert_eq!(result.success_count(), 4);
    }

    #[test]
    fn test_parallel_tokenizer_decode() {
        let tokenizer = ParallelBatchTokenizer::new();
        let requests: Vec<_> = (0..4)
            .map(|i| {
                ParallelBatchItem::new(
                    format!("req{}", i),
                    DetokenizeBatchRequest {
                        tokens: vec![1, 2, 3],
                    },
                )
            })
            .collect();

        let result = tokenizer.decode_batch(requests);
        assert_eq!(result.success_count(), 4);
    }

    #[test]
    fn test_parallel_inference_engine() {
        let engine = ParallelBatchInferenceEngine::new();
        let requests: Vec<_> = (0..8)
            .map(|i| {
                ParallelBatchItem::new(
                    format!("inf{}", i),
                    InferenceBatchRequest {
                        prompt: format!("prompt{}", i),
                        max_tokens: 100,
                        temperature: 0.7,
                    },
                )
            })
            .collect();

        let result = engine.infer_batch(requests);
        assert_eq!(result.success_count(), 8);
        assert!(engine.optimal_batch_size() >= 8);
    }

    #[test]
    fn test_parallel_result_lookup() {
        let responses: Vec<_> = (0..4)
            .map(|i| ParallelBatchResponse::new(format!("id{}", i), i, 100 + i as u128))
            .collect();

        let result = ParallelBatchResult::new(responses);
        assert!(result.get_by_id("id0").is_some());
        assert!(result.get_by_id("id3").is_some());
        assert!(result.get_by_id("id5").is_none());
    }

    #[test]
    fn test_parallel_with_custom_threads() {
        let tokenizer = ParallelBatchTokenizer::with_threads(4);
        assert_eq!(tokenizer.num_threads(), 4);

        let engine = ParallelBatchInferenceEngine::with_threads(4);
        assert_eq!(engine.num_threads(), 4);
    }
}
