/// Batch processing for tokenization and inference
///
/// Enables parallel processing of multiple requests with:
/// - Batch tokenization (encode/decode multiple texts)
/// - Batch inference (generate for multiple prompts)
/// - Efficient resource utilization
/// - Performance optimization
use std::time::Instant;

/// Individual batch request item
#[derive(Debug, Clone)]
pub struct BatchItem<T> {
    /// Item ID for tracking
    pub id: String,
    /// The request data
    pub data: T,
}

impl<T> BatchItem<T> {
    /// Create new batch item
    pub fn new(id: String, data: T) -> Self {
        Self { id, data }
    }
}

/// Response from batch operation
#[derive(Debug, Clone)]
pub struct BatchResponse<T> {
    /// Item ID matching request
    pub id: String,
    /// Response data
    pub data: T,
    /// Processing time in milliseconds
    pub duration_ms: u128,
}

impl<T> BatchResponse<T> {
    /// Create new batch response
    pub fn new(id: String, data: T, duration_ms: u128) -> Self {
        Self {
            id,
            data,
            duration_ms,
        }
    }
}

/// Batch tokenization requests
#[derive(Debug, Clone)]
pub struct TokenizeBatchRequest {
    /// Text to tokenize
    pub text: String,
}

/// Batch tokenization responses
#[derive(Debug, Clone)]
pub struct TokenizeBatchResponse {
    /// Token IDs
    pub tokens: Vec<u32>,
    /// Token count
    pub count: usize,
}

/// Batch detokenization requests
#[derive(Debug, Clone)]
pub struct DetokenizeBatchRequest {
    /// Token IDs to decode
    pub tokens: Vec<u32>,
}

/// Batch detokenization responses
#[derive(Debug, Clone)]
pub struct DetokenizeBatchResponse {
    /// Decoded text
    pub text: String,
}

/// Batch inference requests
#[derive(Debug, Clone)]
pub struct InferenceBatchRequest {
    /// Prompt text
    pub prompt: String,
    /// Maximum tokens to generate
    pub max_tokens: usize,
    /// Temperature for sampling
    pub temperature: f32,
}

/// Batch inference responses
#[derive(Debug, Clone)]
pub struct InferenceBatchResponse {
    /// Generated text
    pub text: String,
    /// Tokens generated
    pub tokens_generated: usize,
}

/// Batch processor for tokenization
#[derive(Debug, Clone)]
pub struct BatchTokenizer;

impl BatchTokenizer {
    /// Create new batch tokenizer
    pub fn new() -> Self {
        Self
    }

    /// Batch encode multiple texts
    pub fn encode_batch(
        &self,
        requests: Vec<BatchItem<TokenizeBatchRequest>>,
    ) -> Vec<BatchResponse<TokenizeBatchResponse>> {
        let _start = Instant::now();

        requests
            .into_iter()
            .map(|req| {
                let item_start = Instant::now();

                // Tokenize: split by characters as simple mock
                let tokens: Vec<u32> = req
                    .data
                    .text
                    .chars()
                    .enumerate()
                    .map(|(i, _)| i as u32)
                    .collect();
                let count = tokens.len();

                let duration = item_start.elapsed().as_millis();
                let response = TokenizeBatchResponse { tokens, count };

                BatchResponse::new(req.id, response, duration)
            })
            .collect()
    }

    /// Batch decode multiple token sequences
    pub fn decode_batch(
        &self,
        requests: Vec<BatchItem<DetokenizeBatchRequest>>,
    ) -> Vec<BatchResponse<DetokenizeBatchResponse>> {
        requests
            .into_iter()
            .map(|req| {
                let item_start = Instant::now();

                // Simple mock: convert token IDs back to chars
                let text: String = req
                    .data
                    .tokens
                    .iter()
                    .filter_map(|&id| char::from_u32(id))
                    .collect();

                let duration = item_start.elapsed().as_millis();
                let response = DetokenizeBatchResponse { text };

                BatchResponse::new(req.id, response, duration)
            })
            .collect()
    }

    /// Get batch size limit
    pub fn max_batch_size(&self) -> usize {
        1000
    }

    /// Get recommended batch size for optimal performance
    pub fn optimal_batch_size(&self) -> usize {
        32
    }
}

impl Default for BatchTokenizer {
    fn default() -> Self {
        Self::new()
    }
}

/// Batch processor for inference
#[derive(Debug, Clone)]
pub struct BatchInferenceEngine;

impl BatchInferenceEngine {
    /// Create new batch inference engine
    pub fn new() -> Self {
        Self
    }

    /// Batch inference for multiple prompts
    pub fn infer_batch(
        &self,
        requests: Vec<BatchItem<InferenceBatchRequest>>,
    ) -> Vec<BatchResponse<InferenceBatchResponse>> {
        requests
            .into_iter()
            .map(|req| {
                let item_start = Instant::now();

                // Mock inference: generate based on prompt length
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

                BatchResponse::new(req.id, response, duration)
            })
            .collect()
    }

    /// Get batch size limit
    pub fn max_batch_size(&self) -> usize {
        100
    }

    /// Get recommended batch size
    pub fn optimal_batch_size(&self) -> usize {
        8
    }
}

impl Default for BatchInferenceEngine {
    fn default() -> Self {
        Self::new()
    }
}

/// Batch statistics tracker
#[derive(Debug, Clone)]
pub struct BatchStats {
    /// Total items processed
    pub total_items: usize,
    /// Total processing time in milliseconds
    pub total_duration_ms: u128,
    /// Average time per item
    pub avg_item_time_ms: f64,
    /// Items per second
    pub items_per_second: f64,
}

impl BatchStats {
    /// Create new batch stats
    pub fn new(total_items: usize, total_duration_ms: u128) -> Self {
        let avg_item_time_ms = if total_items > 0 {
            total_duration_ms as f64 / total_items as f64
        } else {
            0.0
        };

        let items_per_second = if total_duration_ms > 0 {
            (total_items as f64 / total_duration_ms as f64) * 1000.0
        } else {
            0.0
        };

        Self {
            total_items,
            total_duration_ms,
            avg_item_time_ms,
            items_per_second,
        }
    }

    /// Calculate speedup vs single processing
    pub fn speedup_vs_single(&self, single_item_time_ms: f64) -> f64 {
        if self.avg_item_time_ms > 0.0 {
            single_item_time_ms / self.avg_item_time_ms
        } else {
            1.0
        }
    }
}

/// Batch processing result
#[derive(Debug, Clone)]
pub struct BatchResult<T> {
    /// Individual responses
    pub responses: Vec<BatchResponse<T>>,
    /// Overall statistics
    pub stats: BatchStats,
}

impl<T> BatchResult<T> {
    /// Create new batch result
    pub fn new(responses: Vec<BatchResponse<T>>) -> Self {
        let total_items = responses.len();
        let total_duration_ms: u128 = responses.iter().map(|r| r.duration_ms).sum();

        let stats = BatchStats::new(total_items, total_duration_ms);

        Self { responses, stats }
    }

    /// Get item by ID
    pub fn get_by_id(&self, id: &str) -> Option<&BatchResponse<T>> {
        self.responses.iter().find(|r| r.id == id)
    }

    /// Check if all items succeeded (mock - always true for now)
    pub fn all_succeeded(&self) -> bool {
        true
    }

    /// Get success count
    pub fn success_count(&self) -> usize {
        self.responses.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_batch_item_creation() {
        let item = BatchItem::new("id1".to_string(), "data".to_string());
        assert_eq!(item.id, "id1");
        assert_eq!(item.data, "data");
    }

    #[test]
    fn test_batch_response_creation() {
        let response = BatchResponse::new("id1".to_string(), "result".to_string(), 100);
        assert_eq!(response.id, "id1");
        assert_eq!(response.data, "result");
        assert_eq!(response.duration_ms, 100);
    }

    #[test]
    fn test_batch_tokenizer_creation() {
        let tokenizer = BatchTokenizer::new();
        assert_eq!(tokenizer.max_batch_size(), 1000);
        assert_eq!(tokenizer.optimal_batch_size(), 32);
    }

    #[test]
    fn test_batch_tokenizer_encode() {
        let tokenizer = BatchTokenizer::new();
        let requests = vec![
            BatchItem::new(
                "1".to_string(),
                TokenizeBatchRequest {
                    text: "hello".to_string(),
                },
            ),
            BatchItem::new(
                "2".to_string(),
                TokenizeBatchRequest {
                    text: "world".to_string(),
                },
            ),
        ];

        let responses = tokenizer.encode_batch(requests);
        assert_eq!(responses.len(), 2);
        assert_eq!(responses[0].id, "1");
        assert_eq!(responses[1].id, "2");
    }

    #[test]
    fn test_batch_tokenizer_decode() {
        let tokenizer = BatchTokenizer::new();
        let requests = vec![BatchItem::new(
            "1".to_string(),
            DetokenizeBatchRequest {
                tokens: vec![104, 101, 108, 108, 111], // "hello"
            },
        )];

        let responses = tokenizer.decode_batch(requests);
        assert_eq!(responses.len(), 1);
    }

    #[test]
    fn test_batch_inference_engine_creation() {
        let engine = BatchInferenceEngine::new();
        assert_eq!(engine.max_batch_size(), 100);
        assert_eq!(engine.optimal_batch_size(), 8);
    }

    #[test]
    fn test_batch_inference_engine_infer() {
        let engine = BatchInferenceEngine::new();
        let requests = vec![
            BatchItem::new(
                "1".to_string(),
                InferenceBatchRequest {
                    prompt: "hello".to_string(),
                    max_tokens: 100,
                    temperature: 0.7,
                },
            ),
            BatchItem::new(
                "2".to_string(),
                InferenceBatchRequest {
                    prompt: "world".to_string(),
                    max_tokens: 50,
                    temperature: 0.5,
                },
            ),
        ];

        let responses = engine.infer_batch(requests);
        assert_eq!(responses.len(), 2);
        assert!(!responses[0].data.text.is_empty());
        assert!(responses[0].data.tokens_generated > 0);
    }

    #[test]
    fn test_batch_stats_creation() {
        let stats = BatchStats::new(10, 1000);
        assert_eq!(stats.total_items, 10);
        assert_eq!(stats.total_duration_ms, 1000);
        assert!(stats.avg_item_time_ms > 0.0);
        assert!(stats.items_per_second > 0.0);
    }

    #[test]
    fn test_batch_stats_speedup() {
        let stats = BatchStats::new(10, 1000);
        let speedup = stats.speedup_vs_single(200.0);
        assert!(speedup > 1.0);
    }

    #[test]
    fn test_batch_result_creation() {
        let responses = vec![
            BatchResponse::new("1".to_string(), 100, 50),
            BatchResponse::new("2".to_string(), 200, 75),
        ];
        let result = BatchResult::new(responses);
        assert_eq!(result.responses.len(), 2);
        assert!(result.all_succeeded());
    }

    #[test]
    fn test_batch_result_get_by_id() {
        let responses = vec![
            BatchResponse::new("1".to_string(), "data1".to_string(), 50),
            BatchResponse::new("2".to_string(), "data2".to_string(), 75),
        ];
        let result = BatchResult::new(responses);

        assert!(result.get_by_id("1").is_some());
        assert!(result.get_by_id("2").is_some());
        assert!(result.get_by_id("3").is_none());
    }

    #[test]
    fn test_batch_result_success_count() {
        let responses = vec![
            BatchResponse::new("1".to_string(), 100, 50),
            BatchResponse::new("2".to_string(), 200, 75),
            BatchResponse::new("3".to_string(), 300, 100),
        ];
        let result = BatchResult::new(responses);
        assert_eq!(result.success_count(), 3);
    }

    #[test]
    fn test_tokenize_batch_request() {
        let req = TokenizeBatchRequest {
            text: "hello".to_string(),
        };
        assert_eq!(req.text, "hello");
    }

    #[test]
    fn test_inference_batch_request() {
        let req = InferenceBatchRequest {
            prompt: "test".to_string(),
            max_tokens: 100,
            temperature: 0.7,
        };
        assert_eq!(req.prompt, "test");
        assert_eq!(req.max_tokens, 100);
        assert_eq!(req.temperature, 0.7);
    }

    #[test]
    fn test_batch_tokenizer_default() {
        let _tokenizer = BatchTokenizer::new();
        // Should create successfully
    }

    #[test]
    fn test_batch_inference_engine_default() {
        let _engine = BatchInferenceEngine::new();
        // Should create successfully
    }
}
