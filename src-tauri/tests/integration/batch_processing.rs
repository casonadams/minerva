// Batch Processing Integration Tests (Phase 4 Step 6)

use minerva_lib::inference::batch::{
    BatchInferenceEngine, BatchItem, BatchResponse, BatchResult, BatchStats, BatchTokenizer,
    DetokenizeBatchRequest, InferenceBatchRequest, TokenizeBatchRequest,
};

// Batch Tokenization Tests

#[test]
fn test_batch_item_creation() {
    let item = BatchItem::new("test_id".to_string(), "test_data".to_string());
    assert_eq!(item.id, "test_id");
    assert_eq!(item.data, "test_data");
}

#[test]
fn test_batch_response_creation() {
    let response = BatchResponse::new("test_id".to_string(), "result".to_string(), 100);
    assert_eq!(response.id, "test_id");
    assert_eq!(response.data, "result");
    assert_eq!(response.duration_ms, 100);
}

#[test]
fn test_batch_tokenizer_single_text() {
    let tokenizer = BatchTokenizer::new();
    let requests = vec![BatchItem::new(
        "req1".to_string(),
        TokenizeBatchRequest {
            text: "hello world".to_string(),
        },
    )];

    let responses = tokenizer.encode_batch(requests);
    assert_eq!(responses.len(), 1);
    assert_eq!(responses[0].id, "req1");
    assert!(responses[0].data.count > 0);
}

#[test]
fn test_batch_tokenizer_multiple_texts() {
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
        BatchItem::new(
            "3".to_string(),
            TokenizeBatchRequest {
                text: "batch".to_string(),
            },
        ),
    ];

    let responses = tokenizer.encode_batch(requests);
    assert_eq!(responses.len(), 3);
    assert_eq!(responses[0].id, "1");
    assert_eq!(responses[1].id, "2");
    assert_eq!(responses[2].id, "3");
}

#[test]
fn test_batch_tokenizer_empty_text() {
    let tokenizer = BatchTokenizer::new();
    let requests = vec![BatchItem::new(
        "empty".to_string(),
        TokenizeBatchRequest {
            text: "".to_string(),
        },
    )];

    let responses = tokenizer.encode_batch(requests);
    assert_eq!(responses.len(), 1);
    assert_eq!(responses[0].data.count, 0);
}

#[test]
fn test_batch_tokenizer_long_text() {
    let tokenizer = BatchTokenizer::new();
    let long_text = "a".repeat(1000);
    let requests = vec![BatchItem::new(
        "long".to_string(),
        TokenizeBatchRequest { text: long_text },
    )];

    let responses = tokenizer.encode_batch(requests);
    assert_eq!(responses.len(), 1);
    assert_eq!(responses[0].data.count, 1000);
}

#[test]
fn test_batch_tokenizer_decode_single() {
    let tokenizer = BatchTokenizer::new();
    let requests = vec![BatchItem::new(
        "decode1".to_string(),
        DetokenizeBatchRequest {
            tokens: vec![104, 101, 108, 108, 111], // "hello"
        },
    )];

    let responses = tokenizer.decode_batch(requests);
    assert_eq!(responses.len(), 1);
    assert!(!responses[0].data.text.is_empty());
}

#[test]
fn test_batch_tokenizer_decode_multiple() {
    let tokenizer = BatchTokenizer::new();
    let requests = vec![
        BatchItem::new(
            "1".to_string(),
            DetokenizeBatchRequest {
                tokens: vec![1, 2, 3],
            },
        ),
        BatchItem::new(
            "2".to_string(),
            DetokenizeBatchRequest {
                tokens: vec![4, 5, 6],
            },
        ),
    ];

    let responses = tokenizer.decode_batch(requests);
    assert_eq!(responses.len(), 2);
}

#[test]
fn test_batch_tokenizer_decode_empty() {
    let tokenizer = BatchTokenizer::new();
    let requests = vec![BatchItem::new(
        "empty".to_string(),
        DetokenizeBatchRequest { tokens: vec![] },
    )];

    let responses = tokenizer.decode_batch(requests);
    assert_eq!(responses.len(), 1);
    assert_eq!(responses[0].data.text, "");
}

#[test]
fn test_batch_tokenizer_max_batch_size() {
    let tokenizer = BatchTokenizer::new();
    assert_eq!(tokenizer.max_batch_size(), 1000);
}

#[test]
fn test_batch_tokenizer_optimal_batch_size() {
    let tokenizer = BatchTokenizer::new();
    assert_eq!(tokenizer.optimal_batch_size(), 32);
}

// Batch Inference Tests

#[test]
fn test_batch_inference_single_prompt() {
    let engine = BatchInferenceEngine::new();
    let requests = vec![BatchItem::new(
        "inf1".to_string(),
        InferenceBatchRequest {
            prompt: "What is AI?".to_string(),
            max_tokens: 100,
            temperature: 0.7,
        },
    )];

    let responses = engine.infer_batch(requests);
    assert_eq!(responses.len(), 1);
    assert_eq!(responses[0].id, "inf1");
    assert!(!responses[0].data.text.is_empty());
}

#[test]
fn test_batch_inference_multiple_prompts() {
    let engine = BatchInferenceEngine::new();
    let requests = vec![
        BatchItem::new(
            "1".to_string(),
            InferenceBatchRequest {
                prompt: "prompt1".to_string(),
                max_tokens: 100,
                temperature: 0.7,
            },
        ),
        BatchItem::new(
            "2".to_string(),
            InferenceBatchRequest {
                prompt: "prompt2".to_string(),
                max_tokens: 50,
                temperature: 0.5,
            },
        ),
        BatchItem::new(
            "3".to_string(),
            InferenceBatchRequest {
                prompt: "prompt3".to_string(),
                max_tokens: 200,
                temperature: 0.9,
            },
        ),
    ];

    let responses = engine.infer_batch(requests);
    assert_eq!(responses.len(), 3);
    assert!(responses[0].data.tokens_generated > 0);
    assert!(responses[1].data.tokens_generated > 0);
    assert!(responses[2].data.tokens_generated > 0);
}

#[test]
fn test_batch_inference_temperature_effects() {
    let engine = BatchInferenceEngine::new();

    // Low temperature request
    let low_temp = vec![BatchItem::new(
        "low".to_string(),
        InferenceBatchRequest {
            prompt: "test".to_string(),
            max_tokens: 100,
            temperature: 0.1,
        },
    )];

    // High temperature request
    let high_temp = vec![BatchItem::new(
        "high".to_string(),
        InferenceBatchRequest {
            prompt: "test".to_string(),
            max_tokens: 100,
            temperature: 0.9,
        },
    )];

    let low_responses = engine.infer_batch(low_temp);
    let high_responses = engine.infer_batch(high_temp);

    // Mock model: higher temperature produces fewer tokens (1.0 - temp * 0.1)
    assert!(low_responses[0].data.tokens_generated > high_responses[0].data.tokens_generated);
}

#[test]
fn test_batch_inference_max_batch_size() {
    let engine = BatchInferenceEngine::new();
    assert_eq!(engine.max_batch_size(), 100);
}

#[test]
fn test_batch_inference_optimal_batch_size() {
    let engine = BatchInferenceEngine::new();
    assert_eq!(engine.optimal_batch_size(), 8);
}

// Batch Statistics Tests

#[test]
fn test_batch_stats_creation() {
    let stats = BatchStats::new(10, 1000);
    assert_eq!(stats.total_items, 10);
    assert_eq!(stats.total_duration_ms, 1000);
    assert!(stats.avg_item_time_ms > 0.0);
    assert!(stats.items_per_second > 0.0);
}

#[test]
fn test_batch_stats_single_item() {
    let stats = BatchStats::new(1, 100);
    assert_eq!(stats.total_items, 1);
    assert_eq!(stats.avg_item_time_ms, 100.0);
}

#[test]
fn test_batch_stats_many_items() {
    let stats = BatchStats::new(100, 500);
    assert_eq!(stats.total_items, 100);
    assert!(stats.avg_item_time_ms < 10.0); // Average 5ms per item
}

#[test]
fn test_batch_stats_speedup_calculation() {
    let stats = BatchStats::new(10, 1000);
    let speedup = stats.speedup_vs_single(200.0);
    assert!(speedup > 1.0); // Batch should be faster than single
}

#[test]
fn test_batch_stats_zero_items() {
    let stats = BatchStats::new(0, 0);
    assert_eq!(stats.total_items, 0);
    assert_eq!(stats.items_per_second, 0.0);
}

// Batch Result Tests

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
        BatchResponse::new("id1".to_string(), "data1".to_string(), 50),
        BatchResponse::new("id2".to_string(), "data2".to_string(), 75),
    ];
    let result = BatchResult::new(responses);

    assert!(result.get_by_id("id1").is_some());
    assert!(result.get_by_id("id2").is_some());
    assert!(result.get_by_id("id3").is_none());
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
fn test_batch_result_statistics() {
    let responses = vec![
        BatchResponse::new("1".to_string(), "data".to_string(), 100),
        BatchResponse::new("2".to_string(), "data".to_string(), 200),
        BatchResponse::new("3".to_string(), "data".to_string(), 300),
    ];
    let result = BatchResult::new(responses);

    assert_eq!(result.stats.total_items, 3);
    assert_eq!(result.stats.total_duration_ms, 600);
    assert!(result.stats.avg_item_time_ms > 0.0);
}

// Integration Tests

#[test]
fn test_batch_tokenizer_encode_decode_roundtrip() {
    let tokenizer = BatchTokenizer::new();

    // Encode
    let encode_requests = vec![BatchItem::new(
        "round".to_string(),
        TokenizeBatchRequest {
            text: "hello".to_string(),
        },
    )];
    let encode_responses = tokenizer.encode_batch(encode_requests);
    assert_eq!(encode_responses.len(), 1);

    // Decode using the encoded tokens
    let tokens = encode_responses[0].data.tokens.clone();
    let decode_requests = vec![BatchItem::new(
        "round".to_string(),
        DetokenizeBatchRequest { tokens },
    )];
    let decode_responses = tokenizer.decode_batch(decode_requests);

    assert_eq!(decode_responses.len(), 1);
    assert!(!decode_responses[0].data.text.is_empty());
}

#[test]
fn test_batch_large_dataset_tokenization() {
    let tokenizer = BatchTokenizer::new();
    let mut requests = Vec::new();

    // Create 100 requests
    for i in 0..100 {
        requests.push(BatchItem::new(
            format!("req_{}", i),
            TokenizeBatchRequest {
                text: format!("text_{}", i),
            },
        ));
    }

    let responses = tokenizer.encode_batch(requests);
    assert_eq!(responses.len(), 100);

    // Verify all responses have IDs
    for (i, response) in responses.iter().enumerate() {
        assert_eq!(response.id, format!("req_{}", i));
    }
}

#[test]
fn test_batch_inference_with_varying_parameters() {
    let engine = BatchInferenceEngine::new();
    let requests = vec![
        BatchItem::new(
            "max_small".to_string(),
            InferenceBatchRequest {
                prompt: "test".to_string(),
                max_tokens: 10,
                temperature: 0.5,
            },
        ),
        BatchItem::new(
            "max_large".to_string(),
            InferenceBatchRequest {
                prompt: "test".to_string(),
                max_tokens: 500,
                temperature: 0.5,
            },
        ),
    ];

    let responses = engine.infer_batch(requests);
    assert_eq!(responses.len(), 2);

    // Verify both requests processed
    assert_eq!(responses[0].id, "max_small");
    assert_eq!(responses[1].id, "max_large");
}

#[test]
fn test_batch_tokenizer_default_creation() {
    let tokenizer = BatchTokenizer::new();
    assert_eq!(tokenizer.optimal_batch_size(), 32);
}

#[test]
fn test_batch_inference_engine_default_creation() {
    let engine = BatchInferenceEngine::new();
    assert_eq!(engine.optimal_batch_size(), 8);
}
