// Streaming Response Tests - SSE and token streaming validation

use minerva_lib::streaming::{
    ChatCompletionStreamEvent, StreamingChoice, StreamingConfig, StreamingDelta, StreamingValidator,
};

#[test]
fn test_streaming_config_default_values() {
    let config = StreamingConfig::default();

    assert_eq!(
        config.keep_alive_ms, 15000,
        "Keep-alive should be 15 seconds"
    );
    assert_eq!(
        config.max_chunk_size, 50,
        "Max chunk size should be 50 tokens"
    );
}

#[test]
fn test_streaming_config_custom_values() {
    let config = StreamingConfig {
        keep_alive_ms: 30000,
        max_chunk_size: 100,
    };

    assert_eq!(config.keep_alive_ms, 30000);
    assert_eq!(config.max_chunk_size, 100);
}

#[test]
fn test_streaming_delta_with_content() {
    let delta = StreamingDelta {
        content: Some("hello world".to_string()),
        finish_reason: None,
    };

    assert_eq!(
        delta.content,
        Some("hello world".to_string()),
        "Content should match"
    );
    assert!(delta.finish_reason.is_none(), "No finish reason yet");
}

#[test]
fn test_streaming_delta_finish() {
    let delta = StreamingDelta {
        content: None,
        finish_reason: Some("stop".to_string()),
    };

    assert!(delta.content.is_none(), "Content should be None on finish");
    assert_eq!(
        delta.finish_reason,
        Some("stop".to_string()),
        "Finish reason should be stop"
    );
}

#[test]
fn test_streaming_choice_structure() {
    let choice = StreamingChoice {
        index: 0,
        delta: StreamingDelta {
            content: Some("token".to_string()),
            finish_reason: None,
        },
        finish_reason: None,
    };

    assert_eq!(choice.index, 0);
    assert_eq!(choice.delta.content, Some("token".to_string()));
}

#[test]
fn test_streaming_event_creation() {
    let event = ChatCompletionStreamEvent {
        id: "chatcmpl-123".to_string(),
        object: "text_completion.chunk".to_string(),
        created: 1234567890,
        model: "gpt-4".to_string(),
        choices: vec![StreamingChoice {
            index: 0,
            delta: StreamingDelta {
                content: Some("hello".to_string()),
                finish_reason: None,
            },
            finish_reason: None,
        }],
    };

    assert_eq!(event.id, "chatcmpl-123");
    assert_eq!(event.object, "text_completion.chunk");
    assert_eq!(event.created, 1234567890);
    assert_eq!(event.model, "gpt-4");
    assert_eq!(event.choices.len(), 1);
}

#[test]
fn test_streaming_event_multiple_choices() {
    let event = ChatCompletionStreamEvent {
        id: "test".to_string(),
        object: "text_completion.chunk".to_string(),
        created: 100,
        model: "test-model".to_string(),
        choices: vec![
            StreamingChoice {
                index: 0,
                delta: StreamingDelta {
                    content: Some("token1".to_string()),
                    finish_reason: None,
                },
                finish_reason: None,
            },
            StreamingChoice {
                index: 1,
                delta: StreamingDelta {
                    content: Some("token2".to_string()),
                    finish_reason: None,
                },
                finish_reason: None,
            },
        ],
    };

    assert_eq!(event.choices.len(), 2, "Should have 2 choices");
    assert_eq!(event.choices[0].index, 0);
    assert_eq!(event.choices[1].index, 1);
}

#[test]
fn test_streaming_event_finish_event() {
    let event = ChatCompletionStreamEvent {
        id: "test".to_string(),
        object: "text_completion.chunk".to_string(),
        created: 100,
        model: "gpt-4".to_string(),
        choices: vec![StreamingChoice {
            index: 0,
            delta: StreamingDelta {
                content: None,
                finish_reason: Some("stop".to_string()),
            },
            finish_reason: Some("stop".to_string()),
        }],
    };

    let choice = &event.choices[0];
    assert!(
        choice.delta.content.is_none(),
        "Finish event has no content"
    );
    assert_eq!(choice.finish_reason, Some("stop".to_string()));
}

#[test]
fn test_streaming_event_serialization() {
    let event = ChatCompletionStreamEvent {
        id: "test-id".to_string(),
        object: "text_completion.chunk".to_string(),
        created: 100,
        model: "gpt-4".to_string(),
        choices: vec![StreamingChoice {
            index: 0,
            delta: StreamingDelta {
                content: Some("hello".to_string()),
                finish_reason: None,
            },
            finish_reason: None,
        }],
    };

    let json = serde_json::to_string(&event).expect("Should serialize");
    assert!(json.contains("test-id"), "Should contain event ID");
    assert!(
        json.contains("text_completion.chunk"),
        "Should contain object type"
    );
    assert!(json.contains("hello"), "Should contain token content");
}

#[test]
fn test_streaming_validator_chunk_size_valid() {
    assert!(
        StreamingValidator::validate_chunk_size(1).is_ok(),
        "Size 1 should be valid"
    );
    assert!(
        StreamingValidator::validate_chunk_size(50).is_ok(),
        "Size 50 should be valid"
    );
    assert!(
        StreamingValidator::validate_chunk_size(1000).is_ok(),
        "Size 1000 should be valid"
    );
}

#[test]
fn test_streaming_validator_chunk_size_invalid() {
    assert!(
        StreamingValidator::validate_chunk_size(0).is_err(),
        "Size 0 should fail"
    );
    assert!(
        StreamingValidator::validate_chunk_size(1001).is_err(),
        "Size > 1000 should fail"
    );
}

#[test]
fn test_streaming_validator_chunk_size_boundary() {
    assert!(
        StreamingValidator::validate_chunk_size(1).is_ok(),
        "Minimum valid"
    );
    assert!(
        StreamingValidator::validate_chunk_size(1000).is_ok(),
        "Maximum valid"
    );
    assert!(
        StreamingValidator::validate_chunk_size(1001).is_err(),
        "Just over maximum"
    );
}

#[test]
fn test_streaming_validator_stream_parameter() {
    // Stream parameter is always valid since it's boolean
    assert!(StreamingValidator::validate_stream(true).is_ok());
    assert!(StreamingValidator::validate_stream(false).is_ok());
}

#[test]
fn test_streaming_validator_model_support() {
    assert!(
        StreamingValidator::is_streaming_supported("gpt-4"),
        "gpt-4 should support streaming"
    );
    assert!(
        StreamingValidator::is_streaming_supported("llama-2-7b"),
        "llama-2-7b should support streaming"
    );
    assert!(
        StreamingValidator::is_streaming_supported("any-model"),
        "All models support streaming"
    );
}

#[test]
fn test_streaming_validator_empty_model() {
    assert!(
        !StreamingValidator::is_streaming_supported(""),
        "Empty model should not be supported"
    );
}

#[test]
fn test_streaming_event_openai_format_compatibility() {
    let event = ChatCompletionStreamEvent {
        id: "chatcmpl-8Pz6B0N1T2q9L7M6".to_string(),
        object: "text_completion.chunk".to_string(),
        created: 1700000000,
        model: "gpt-4".to_string(),
        choices: vec![StreamingChoice {
            index: 0,
            delta: StreamingDelta {
                content: Some(" world".to_string()),
                finish_reason: None,
            },
            finish_reason: None,
        }],
    };

    // Should be serializable to OpenAI format
    let json = serde_json::to_string(&event).expect("Should serialize");
    let value: serde_json::Value = serde_json::from_str(&json).expect("Should parse");

    assert!(value.get("id").is_some(), "Should have id field");
    assert!(value.get("object").is_some(), "Should have object field");
    assert!(value.get("created").is_some(), "Should have created field");
    assert!(value.get("model").is_some(), "Should have model field");
    assert!(value.get("choices").is_some(), "Should have choices field");
}

#[test]
fn test_streaming_multiple_tokens_sequence() {
    // Simulate streaming multiple tokens
    let tokens = vec!["Hello", " ", "world", "!"];
    let mut events = Vec::new();

    for (idx, token) in tokens.iter().enumerate() {
        let event = ChatCompletionStreamEvent {
            id: "test".to_string(),
            object: "text_completion.chunk".to_string(),
            created: 100 + idx as i64,
            model: "gpt-4".to_string(),
            choices: vec![StreamingChoice {
                index: 0,
                delta: StreamingDelta {
                    content: Some(token.to_string()),
                    finish_reason: None,
                },
                finish_reason: None,
            }],
        };
        events.push(event);
    }

    assert_eq!(events.len(), 4, "Should have 4 token events");
    assert_eq!(
        events[0].choices[0].delta.content,
        Some("Hello".to_string())
    );
    assert_eq!(events[3].choices[0].delta.content, Some("!".to_string()));
}

#[test]
fn test_streaming_content_accumulation() {
    // Test that we can accumulate tokens to form complete response
    let tokens = vec!["The", " ", "model", " ", "responded"];
    let mut accumulated = String::new();

    for token in &tokens {
        accumulated.push_str(token);
    }

    assert_eq!(
        accumulated, "The model responded",
        "Tokens should accumulate correctly"
    );
}

#[test]
fn test_streaming_empty_delta_finish() {
    // Finish event should have no content
    let delta = StreamingDelta {
        content: None,
        finish_reason: Some("length".to_string()),
    };

    let choice = StreamingChoice {
        index: 0,
        delta,
        finish_reason: Some("length".to_string()),
    };

    let event = ChatCompletionStreamEvent {
        id: "test".to_string(),
        object: "text_completion.chunk".to_string(),
        created: 100,
        model: "gpt-4".to_string(),
        choices: vec![choice],
    };

    assert!(
        event.choices[0].delta.content.is_none(),
        "Finish should have no content"
    );
    assert_eq!(event.choices[0].finish_reason, Some("length".to_string()));
}
