// Streaming Handler Tests - Integration with API handlers

use minerva_lib::streaming::{
    ChatCompletionStreamEvent, StreamingChoice, StreamingDelta, create_streaming_events,
    format_streaming_event,
};

#[test]
fn test_create_streaming_events_basic() {
    let tokens = vec!["hello".to_string(), " ".to_string(), "world".to_string()];
    let events = create_streaming_events("req-123".to_string(), "gpt-4".to_string(), tokens);

    // Should have 3 token events + 1 finish event
    assert_eq!(events.len(), 4, "Should have 4 events total");
}

#[test]
fn test_create_streaming_events_single_token() {
    let tokens = vec!["response".to_string()];
    let events = create_streaming_events("req-1".to_string(), "model".to_string(), tokens);

    assert_eq!(events.len(), 2, "Should have 1 token + 1 finish event");
}

#[test]
fn test_create_streaming_events_empty_tokens() {
    let tokens: Vec<String> = vec![];
    let events = create_streaming_events("req-1".to_string(), "model".to_string(), tokens);

    // Should still have finish event
    assert_eq!(events.len(), 1, "Should have 1 finish event");
}

#[test]
fn test_streaming_events_model_preserved() {
    let tokens = vec!["test".to_string()];
    let events = create_streaming_events("req-1".to_string(), "gpt-4-turbo".to_string(), tokens);

    // Check model is in both token and finish events
    assert_eq!(
        events[0].model, "gpt-4-turbo",
        "Token event should have model"
    );
    assert_eq!(
        events[1].model, "gpt-4-turbo",
        "Finish event should have model"
    );
}

#[test]
fn test_streaming_events_id_preserved() {
    let tokens = vec!["test".to_string()];
    let request_id = "chatcmpl-abc123".to_string();
    let events = create_streaming_events(request_id.clone(), "model".to_string(), tokens);

    assert_eq!(
        events[0].id, request_id,
        "Token event should have request ID"
    );
    assert_eq!(
        events[1].id, request_id,
        "Finish event should have request ID"
    );
}

#[test]
fn test_streaming_events_finish_event_structure() {
    let tokens = vec!["test".to_string()];
    let events = create_streaming_events("req-1".to_string(), "model".to_string(), tokens);

    let finish_event = &events[events.len() - 1];
    assert!(
        finish_event.choices[0].delta.content.is_none(),
        "Finish event should have no content"
    );
    assert_eq!(
        finish_event.choices[0].finish_reason,
        Some("stop".to_string()),
        "Finish event should have stop reason"
    );
}

#[test]
fn test_format_streaming_event_serialization() {
    let event = ChatCompletionStreamEvent {
        id: "test-id".to_string(),
        object: "text_completion.chunk".to_string(),
        created: 100,
        model: "gpt-4".to_string(),
        choices: vec![StreamingChoice {
            index: 0,
            delta: StreamingDelta {
                content: Some("data".to_string()),
                finish_reason: None,
            },
            finish_reason: None,
        }],
    };

    let json = format_streaming_event(&event);
    assert!(!json.is_empty(), "Should produce JSON");
    assert!(json.contains("test-id"), "Should contain event ID");
}

#[test]
fn test_streaming_events_token_content_preserved() {
    let tokens = vec![
        "The".to_string(),
        " ".to_string(),
        "model".to_string(),
        " ".to_string(),
        "responded".to_string(),
    ];
    let events = create_streaming_events("req-1".to_string(), "model".to_string(), tokens.clone());

    // Check each token is preserved
    for (i, token) in tokens.iter().enumerate() {
        assert_eq!(
            events[i].choices[0].delta.content,
            Some(token.clone()),
            "Token {} should be preserved",
            i
        );
    }
}

#[test]
fn test_streaming_events_timestamps_set() {
    let tokens = vec!["test".to_string()];
    let events = create_streaming_events("req-1".to_string(), "model".to_string(), tokens);

    for event in events {
        assert!(event.created > 0, "Event should have timestamp");
    }
}

#[test]
fn test_streaming_event_object_type() {
    let tokens = vec!["test".to_string()];
    let events = create_streaming_events("req-1".to_string(), "model".to_string(), tokens);

    for event in &events {
        assert_eq!(
            event.object, "text_completion.chunk",
            "Event object type should be text_completion.chunk"
        );
    }
}

#[test]
fn test_streaming_event_choice_index() {
    let tokens = vec!["test".to_string()];
    let events = create_streaming_events("req-1".to_string(), "model".to_string(), tokens);

    for event in &events {
        assert_eq!(event.choices[0].index, 0, "Choice index should be 0");
    }
}

#[test]
fn test_streaming_accumulation_from_events() {
    let tokens = vec!["Hello".to_string(), " ".to_string(), "World".to_string()];
    let events = create_streaming_events("req-1".to_string(), "model".to_string(), tokens);

    // Accumulate content from events (excluding finish)
    let mut accumulated = String::new();
    for event in events.iter().take(events.len() - 1) {
        if let Some(content) = &event.choices[0].delta.content {
            accumulated.push_str(content);
        }
    }

    assert_eq!(
        accumulated, "Hello World",
        "Should accumulate to complete message"
    );
}

#[test]
fn test_format_streaming_event_with_finish() {
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

    let json = format_streaming_event(&event);
    let value: serde_json::Value = serde_json::from_str(&json).expect("Should parse");

    assert!(
        value["choices"][0]["finish_reason"].is_string(),
        "Should have finish_reason"
    );
    assert_eq!(value["choices"][0]["finish_reason"], "stop");
}

#[test]
fn test_streaming_large_token_sequence() {
    // Test with many tokens
    let tokens: Vec<String> = (0..100).map(|i| format!("token{} ", i)).collect();

    let events = create_streaming_events("req-1".to_string(), "model".to_string(), tokens.clone());

    assert_eq!(events.len(), 101, "Should have 100 tokens + 1 finish");
}

#[test]
fn test_streaming_special_characters_in_tokens() {
    let tokens = vec![
        "Hello".to_string(),
        " \"quoted\"".to_string(),
        " \n".to_string(),
        " \\escaped\\".to_string(),
    ];

    let events = create_streaming_events("req-1".to_string(), "model".to_string(), tokens.clone());

    // All tokens should be preserved exactly
    for (i, token) in tokens.iter().enumerate() {
        assert_eq!(events[i].choices[0].delta.content, Some(token.clone()));
    }
}
