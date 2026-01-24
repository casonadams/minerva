//! Streaming event handling

use super::types::{ChatCompletionStreamEvent, StreamingChoice, StreamingDelta};

/// Formats a single streaming event as JSON
pub fn format_streaming_event(event: &ChatCompletionStreamEvent) -> String {
    serde_json::to_string(event).unwrap_or_else(|_| "{}".to_string())
}

/// Creates a list of streaming events from tokens
pub fn create_streaming_events(
    request_id: String,
    model: String,
    tokens: Vec<String>,
) -> Vec<ChatCompletionStreamEvent> {
    let mut events = Vec::new();

    // Add token events
    for token in tokens {
        events.push(ChatCompletionStreamEvent {
            id: request_id.clone(),
            object: "text_completion.chunk".to_string(),
            created: chrono::Utc::now().timestamp(),
            model: model.clone(),
            choices: vec![StreamingChoice {
                index: 0,
                delta: StreamingDelta {
                    content: Some(token),
                    finish_reason: None,
                },
                finish_reason: None,
            }],
        });
    }

    // Add finish event
    events.push(ChatCompletionStreamEvent {
        id: request_id,
        object: "text_completion.chunk".to_string(),
        created: chrono::Utc::now().timestamp(),
        model,
        choices: vec![StreamingChoice {
            index: 0,
            delta: StreamingDelta {
                content: None,
                finish_reason: Some("stop".to_string()),
            },
            finish_reason: Some("stop".to_string()),
        }],
    });

    events
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chat_completion_stream_event_creation() {
        let event = ChatCompletionStreamEvent {
            id: "test-id".to_string(),
            object: "text_completion.chunk".to_string(),
            created: 123456789,
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

        assert_eq!(event.id, "test-id");
        assert_eq!(event.object, "text_completion.chunk");
        assert_eq!(event.model, "gpt-4");
        assert_eq!(event.choices.len(), 1);
    }

    #[test]
    fn test_streaming_event_serialization() {
        let event = ChatCompletionStreamEvent {
            id: "test".to_string(),
            object: "chunk".to_string(),
            created: 100,
            model: "test-model".to_string(),
            choices: vec![StreamingChoice {
                index: 0,
                delta: StreamingDelta {
                    content: Some("data".to_string()),
                    finish_reason: None,
                },
                finish_reason: None,
            }],
        };

        let json = serde_json::to_string(&event).expect("Should serialize");
        assert!(
            json.contains("test-id") || json.contains("test"),
            "Should contain ID"
        );
        assert!(json.contains("chunk"), "Should contain object type");
    }
}
