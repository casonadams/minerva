use super::chat::build_chat_prompt;
use crate::models::ChatCompletionRequest;
use axum::response::sse::{Event, KeepAlive};
use futures::stream;
use uuid::Uuid;

pub fn create_streaming_response(
    req: ChatCompletionRequest,
) -> axum::response::sse::Sse<futures::stream::Iter<std::vec::IntoIter<Result<Event, String>>>> {
    let completion_id = format!("chatcmpl-{}", Uuid::new_v4());
    let created = chrono::Utc::now().timestamp();
    let model = req.model.clone();
    let prompt = build_chat_prompt(&req.messages);

    let response_content = format!(
        "Minerva inference response to: \"{}\" - Mock streaming response for testing",
        prompt.chars().take(50).collect::<String>()
    );

    let tokens: Vec<String> = response_content
        .split_whitespace()
        .map(|w| format!("{} ", w))
        .collect();

    let token_count = tokens.len();
    let chunks = build_stream_chunks(tokens, token_count, completion_id, created, model);

    let stream_iter = stream::iter(chunks);
    axum::response::sse::Sse::new(stream_iter).keep_alive(KeepAlive::default())
}

fn build_stream_chunks(
    tokens: Vec<String>,
    token_count: usize,
    completion_id: String,
    created: i64,
    model: String,
) -> Vec<Result<Event, String>> {
    tokens
        .into_iter()
        .enumerate()
        .map(|(idx, token)| {
            let is_first = idx == 0;
            let is_last = idx == token_count - 1;

            let chunk = crate::models::ChatCompletionChunk {
                id: completion_id.clone(),
                object: "chat.completion.chunk".to_string(),
                created,
                model: model.clone(),
                choices: vec![crate::models::ChoiceDelta {
                    index: 0,
                    delta: crate::models::DeltaMessage {
                        role: if is_first {
                            Some("assistant".to_string())
                        } else {
                            None
                        },
                        content: Some(token),
                    },
                    finish_reason: if is_last {
                        Some("stop".to_string())
                    } else {
                        None
                    },
                }],
            };

            Ok::<_, String>(Event::default().json_data(chunk).unwrap())
        })
        .collect()
}
