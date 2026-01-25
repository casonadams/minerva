use crate::error::MinervaResult;
use crate::models::{ChatCompletionRequest, ChatCompletionResponse, ChatMessage, Choice, Usage};
use axum::Json;
use uuid::Uuid;

pub async fn create_completion_response(
    req: ChatCompletionRequest,
) -> MinervaResult<Json<ChatCompletionResponse>> {
    let completion_id = format!("chatcmpl-{}", Uuid::new_v4());
    let created = chrono::Utc::now().timestamp();
    let prompt = build_chat_prompt(&req.messages);

    let response_content = format!(
        "Minerva inference response to: \"{}\" - Mock response for testing",
        prompt.chars().take(50).collect::<String>()
    );

    let prompt_tokens = estimate_tokens(&prompt);
    let completion_tokens = estimate_tokens(&response_content);

    Ok(Json(ChatCompletionResponse {
        id: completion_id,
        object: "chat.completion".to_string(),
        created,
        model: req.model,
        choices: vec![Choice {
            index: 0,
            message: ChatMessage {
                role: "assistant".to_string(),
                content: response_content,
            },
            finish_reason: "stop".to_string(),
        }],
        usage: Usage {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
        },
    }))
}

pub fn build_chat_prompt(messages: &[ChatMessage]) -> String {
    messages
        .iter()
        .map(|msg| format!("{}: {}", msg.role, msg.content))
        .collect::<Vec<_>>()
        .join("\n")
}

pub fn estimate_tokens(text: &str) -> usize {
    (text.len() / 4).max(1)
}
