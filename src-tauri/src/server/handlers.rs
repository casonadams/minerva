use axum::{Json, response::IntoResponse};
use crate::error::MinervaResult;
use crate::models::{
    ChatCompletionRequest, ChatCompletionResponse, Choice,
    ChatMessage, Usage,
};
use axum::http::HeaderMap;
use uuid::Uuid;
use crate::server::ServerState;
use crate::middleware::Validator;

pub async fn list_models(
    axum::extract::State(state): axum::extract::State<ServerState>,
) -> MinervaResult<Json<crate::models::ModelsListResponse>> {
    let registry = state.model_registry.lock().await;
    let models = registry.list_models();

    Ok(Json(crate::models::ModelsListResponse {
        object: "list".to_string(),
        data: models,
    }))
}

pub async fn chat_completions(
    axum::extract::State(state): axum::extract::State<ServerState>,
    headers: HeaderMap,
    Json(req): Json<ChatCompletionRequest>,
) -> MinervaResult<axum::response::Response> {
    let client_id = headers
        .get("x-client-id")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("anonymous");

    Validator::model_id(&req.model)?;
    let prompt = build_chat_prompt(&req.messages);
    Validator::prompt(&prompt, 2000)?;

    if let Some(temp) = req.temperature {
        Validator::temperature(temp)?;
    }
    if let Some(tp) = req.top_p {
        Validator::top_p(tp)?;
    }

    if !state.rate_limiter.allow_request(client_id, 1.0).await {
        let retry = state.rate_limiter.retry_after(client_id, 1.0).await;
        return Err(crate::error::MinervaError::InvalidRequest(format!(
            "Rate limit exceeded. Retry after {} seconds",
            retry
        )));
    }

    let registry = state.model_registry.lock().await;
    registry
        .get_model(&req.model)
        .ok_or_else(|| crate::error::MinervaError::ModelNotFound(
            format!("Model '{}' not found", req.model)
        ))?;

    let is_streaming = req.stream.unwrap_or(false);

    if is_streaming {
        Ok(create_streaming_response(req).into_response())
    } else {
        Ok(create_completion_response(req).await?.into_response())
    }
}

async fn create_completion_response(
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

fn build_chat_prompt(messages: &[ChatMessage]) -> String {
    messages
        .iter()
        .map(|msg| format!("{}: {}", msg.role, msg.content))
        .collect::<Vec<_>>()
        .join("\n")
}

fn estimate_tokens(text: &str) -> usize {
    (text.len() / 4).max(1)
}

fn create_streaming_response(
    req: ChatCompletionRequest,
) -> axum::response::sse::Sse<
    futures::stream::Iter<std::vec::IntoIter<Result<axum::response::sse::Event, String>>>,
> {
    use axum::response::sse::{Event, KeepAlive};
    use futures::stream;

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
    let chunks: Vec<_> = tokens
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
                        role: if is_first { Some("assistant".to_string()) } else { None },
                        content: Some(token),
                    },
                    finish_reason: if is_last { Some("stop".to_string()) } else { None },
                }],
            };

            Ok::<_, String>(Event::default().json_data(chunk).unwrap())
        })
        .collect();

    let stream_iter = stream::iter(chunks);
    axum::response::sse::Sse::new(stream_iter).keep_alive(KeepAlive::default())
}
