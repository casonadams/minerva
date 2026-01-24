use super::chat::build_chat_prompt;
use crate::error::MinervaResult;
use crate::middleware::Validator;
use crate::models::ChatCompletionRequest;

pub fn validate_chat_request(req: &ChatCompletionRequest) -> MinervaResult<()> {
    Validator::model_id(&req.model)?;
    let prompt = build_chat_prompt(&req.messages);
    Validator::prompt(&prompt, 2000)?;

    if let Some(temp) = req.temperature {
        Validator::temperature(temp)?;
    }
    if let Some(tp) = req.top_p {
        Validator::top_p(tp)?;
    }

    Ok(())
}
