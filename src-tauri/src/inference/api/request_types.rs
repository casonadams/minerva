/// API Request Types
///
/// Lean request structures for inference API.
/// Follows OpenAI API format for compatibility.
use serde::{Deserialize, Serialize};

/// Inference request (prompt completion)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceRequest {
    pub model: String,
    pub prompt: String,
    #[serde(default)]
    pub max_tokens: Option<usize>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_k: Option<usize>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub seed: Option<u64>,
}

/// Model load request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadModelRequest {
    pub model_id: String,
    pub model_dir: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_inference_request_deserialize() {
        let json = r#"{
            "model": "llama",
            "prompt": "what is rust?",
            "max_tokens": 100,
            "temperature": 0.7
        }"#;
        let req: InferenceRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.model, "llama");
        assert_eq!(req.prompt, "what is rust?");
        assert_eq!(req.max_tokens, Some(100));
    }
}
