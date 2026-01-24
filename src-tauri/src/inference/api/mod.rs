/// Inference API Module
///
/// REST/IPC interface for inference operations.
/// Split into focused sub-modules for maintainability.

pub mod types;
pub mod handlers;

pub use types::{
    InferenceRequest, InferenceResponse, TokenResponse,
    LoadModelRequest, ModelInfoResponse, ModelsResponse,
};
pub use handlers::{infer_prompt, load_model, list_models, unload_model};
