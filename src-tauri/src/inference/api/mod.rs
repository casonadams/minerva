//! Inference API Module
//!
//! REST/IPC interface for inference operations.
//! Split into focused sub-modules for maintainability.

pub mod handlers;
pub mod request_types;
pub mod response_types;
pub mod types;

pub use handlers::{infer_prompt, list_models, load_model, unload_model};
pub use types::{
    InferenceRequest, InferenceResponse, LoadModelRequest, ModelInfoResponse, ModelsResponse,
    TokenResponse,
};
