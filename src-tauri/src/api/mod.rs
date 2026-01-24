//! API Protocol Layer
//! Ensures consistent OpenAI-compatible responses
//! Handles request validation and response envelope standardization

pub mod response;
pub mod types;
pub mod validator;

pub use types::{ApiError, ApiErrorResponse, ApiResponse, ResponseMetadata};
pub use validator::ProtocolValidator;
