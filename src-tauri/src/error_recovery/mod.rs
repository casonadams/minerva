//! Error Recovery Strategies
//!
//! This module provides recovery mechanisms for common failure scenarios:
//! - GPU out of memory → fallback to CPU
//! - GPU context loss → reinitialize
//! - Model corruption → validation and reload
//! - Streaming errors → retry mechanism

pub mod handler;
pub mod types;

pub use handler::ErrorRecovery;
pub use types::RecoveryStrategy;
