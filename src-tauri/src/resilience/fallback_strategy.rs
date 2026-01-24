use crate::error::MinervaError;

/// Fallback strategy options
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FallbackStrategy {
    /// Use CPU instead of GPU
    GpuToCpu,
    /// Use smaller/faster model
    UseAltModel,
    /// Switch from streaming to batch
    StreamingToBatch,
    /// Reduce batch size and retry
    ReduceBatchSize,
    /// Use cached result if available
    UseCache,
    /// No fallback available
    None,
}

/// Fallback decision for errors
pub struct FallbackDecision;

impl FallbackDecision {
    /// Determine fallback strategy for an error
    pub fn strategy_for(error: &MinervaError) -> FallbackStrategy {
        match error {
            MinervaError::GpuOutOfMemory(_) | MinervaError::GpuContextLost(_) => {
                FallbackStrategy::GpuToCpu
            }
            MinervaError::OutOfMemory(_) => FallbackStrategy::ReduceBatchSize,
            MinervaError::ModelCorrupted(_) => FallbackStrategy::UseAltModel,
            MinervaError::StreamingError(_) => FallbackStrategy::StreamingToBatch,
            _ => FallbackStrategy::None,
        }
    }

    /// Get human-readable fallback message
    pub fn message(strategy: FallbackStrategy) -> &'static str {
        match strategy {
            FallbackStrategy::GpuToCpu => "GPU unavailable, switching to CPU inference...",
            FallbackStrategy::UseAltModel => "Primary model unavailable, trying alternate model...",
            FallbackStrategy::StreamingToBatch => {
                "Streaming failed, switching to batch inference..."
            }
            FallbackStrategy::ReduceBatchSize => "Reducing batch size due to memory constraints...",
            FallbackStrategy::UseCache => "Using cached result...",
            FallbackStrategy::None => "No fallback strategy available",
        }
    }

    /// Is fallback available?
    pub fn is_available(strategy: FallbackStrategy) -> bool {
        !matches!(strategy, FallbackStrategy::None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fallback_gpu_error() {
        let err = MinervaError::GpuOutOfMemory("16GB".to_string());
        let strategy = FallbackDecision::strategy_for(&err);
        assert_eq!(strategy, FallbackStrategy::GpuToCpu);
    }

    #[test]
    fn test_fallback_streaming_error() {
        let err = MinervaError::StreamingError("connection".to_string());
        let strategy = FallbackDecision::strategy_for(&err);
        assert_eq!(strategy, FallbackStrategy::StreamingToBatch);
    }

    #[test]
    fn test_fallback_memory_error() {
        let err = MinervaError::OutOfMemory("heap".to_string());
        let strategy = FallbackDecision::strategy_for(&err);
        assert_eq!(strategy, FallbackStrategy::ReduceBatchSize);
    }

    #[test]
    fn test_fallback_model_corrupted() {
        let err = MinervaError::ModelCorrupted("header".to_string());
        let strategy = FallbackDecision::strategy_for(&err);
        assert_eq!(strategy, FallbackStrategy::UseAltModel);
    }

    #[test]
    fn test_fallback_no_strategy() {
        let err = MinervaError::InvalidRequest("bad".to_string());
        let strategy = FallbackDecision::strategy_for(&err);
        assert_eq!(strategy, FallbackStrategy::None);
    }

    #[test]
    fn test_fallback_is_available() {
        assert!(FallbackDecision::is_available(FallbackStrategy::GpuToCpu));
        assert!(!FallbackDecision::is_available(FallbackStrategy::None));
    }

    #[test]
    fn test_fallback_messages() {
        let msg = FallbackDecision::message(FallbackStrategy::GpuToCpu);
        assert!(msg.contains("CPU"));

        let msg = FallbackDecision::message(FallbackStrategy::StreamingToBatch);
        assert!(msg.contains("batch"));
    }
}
