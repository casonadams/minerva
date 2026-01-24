pub use super::gguf_data_type::GGUFDataType;
/// GGUF Tensor Support for Real Model Loading (Phase 6)
///
/// This module provides comprehensive support for loading and managing GGUF tensors
/// with all quantization formats and data types.
use std::fmt;

/// Input for creating GGUF tensors
#[derive(Clone)]
pub struct GGUFTensorData {
    /// Tensor name (e.g., "token_embd.weight")
    pub name: String,
    /// Data type of tensor
    pub data_type: GGUFDataType,
    /// Tensor shape (dimensions)
    pub shape: Vec<u64>,
    /// Raw tensor data
    pub data: Vec<u8>,
}

/// GGUF tensor information and data
#[derive(Clone)]
pub struct GGUFTensor {
    /// Tensor name (e.g., "token_embd.weight")
    pub name: String,
    /// Data type of tensor
    pub data_type: GGUFDataType,
    /// Tensor shape (dimensions)
    pub shape: Vec<u64>,
    /// Raw tensor data
    pub data: Vec<u8>,
}

impl From<GGUFTensorData> for GGUFTensor {
    fn from(input: GGUFTensorData) -> Self {
        Self {
            name: input.name,
            data_type: input.data_type,
            shape: input.shape,
            data: input.data,
        }
    }
}

impl GGUFTensor {
    /// Create a new tensor from data
    pub fn new(input: GGUFTensorData) -> Self {
        Self::from(input)
    }

    /// Get total number of elements in tensor
    pub fn element_count(&self) -> u64 {
        self.shape.iter().product()
    }

    /// Get tensor shape as string (e.g., "4096x32000")
    pub fn shape_str(&self) -> String {
        self.shape
            .iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
            .join("x")
    }

    /// Get total size in bytes
    pub fn total_bytes(&self) -> usize {
        self.data.len()
    }

    /// Get expected size based on shape and data type
    pub fn expected_bytes(&self) -> usize {
        self.data_type.total_size(self.element_count() as usize)
    }

    /// Check if tensor data matches expected size
    pub fn is_valid(&self) -> bool {
        self.total_bytes() == self.expected_bytes()
    }
}

impl std::fmt::Debug for GGUFTensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GGUFTensor")
            .field("name", &self.name)
            .field("data_type", &self.data_type)
            .field("shape", &self.shape)
            .field("element_count", &self.element_count())
            .field("total_bytes", &self.total_bytes())
            .field("expected_bytes", &self.expected_bytes())
            .field("valid", &self.is_valid())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_creation() {
        let input = GGUFTensorData {
            name: "test.weight".to_string(),
            data_type: GGUFDataType::F32,
            shape: vec![10, 20],
            data: vec![0u8; 800],
        };
        let tensor = GGUFTensor::new(input);

        assert_eq!(tensor.name, "test.weight");
        assert_eq!(tensor.element_count(), 200);
        assert_eq!(tensor.total_bytes(), 800);
        assert!(tensor.is_valid());
    }

    #[test]
    fn test_tensor_shape_string() {
        let input = GGUFTensorData {
            name: "weight".to_string(),
            data_type: GGUFDataType::F32,
            shape: vec![4096, 32000],
            data: vec![0u8; 1],
        };
        let tensor = GGUFTensor::new(input);

        assert_eq!(tensor.shape_str(), "4096x32000");
    }

    #[test]
    fn test_tensor_invalid_size() {
        let input = GGUFTensorData {
            name: "test.weight".to_string(),
            data_type: GGUFDataType::F32,
            shape: vec![10, 20],
            data: vec![0u8; 100],
        };
        let tensor = GGUFTensor::new(input);

        assert!(!tensor.is_valid());
        assert_eq!(tensor.total_bytes(), 100);
        assert_eq!(tensor.expected_bytes(), 800);
    }
}
