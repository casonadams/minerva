/// GGUF Tensor Support for Real Model Loading (Phase 6)
///
/// This module provides comprehensive support for loading and managing GGUF tensors
/// with all quantization formats and data types.
use std::fmt;

/// Supported GGUF data types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(non_camel_case_types)]
pub enum GGUFDataType {
    F32,   // 32-bit floating point
    F16,   // 16-bit floating point (half precision)
    Q8_0,  // Quantized 8-bit (per-row)
    Q8_1,  // Quantized 8-bit with scale/zero (per-row)
    Q4_0,  // Quantized 4-bit (per-row)
    Q4_1,  // Quantized 4-bit with scale (per-row)
    Q5_0,  // Quantized 5-bit (per-row)
    Q5_1,  // Quantized 5-bit with scale (per-row)
    Q2_K,  // Quantized 2-bit with k-quant
    Q3_K,  // Quantized 3-bit with k-quant
    Q4_K,  // Quantized 4-bit with k-quant
    Q5_K,  // Quantized 5-bit with k-quant
    Q6_K,  // Quantized 6-bit with k-quant
    I8,    // 8-bit integer
    I16,   // 16-bit integer
    I32,   // 32-bit integer
    Count, // Internal use only
}

impl GGUFDataType {
    /// Get the size in bytes of a single element of this type
    pub fn element_size(&self) -> usize {
        match self {
            GGUFDataType::F32 => 4,
            GGUFDataType::F16 => 2,
            GGUFDataType::Q8_0 => 1,
            GGUFDataType::Q8_1 => 1,
            GGUFDataType::Q4_0 => 1,
            GGUFDataType::Q4_1 => 1,
            GGUFDataType::Q5_0 => 1,
            GGUFDataType::Q5_1 => 1,
            GGUFDataType::Q2_K => 1,
            GGUFDataType::Q3_K => 1,
            GGUFDataType::Q4_K => 1,
            GGUFDataType::Q5_K => 1,
            GGUFDataType::Q6_K => 1,
            GGUFDataType::I8 => 1,
            GGUFDataType::I16 => 2,
            GGUFDataType::I32 => 4,
            GGUFDataType::Count => 0,
        }
    }

    /// Calculate total size in bytes for a tensor of this type with given element count
    pub fn total_size(&self, element_count: usize) -> usize {
        match self {
            // Quantized types store multiple values in fewer bytes
            // Q4_0: 32 values in 18 bytes (2 scale bytes + 16 data bytes)
            GGUFDataType::Q4_0 => element_count.div_ceil(32) * 18,
            // Q4_1: 32 values in 20 bytes (4 scale bytes + 16 data bytes)
            GGUFDataType::Q4_1 => element_count.div_ceil(32) * 20,
            // Q5_0: 32 values in 22 bytes (2 scale bytes + 20 data bytes)
            GGUFDataType::Q5_0 => element_count.div_ceil(32) * 22,
            // Q5_1: 32 values in 24 bytes (8 scale bytes + 16 data bytes)
            GGUFDataType::Q5_1 => element_count.div_ceil(32) * 24,
            // Q8_0: 32 values in 18 bytes (2 scale bytes + 16 data bytes)
            GGUFDataType::Q8_0 => element_count.div_ceil(32) * 18,
            // Q8_1: 32 values in 20 bytes (4 scale bytes + 16 data bytes)
            GGUFDataType::Q8_1 => element_count.div_ceil(32) * 20,
            _ => element_count * self.element_size(),
        }
    }

    /// Convert type number to enum
    pub fn from_u32(value: u32) -> Option<Self> {
        match value {
            0 => Some(GGUFDataType::F32),
            1 => Some(GGUFDataType::F16),
            2 => Some(GGUFDataType::Q8_0),
            3 => Some(GGUFDataType::Q8_1),
            4 => Some(GGUFDataType::Q4_0),
            5 => Some(GGUFDataType::Q4_1),
            6 => Some(GGUFDataType::Q5_0),
            7 => Some(GGUFDataType::Q5_1),
            8 => Some(GGUFDataType::Q2_K),
            9 => Some(GGUFDataType::Q3_K),
            10 => Some(GGUFDataType::Q4_K),
            11 => Some(GGUFDataType::Q5_K),
            12 => Some(GGUFDataType::Q6_K),
            13 => Some(GGUFDataType::I8),
            14 => Some(GGUFDataType::I16),
            15 => Some(GGUFDataType::I32),
            _ => None,
        }
    }
}

impl fmt::Display for GGUFDataType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GGUFDataType::F32 => write!(f, "F32"),
            GGUFDataType::F16 => write!(f, "F16"),
            GGUFDataType::Q8_0 => write!(f, "Q8_0"),
            GGUFDataType::Q8_1 => write!(f, "Q8_1"),
            GGUFDataType::Q4_0 => write!(f, "Q4_0"),
            GGUFDataType::Q4_1 => write!(f, "Q4_1"),
            GGUFDataType::Q5_0 => write!(f, "Q5_0"),
            GGUFDataType::Q5_1 => write!(f, "Q5_1"),
            GGUFDataType::Q2_K => write!(f, "Q2_K"),
            GGUFDataType::Q3_K => write!(f, "Q3_K"),
            GGUFDataType::Q4_K => write!(f, "Q4_K"),
            GGUFDataType::Q5_K => write!(f, "Q5_K"),
            GGUFDataType::Q6_K => write!(f, "Q6_K"),
            GGUFDataType::I8 => write!(f, "I8"),
            GGUFDataType::I16 => write!(f, "I16"),
            GGUFDataType::I32 => write!(f, "I32"),
            GGUFDataType::Count => write!(f, "Count"),
        }
    }
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

impl GGUFTensor {
    /// Create a new tensor
    pub fn new(name: String, data_type: GGUFDataType, shape: Vec<u64>, data: Vec<u8>) -> Self {
        Self {
            name,
            data_type,
            shape,
            data,
        }
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
    fn test_data_type_conversion() {
        assert_eq!(GGUFDataType::from_u32(0), Some(GGUFDataType::F32));
        assert_eq!(GGUFDataType::from_u32(1), Some(GGUFDataType::F16));
        assert_eq!(GGUFDataType::from_u32(2), Some(GGUFDataType::Q8_0));
        assert_eq!(GGUFDataType::from_u32(255), None);
    }

    #[test]
    fn test_element_size() {
        assert_eq!(GGUFDataType::F32.element_size(), 4);
        assert_eq!(GGUFDataType::F16.element_size(), 2);
        assert_eq!(GGUFDataType::I32.element_size(), 4);
        assert_eq!(GGUFDataType::I16.element_size(), 2);
        assert_eq!(GGUFDataType::I8.element_size(), 1);
    }

    #[test]
    fn test_total_size_unquantized() {
        assert_eq!(GGUFDataType::F32.total_size(1000), 4000);
        assert_eq!(GGUFDataType::F16.total_size(1000), 2000);
        assert_eq!(GGUFDataType::I32.total_size(100), 400);
    }

    #[test]
    fn test_total_size_quantized() {
        // Q4_0: 32 values in 18 bytes
        assert_eq!(GGUFDataType::Q4_0.total_size(32), 18);
        assert_eq!(GGUFDataType::Q4_0.total_size(64), 36);
        // Q8_0: 32 values in 18 bytes
        assert_eq!(GGUFDataType::Q8_0.total_size(32), 18);
    }

    #[test]
    fn test_tensor_creation() {
        let tensor = GGUFTensor::new(
            "test.weight".to_string(),
            GGUFDataType::F32,
            vec![10, 20],
            vec![0u8; 800],
        );

        assert_eq!(tensor.name, "test.weight");
        assert_eq!(tensor.element_count(), 200);
        assert_eq!(tensor.total_bytes(), 800);
        assert!(tensor.is_valid());
    }

    #[test]
    fn test_tensor_shape_string() {
        let tensor = GGUFTensor::new(
            "weight".to_string(),
            GGUFDataType::F32,
            vec![4096, 32000],
            vec![0u8; 1],
        );

        assert_eq!(tensor.shape_str(), "4096x32000");
    }

    #[test]
    fn test_tensor_invalid_size() {
        let tensor = GGUFTensor::new(
            "test.weight".to_string(),
            GGUFDataType::F32,
            vec![10, 20],
            vec![0u8; 100], // Wrong size, should be 800
        );

        assert!(!tensor.is_valid());
        assert_eq!(tensor.total_bytes(), 100);
        assert_eq!(tensor.expected_bytes(), 800);
    }

    #[test]
    fn test_data_type_display() {
        assert_eq!(GGUFDataType::F32.to_string(), "F32");
        assert_eq!(GGUFDataType::Q4_0.to_string(), "Q4_0");
        assert_eq!(GGUFDataType::I32.to_string(), "I32");
    }
}
