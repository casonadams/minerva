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
}

#[cfg(test)]
mod tests {
    use super::*;

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
        assert_eq!(GGUFDataType::Q4_0.total_size(32), 18);
        assert_eq!(GGUFDataType::Q4_0.total_size(64), 36);
        assert_eq!(GGUFDataType::Q8_0.total_size(32), 18);
    }
}
